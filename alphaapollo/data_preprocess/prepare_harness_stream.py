# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build the fixed AIME problem streams the three Task C arms share.

Two files, split strictly by year -- never shuffled and then split, which the assignment rules
out explicitly:

- **adaptation**: ``gneubig/aime-1983-2024``, years 2018-2022. 150 problems, of which one is
  dropped (see ``normalise_aime_row``), giving 149 -- inside the assignment's 80-150 band.
- **held-out**: ``MathArena/aime_2025``. One complete, recent year that no skill update ever
  sees, and the only source here that ships human ``problem_type`` annotations.

Design points worth stating, because each was a decision rather than a default:

**Year order, not topic-interleaved.** The assignment permits year order, topic-segmented, or
topic-interleaved, and "problems arrive in sequence" reads most naturally as time order. An
earlier draft interleaved by topic to shorten the gap between minting a skill and getting a
chance to reuse it, but AIME mixes all four topics within a single year's 30 problems, so a topic
recurs every few problems anyway -- and interleaving would have required topic labels *before*
the run, which is precisely the dependency the method turned out not to have.

**Topic labels are analysis metadata, not method input.** Since ``reflect`` names its own topics
and ``selector`` retrieves without a topic filter, nothing in the method reads the ``topic`` field
of a problem. It exists so the write-up can break results down per topic -- a required result for
*all three* arms, hence labelled once here and shared byte-identically rather than derived per
arm. Labelling costs one model call per unlabelled problem, reads only the question, and is
accounted under the ``offline_labeling`` management role.

**The stream file satisfies two readers.** ``harness.loader.load_stream`` reads the ``extra_info``
dict; upstream's ``run_problem`` indexes ``question``/``ground_truth``/``gt_traj``/``data_source``
off the problem unconditionally. The rows written here keep upstream's own column layout
(``prepare_evolving_data.py``) so either consumer works.

Usage::

    python -m alphaapollo.data_preprocess.prepare_harness_stream \\
        --out_dir ./data/harness --label_model qwen3-32b
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from alphaapollo.core.harness.topic import classify_topic

logger = logging.getLogger(__name__)

ABILITY = "math"

# AIME answers are integers 0-999, and the scorer compares them as strings. Anything else is not
# a usable label: see `normalise_aime_row`.
ANSWER_RE = re.compile(r"\d{1,3}")

# I before II. AIME has exactly two contests per year; a plain string sort happens to order these
# two correctly, but `number` does not ("10" < "2"), which is why ordering goes through
# `_sort_key` rather than sorting the raw strings.
_CONTEST_ORDER = {"I": 0, "II": 1}

_PROMPT_TEMPLATE = "{question}"


def _build_prompt(question: str) -> list[dict]:
    """Upstream's `prompt` column shape (a chat-style list), kept identical so a stream file
    written here is readable by the rest of the repo's pipeline, not only by this project."""
    return [{"role": "user", "content": _PROMPT_TEMPLATE.format(question=question)}]


def _sort_key(row: dict) -> tuple:
    return (int(row.get("year") or 0), _CONTEST_ORDER.get(str(row.get("contest")), 9), int(row.get("number") or 0))


def normalise_aime_row(raw: dict, *, data_source: str) -> dict | None:
    """Turn one source row into a stream problem, or ``None`` if it is unusable.

    Rows are dropped rather than repaired when the answer is not a plain AIME integer. The real
    case this exists for is ``2022-II-8``, whose published answer is
    ``"080 or 081 (both were accepted)"``. The scorer compares answer strings, so keeping it would
    mark every attempt on that problem wrong for every arm -- not a fair-but-hard problem, just a
    silently unscoreable one. One row out of 150 is a cheaper loss than an unscoreable problem
    sitting in the middle of the adaptation stream.
    """
    question = str(raw.get("Question") or raw.get("question") or raw.get("problem") or "").strip()
    if not question:
        return None

    answer = str(raw.get("Answer") or raw.get("answer") or "").strip()
    if not ANSWER_RE.fullmatch(answer):
        return None

    number_raw = raw.get("Problem Number", raw.get("problem_idx", 0))
    try:
        number = int(str(number_raw).strip())
    except (TypeError, ValueError):
        number = 0

    return {
        "question": question,
        "ground_truth": answer,
        # Upstream reads `gt_traj` unconditionally. These sources ship no worked solution, so it
        # is blank -- never the answer, which would hand Reflect a ground-truth channel.
        "gt_traj": "",
        "data_source": data_source,
        "topic": str(raw.get("topic") or "").strip(),
        "problem_shape": "",
        "technique": "",
        "year": int(raw.get("Year") or raw.get("year") or 0),
        "contest": str(raw.get("Part") or raw.get("contest") or "").strip(),
        "number": number,
        "source_id": str(raw.get("ID") or ""),
    }


def build_rows(raw_rows: list[dict], *, data_source: str, agent=None) -> list[dict]:
    """Normalise, de-duplicate, order, label and index a source split into stream problems.

    De-duplication is by question text and keeps the earliest occurrence: a repeated problem
    would be solved twice under different harness states, making its per-problem record ambiguous
    and double-counting it in the success rate.

    ``problem_idx`` is assigned **after** ordering, so it is the position in the stream the arms
    actually walk -- the index every log line and every ``p_<idx>`` evidence tag refers to.
    """
    rows: list[dict] = []
    seen_questions: set[str] = set()
    for raw in raw_rows:
        row = normalise_aime_row(raw, data_source=data_source)
        if row is None:
            continue
        if row["question"] in seen_questions:
            logger.info("dropping duplicate question from %s", row.get("source_id"))
            continue
        seen_questions.add(row["question"])
        rows.append(row)

    rows.sort(key=_sort_key)

    if agent is not None:
        cache: dict[str, str] = {}
        for row in rows:
            if row["topic"]:
                continue
            question = row["question"]
            if question not in cache:
                cache[question] = classify_topic(agent, question) or ""
            row["topic"] = cache[question]

    for idx, row in enumerate(rows):
        row["problem_idx"] = idx
    return rows


def write_stream(rows: list[dict], path: str | Path) -> Path:
    """Write ``rows`` as a parquet file readable by both ``load_stream`` and upstream's loader.

    Raises on an empty stream rather than writing a valid-but-dead file: every arm would then
    score 0/0 and the run would look completed.
    """
    if not rows:
        raise ValueError("refusing to write an empty stream: every arm would score 0/0 and the run would look complete")

    records: list[dict[str, Any]] = []
    for row in rows:
        extra_info = {k: v for k, v in row.items()}
        records.append({
            "data_source": row["data_source"],
            "prompt": _build_prompt(row["question"]),
            "ability": ABILITY,
            "reward_model": {"style": "rule", "ground_truth": row["ground_truth"]},
            "extra_info": extra_info,
            "metadata": None,
            "env_kwargs": {
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "gt_traj": row["gt_traj"],
                "data_source": row["data_source"],
            },
        })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path)
    return path


def _load_adaptation(first_year: int, last_year: int) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset("gneubig/aime-1983-2024")["train"]
    return [r for r in dataset if first_year <= int(r["Year"]) <= last_year]


def _load_heldout() -> list[dict]:
    from datasets import load_dataset

    rows = []
    for r in load_dataset("MathArena/aime_2025")["train"]:
        types = r.get("problem_type") or []
        # Multi-label rows exist (2 of 30); the first label is taken as the primary bucket, which
        # is what the per-topic breakdown needs. The full list stays out of the stream: a problem
        # counted under two topics would double-count in the breakdown.
        primary = str(types[0]).strip().lower().replace(" ", "_") if types else ""
        rows.append({"question": r["problem"], "answer": r["answer"], "topic": primary,
                     "year": 2025, "contest": "I", "problem_idx": r.get("problem_idx", 0),
                     "ID": f"2025-{r.get('problem_idx')}"})
    return rows


def main(out_dir: str = "./data/harness", label_model: str = "", base_url: str = "",
         first_year: int = 2018, last_year: int = 2022) -> None:
    """``python -m alphaapollo.data_preprocess.prepare_harness_stream --out_dir ./data/harness``

    ``label_model`` is optional: without it the ``topic`` column is left blank and every other
    field is still correct, so the streams are usable immediately and the (purely analytical)
    labels can be filled in by a later pass.
    """
    logging.basicConfig(level=logging.INFO)

    agent = None
    if label_model:
        from alphaapollo.core.generation.evolving.utils.agent import Agent
        agent = Agent({"model_name": label_model, "base_url": base_url, "api_key": "",
                       "temperature": 0.0, "max_tokens": 512})

    out = Path(out_dir)
    adaptation = build_rows(_load_adaptation(first_year, last_year),
                            data_source="gneubig/aime-1983-2024", agent=agent)
    heldout = build_rows(_load_heldout(), data_source="MathArena/aime_2025", agent=agent)

    write_stream(adaptation, out / "adaptation.parquet")
    write_stream(heldout, out / "heldout.parquet")

    def summarise(name: str, rows: list[dict]) -> None:
        from collections import Counter
        topics = Counter(r["topic"] or "(unlabelled)" for r in rows)
        years = Counter(r["year"] for r in rows)
        logger.info("%s: %d problems | years=%s | topics=%s", name, len(rows),
                    dict(sorted(years.items())), dict(sorted(topics.items())))

    summarise("adaptation", adaptation)
    summarise("held-out", heldout)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
