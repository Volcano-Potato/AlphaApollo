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
"""Turn a finished run's ``SkillStore`` into the deliverable artifacts: a readable harness, a
machine-readable summary, and the full evolution log.

Task A requires the mechanism to "export the final harness plus its evolution log". The store
already persists everything in durable form (``skills/*.md`` plus two JSONL logs), so this module
adds no new source of truth -- it is a *view*. Its value is that the numbers the write-up has to
report (harness size and growth, per-skill usage frequency, injected-context token cost,
accept/reject counts) are computed once, here, from the logs, rather than re-derived by hand for
each table and each slide.

Three files are written:

- ``harness.md`` -- the final skill set, split into the two layers, with each skill's usage
  record. This is the artifact a reader actually reads.
- ``summary.json`` -- the same content as aggregates, for the results tables.
- ``evolution.jsonl`` -- a verbatim copy of ``harness_log.jsonl``, so the exported bundle is
  self-contained and can be audited without the original run directory.

``build_report`` is kept separate from ``export_harness`` and touches no disk, so the numbers can
be asserted directly and reused without a filesystem round-trip.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from alphaapollo.core.harness.store import SkillStore


def _read_jsonl(path: Path) -> list[dict]:
    """Tolerant JSONL read: a run killed mid-write can leave a truncated final line, and losing
    the whole export over one partial record would be worse than skipping it."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def build_report(store: SkillStore) -> dict[str, Any]:
    """Aggregate a store's on-disk state and logs into the reportable numbers. Pure: reads only.

    Usage is joined from the selection log rather than read off each ``Skill``'s own counters,
    because a skill deleted by a later curator cycle no longer exists to hold a counter while its
    earlier selections still happened and still cost tokens. Those skills appear here with
    ``present_in_final_harness: False`` -- dropping them would understate the harness's true
    injected-context cost.
    """
    skills = store.all()
    edits = _read_jsonl(store.harness_log)
    selections = _read_jsonl(store.selection_log)

    live_ids = {skill.id for skill in skills}
    usage: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n_selected": 0, "n_selected_success": 0, "present_in_final_harness": False}
    )
    for record in selections:
        success = bool(record.get("success"))
        for skill_id in record.get("skill_ids") or []:
            entry = usage[skill_id]
            entry["n_selected"] += 1
            entry["n_selected_success"] += int(success)
            entry["present_in_final_harness"] = skill_id in live_ids

    injected = [int(r.get("n_tokens", 0) or 0) for r in selections]
    by_op: dict[str, int] = defaultdict(int)
    for record in edits:
        by_op[str(record.get("op"))] += 1

    topics: dict[str, int] = defaultdict(int)
    for skill in skills:
        if skill.level != "general":
            topics[str(skill.topic)] += 1

    return {
        "skills": {
            "n_general": sum(1 for s in skills if s.level == "general"),
            "n_topic": sum(1 for s in skills if s.level != "general"),
            "n_topic_by_topic": dict(sorted(topics.items())),
            "total_tokens": sum(s.n_tokens for s in skills),
        },
        "edits": {
            "total": len(edits),
            "accepted": sum(1 for r in edits if r.get("accepted")),
            "rejected": sum(1 for r in edits if not r.get("accepted")),
            "by_op": dict(sorted(by_op.items())),
        },
        "selection": {
            "n_problems": len(selections),
            "total_injected_tokens": sum(injected),
            "mean_injected_tokens": (sum(injected) / len(injected)) if injected else 0.0,
        },
        "usage": {k: dict(v) for k, v in sorted(usage.items())},
    }


def _render_markdown(store: SkillStore, report: dict[str, Any]) -> str:
    skills = store.all()
    usage = report["usage"]

    lines = [
        "# Cross-problem skill harness",
        "",
        f"{report['skills']['n_general']} general skills, {report['skills']['n_topic']} topic "
        f"skills, {report['skills']['total_tokens']} tokens total.",
        "",
        f"Compiled over {report['selection']['n_problems']} problems from "
        f"{report['edits']['total']} curator decisions "
        f"({report['edits']['accepted']} accepted, {report['edits']['rejected']} rejected).",
        "",
    ]

    def render_skill(skill) -> list[str]:
        stats = usage.get(skill.id, {})
        n_selected = stats.get("n_selected", 0)
        n_success = stats.get("n_selected_success", 0)
        return [
            f"### `{skill.id}` {skill.name}",
            "",
            f"*Selected {n_selected}x, {n_success} of those solved. "
            f"{skill.n_tokens} tokens. Evidence: {', '.join(skill.evidence) or 'none'}.*",
            "",
            f"**When to use.** {skill.trigger}",
            "",
            f"**Strategy.** {skill.lesson}",
            "",
            f"**Avoid.** {skill.failure_mode}",
            "",
        ]

    lines += ["## General skills", ""]
    general = [s for s in skills if s.level == "general"]
    if not general:
        lines += ["*(none)*", ""]
    for skill in general:
        lines += render_skill(skill)

    lines += ["## Topic skills", ""]
    topic_skills = [s for s in skills if s.level != "general"]
    if not topic_skills:
        lines += ["*(none)*", ""]
    by_topic: dict[str, list] = defaultdict(list)
    for skill in topic_skills:
        by_topic[str(skill.topic)].append(skill)
    for topic in sorted(by_topic):
        lines += [f"### Topic: {topic}", ""]
        for skill in by_topic[topic]:
            lines += render_skill(skill)

    return "\n".join(lines) + "\n"


def export_harness(store_root: str | Path, out_dir: str | Path) -> Path:
    """Write ``harness.md`` / ``summary.json`` / ``evolution.jsonl`` into ``out_dir``.

    Returns ``out_dir`` as a ``Path``. Every file is written whole (never appended), so
    re-exporting over a previous export's destination replaces it rather than accumulating a
    second copy of the log.
    """
    store = SkillStore(store_root)
    report = build_report(store)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "harness.md").write_text(_render_markdown(store, report))
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    edits = _read_jsonl(store.harness_log)
    (out / "evolution.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in edits))

    return out


def main(store_root: str, out_dir: str) -> None:
    """``python -m alphaapollo.core.harness.export --store_root <dir> --out_dir <dir>``"""
    out = export_harness(store_root, out_dir)
    report = build_report(SkillStore(store_root))
    print(f"[harness] exported to {out}")
    print(json.dumps(report["skills"], indent=2))
    print(json.dumps(report["edits"], indent=2))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
