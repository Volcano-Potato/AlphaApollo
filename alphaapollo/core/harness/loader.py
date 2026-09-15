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
"""Topic-preserving problem-stream loader for the cross-problem skill harness.

The upstream loader (``alphaapollo/core/generation/evolving/utils/dataset_loader.py``,
``load_informal_math_data``) keeps only ``question`` / ``ground_truth`` / ``gt_traj`` from each
row's ``extra_info`` -- everything else, including ``topic`` and ``year``, is dropped on the
floor. That is fine for the baseline evolving loop, which never needs to know what a problem is
*about*, but it is exactly the information Task A/B need to scope skills to a topic and Task C
needs to split AIME strictly by year. Hence a separate loader here rather than reusing (or
patching) the upstream one -- see the global constraint that upstream files stay untouched.

``load_stream()`` also guarantees ``gt_traj``, ``ground_truth`` and ``data_source`` are always
present (empty string if absent), because ``evolving_main.run_problem`` indexes all three straight
off ``current_problem`` with ``[...]`` rather than ``.get(...)`` (evolving_main.py:552, :563) -- a
missing key there is a ``KeyError``, not a graceful fallback. That ``KeyError`` is raised inside
the driver's worker thread, where ``run_stream`` catches it, degrades the problem to an all-zero
result and keeps going, so the failure mode is not a crash but a *silent* one: every problem
yields zero solver signal while Reflect still fires and compiles skills out of empty
trajectories. ``data_source`` was missing from this tuple originally and was caught only by
running the driver end to end, which is why the list is now pinned by a test enumerating
upstream's unconditional reads.

``interleave()`` implements the topic-interleaved task stream (design doc S15.3), and is
**currently not used by any production path** -- ``prepare_harness_stream`` builds the streams in
plain year order. It is kept deliberately rather than deleted.

The reasoning it was written for still holds: AIME arrives in year order, so a fine-grained
technique can recur dozens of problems apart, and a three-arm comparison over such a stream cannot
distinguish "the skill did not help" from "the skill never got a chance to fire". What changed is
that interleaving needs topic labels *before* the run, and the method turned out not to need
per-problem topics at all (``reflect`` names its own; ``selector`` retrieves without a topic
filter) -- so requiring them purely to order the stream would have reintroduced a dependency the
design had just shed. AIME also mixes all four topics within a single year's 30 problems, so a
topic recurs every few problems anyway.

If the experiment shows skills being minted far from any reuse opportunity, this is the ready
alternative ordering, and the assignment explicitly permits it. Its behaviour is pinned by the
tests in ``tests/harness/test_loader.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_TOPICS = ("algebra", "number_theory", "combinatorics", "geometry")

_FIELDS = ("question", "ground_truth", "gt_traj", "data_source", "topic", "technique", "year", "contest", "number")


def load_stream(path: str | Path) -> list[dict]:
    """Load a parquet problem stream, preserving every field in ``_FIELDS``.

    Each row's ``extra_info`` cell is expected to be a dict; missing or malformed (non-dict,
    ``None``, absent column) ``extra_info`` degrades to ``{}`` rather than raising, so a
    partially-populated dataset still loads with blank fields rather than crashing the run.
    Every field in ``_FIELDS`` is always present in the returned dict, defaulting to ``""`` when
    absent -- callers (including upstream code that reads ``gt_traj`` unconditionally) never need
    to guard with ``.get(...)``.

    ``problem_idx`` is assigned from the row's position in the parquet file (0-based). This is
    later overwritten by ``interleave()`` once problems are reordered into a stream; both
    assignments matter, since a caller using ``load_stream`` output directly (no interleaving)
    still needs a stable index.
    """
    df = pd.read_parquet(path)
    problems: list[dict] = []
    for row_idx, row in enumerate(df.to_dict("records")):
        info = row.get("extra_info", {})
        if not isinstance(info, dict):
            info = {}
        problem = {field: info.get(field, "") for field in _FIELDS}
        problem["problem_idx"] = row_idx
        problems.append(problem)
    return problems


def batches(problems: list[dict], batch_size: int = 8, *, drop_last: bool = True) -> list[list[dict]]:
    """Split ``problems`` into fixed-size batches.

    ``drop_last=True`` (the default, and what an adaptation run wants) discards an incomplete
    tail: a half-batch would make the spacing between harness-update points inconsistent across
    the stream, and the update cohort is what a batch *is* during adaptation.

    ``drop_last=False`` is for a **frozen** run. There, nothing updates -- batching is pure
    chunking for the thread pool and carries no protocol meaning -- so dropping the tail buys
    nothing and costs evaluation problems. It cost 6 of the 30 held-out problems at
    ``batch_size=8``, which would have quietly turned "at least one complete recent year held
    out for final evaluation" (the assignment's wording) into 80% of one, and shrunk the only
    set the headline number is computed on.
    """
    n_full = len(problems) // batch_size
    out = [problems[i * batch_size:(i + 1) * batch_size] for i in range(n_full)]
    tail = problems[n_full * batch_size:]
    if tail and not drop_last:
        out.append(tail)
    return out


def interleave(
    problems: list[dict],
    batch_size: int = 8,
    topics: tuple[str, ...] = DEFAULT_TOPICS,
) -> list[dict]:
    """Reorder ``problems`` into a topic-interleaved stream (design doc S15.3).

    Each output batch of ``batch_size`` contains exactly ``batch_size // len(topics)`` problems
    from every topic, drawn in that topic's own chronological order. Buckets are drained in
    lockstep: as soon as any one topic can no longer supply a full share, the whole round stops
    and every bucket's remainder is discarded -- partially filling a batch from the topics that
    still have problems left would break the fixed cross-topic composition the rest of the
    pipeline (and the interleaving's whole point) depends on.

    Raises:
        ValueError: if ``batch_size`` is not evenly divisible by ``len(topics)``.
    """
    if batch_size % len(topics) != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by the number of topics ({len(topics)})")
    share = batch_size // len(topics)

    buckets: dict[str, list[dict]] = {topic: [] for topic in topics}
    for problem in problems:
        topic = problem.get("topic")
        if topic in buckets:
            buckets[topic].append(problem)
    for topic in topics:
        # Only `contest` is cast to str: real datasets mix `None` and "I"/"II" for it, which raises
        # TypeError when Python compares them during sort. `year`/`number` are left as-is (ints in
        # practice) -- stringifying them would corrupt numeric order for multi-digit problem
        # numbers (e.g. "10" sorts before "2" lexicographically).
        buckets[topic].sort(key=lambda p: (p.get("year", 0), str(p.get("contest", "")), p.get("number", 0)))

    cursors = {topic: 0 for topic in topics}
    stream: list[dict] = []
    while all(cursors[topic] + share <= len(buckets[topic]) for topic in topics):
        for topic in topics:
            start = cursors[topic]
            stream.extend(buckets[topic][start:start + share])
            cursors[topic] += share

    for position, problem in enumerate(stream):
        problem["problem_idx"] = position
    return stream
