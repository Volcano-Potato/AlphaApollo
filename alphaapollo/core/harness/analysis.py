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
"""Turn six runs' raw logs into the numbers the assignment asks for.

``export.py`` reports on a *harness* -- what it contains, how it was edited, which skills got
picked. This module reports on an *experiment*: it reads ``metrics.jsonl`` across arms and
phases and produces the seven required results, of which ``export.py`` alone covered three.

The one judgement call encoded here is **which problems count**. ``run_stream`` degrades a
problem whose rollout raised into an all-zero result, so a crashed problem and a wrong answer are
both ``pass_final == 0``; ``adapt/error`` distinguishes them. The three arms run concurrently
against one provider and lose *different* problems to the same rate-limit storm, so comparing
their raw rates compares different problem sets. :func:`compare_arms` therefore reports on the
intersection of problems every arm completed, and states how many were dropped to get there. A
per-arm rate over its own completed problems is also reported, because the two differing is
itself a finding worth seeing rather than hiding.

Nothing here is used by the method. It reads artifacts after the fact and writes markdown.

Usage::

    python -m alphaapollo.core.harness.analysis --out_dir ./outputs/harness/report
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ARMS = ("baseline", "raw", "evo")
MGMT_PREFIXES = ("reflect", "topic_curator", "general_curator", "raw_summarizer",
                 "offline_labeling", "selector")


@dataclass
class RunData:
    """One run directory's ``metrics.jsonl``, split into its three kinds of row."""

    name: str
    problems: dict[int, dict] = field(default_factory=dict)   # problem_idx -> row
    harness_series: list[dict] = field(default_factory=list)  # per-batch harness scale
    accounting: dict = field(default_factory=dict)            # last per-batch accounting row

    @property
    def completed(self) -> set[int]:
        """Problems that actually produced a rollout. See the module docstring."""
        return {i for i, r in self.problems.items() if not r.get("adapt/error")}

    @property
    def n_lost(self) -> int:
        return len(self.problems) - len(self.completed)


def load_run(run_dir: str | Path, name: str = "") -> RunData:
    path = Path(run_dir) / "metrics.jsonl"
    data = RunData(name=name or Path(run_dir).name)
    if not path.exists():
        logger.warning("no metrics at %s; treating the run as absent", path)
        return data

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "adapt/pass_final" in row:
            # Last write wins: a resumed run re-runs its aborted batch, and `resume` truncates
            # the partial rows -- but being tolerant of a duplicate here costs nothing and means
            # an analysis never silently double-counts if that truncation was ever bypassed.
            data.problems[int(row["step"])] = row
        elif "harness/n_general" in row:
            data.harness_series.append(row)
        elif "calls/solver" in row:
            data.accounting = row
    return data


def _rate(rows: list[dict], key: str) -> float:
    return (sum(int(r.get(key, 0) or 0) for r in rows) / len(rows)) if rows else 0.0


def overall(run: RunData, only: set[int] | None = None) -> dict:
    """Head-line accuracy. ``pass1_round0`` is Pass@1 before any in-problem evolution; the
    assignment asks for both it and the final rate, and on this model/dataset they differ."""
    idxs = sorted(run.completed if only is None else (run.completed & only))
    rows = [run.problems[i] for i in idxs]
    return {
        "n": len(rows),
        "pass1_round0": _rate(rows, "adapt/pass1_round0"),
        "pass_final": _rate(rows, "adapt/pass_final"),
        "n_lost": run.n_lost,
    }


def adaptation_curve(run: RunData, window: int = 25, only: set[int] | None = None) -> list[dict]:
    """Success rate over consecutive windows of the stream, in stream order.

    Windows, not a cumulative average: the question is whether the arm gets *better as it goes*,
    and a cumulative mean buries a late improvement under everything before it. Windows are cut
    on the problem's stream position, so a window that lost problems reports on fewer -- with
    ``n`` alongside the rate so a small window is visible as small rather than noisy-looking.
    """
    idxs = sorted(run.completed if only is None else (run.completed & only))
    if not idxs:
        return []
    buckets: dict[int, list[dict]] = defaultdict(list)
    for i in idxs:
        buckets[i // window].append(run.problems[i])
    return [
        {"window": f"{b * window}-{b * window + window - 1}", "n": len(rows),
         "pass1_round0": _rate(rows, "adapt/pass1_round0"),
         "pass_final": _rate(rows, "adapt/pass_final")}
        for b, rows in sorted(buckets.items())
    ]


def cumulative_curve(run: RunData, window: int = 25, only: set[int] | None = None) -> list[dict]:
    """Running accuracy after each completed problem, in stream order.

    :func:`adaptation_curve` cuts the stream into *disjoint* windows, which is the right shape for
    a markdown table -- a handful of rows a reader can scan. It is the wrong shape for a chart: a
    six-point step function is not a curve, and the report's tables already cover it.

    This is the complementary, per-problem view, and it carries both readings of "over time" so a
    dashboard can plot them on one x-axis:

    * ``cum_*`` -- the mean over every completed problem so far. This is the arm's headline number
      as it converges, and the curve that answers "is this arm ahead, and since when".
    * ``roll_*`` -- the mean over the trailing ``window`` completed problems. The cumulative mean
      goes deaf to late changes (by problem 120, one flip moves it by <1pt), which is exactly where
      a cross-problem mechanism's effect would show up, so the trailing mean is what keeps that
      visible at the cost of being noisier.

    ``n`` and ``n_roll`` travel with the rates for the same reason the window tables carry ``n``:
    an early point is a mean over three problems and must be readable as one.
    """
    idxs = sorted(run.completed if only is None else (run.completed & only))
    keys = ("adapt/pass1_round0", "adapt/pass_final")
    out: list[dict] = []
    seen: list[dict] = []
    totals = dict.fromkeys(keys, 0)
    for i in idxs:
        row = run.problems[i]
        seen.append(row)
        for k in keys:
            totals[k] += int(row.get(k, 0) or 0)
        n = len(seen)
        trailing = seen[-window:]
        out.append({
            "problem_idx": i,
            "n": n,
            "cum_pass1_round0": totals["adapt/pass1_round0"] / n,
            "cum_pass_final": totals["adapt/pass_final"] / n,
            "n_roll": len(trailing),
            "roll_pass1_round0": _rate(trailing, "adapt/pass1_round0"),
            "roll_pass_final": _rate(trailing, "adapt/pass_final"),
        })
    return out


def by_topic(run: RunData, only: set[int] | None = None) -> dict[str, dict]:
    """Per-topic accuracy -- the *performance* breakdown, which is a different question from
    ``export.build_report``'s ``n_topic_by_topic`` (how many skills the harness holds per topic).
    Both are required; only the latter existed."""
    idxs = sorted(run.completed if only is None else (run.completed & only))
    buckets: dict[str, list[dict]] = defaultdict(list)
    for i in idxs:
        buckets[str(run.problems[i].get("topic") or "(unlabelled)")].append(run.problems[i])
    return {
        topic: {"n": len(rows), "pass1_round0": _rate(rows, "adapt/pass1_round0"),
                "pass_final": _rate(rows, "adapt/pass_final")}
        for topic, rows in sorted(buckets.items())
    }


def cost(run: RunData) -> dict:
    """Solver-side and management-side calls and tokens, reported separately.

    The assignment asks for them split, and the split is the honest accounting of what a
    cross-problem mechanism costs: an arm that wins on accuracy while spending 40% more calls has
    not obviously won. Roles come from `accounting.py`'s `role_scope`; anything under
    ``calls/unscoped`` is a bug in that instrumentation, so it is surfaced rather than folded in.
    """
    acc = run.accounting
    n_problems = len(run.problems) or 1
    by_role = {k.split("/", 1)[1]: v for k, v in acc.items() if k.startswith("calls/")}
    return {
        "calls_solver": acc.get("calls/solver", 0),
        "calls_mgmt": acc.get("calls/mgmt_side_total", 0),
        "calls_by_role": {k: v for k, v in sorted(by_role.items())
                          if k in MGMT_PREFIXES or k == "solver"},
        "calls_unscoped": by_role.get("unscoped", 0),
        "tokens_in": sum(v for k, v in acc.items() if k.endswith("_in")),
        "tokens_out": sum(v for k, v in acc.items() if k.endswith("_out")),
        "calls_per_problem": (acc.get("calls/solver", 0) + acc.get("calls/mgmt_side_total", 0)) / n_problems,
    }


def harness_growth(run: RunData) -> list[dict]:
    """Size and length of the harness after each batch -- the growth trend, and the evidence
    that the cap actually bound."""
    return [
        {"step": r["step"], "n_general": r.get("harness/n_general", 0),
         "n_topic": r.get("harness/n_topic_total", 0),
         "total_tokens": r.get("harness/total_tokens", 0),
         "mean_skill_tokens": r.get("harness/mean_skill_tokens", 0)}
        for r in sorted(run.harness_series, key=lambda r: r["step"])
    ]


def load_selections(store_root: str | Path, only: set[int] | None = None) -> dict[int, dict]:
    """One selection row per problem, keyed by stream position.

    The de-duplication here is not defensive tidying; it is required for correctness on the
    held-out phase. ``run_experiments.sh`` gives a frozen run its state by copying the adaptation
    arm's whole directory, selection log included, and the held-out run then *appends* its own
    rows to that file. So ``heldout-evo/store/selection_log.jsonl`` holds 174 rows: 144 from the
    adaptation stream followed by 30 from the held-out year -- and because each phase numbers its
    problems from zero, the held-out rows collide with adaptation positions 0-29 rather than
    extending past them. Reading the file whole therefore answers a question about the held-out
    year with mostly adaptation data (measured: 174 "problems" instead of 30, and a mean injected
    length of 189 tokens instead of 209).

    Two rules fix it, and both are needed:

    * **Last write wins**, so a colliding position resolves to the phase that ran last -- which is
      the run being reported on, since it is the one that appended.
    * **``only``**, the set of stream positions this run actually has metrics for, so the 114
      inherited rows above the held-out range are dropped instead of being counted as problems
      that this run never saw.

    ``only`` should be the run's whole problem set (``RunData.problems``), not
    ``RunData.completed``: a problem whose rollout crashed still had skills selected into its
    context and still paid for them, and cost reporting that silently drops those understates it.
    """
    path = Path(store_root) / "selection_log.jsonl"
    if not path.exists():
        return {}
    rows: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        idx = int(row.get("problem_idx", -1))
        if only is not None and idx not in only:
            continue
        rows[idx] = row
    return dict(sorted(rows.items()))


def injected_context(store_root: str | Path, only: set[int] | None = None) -> dict:
    """Mean injected tokens per problem, and how often anything was injected at all.

    Read from the selection log rather than from the harness's size: what matters is what
    actually entered a context, and early problems see an empty harness. See
    :func:`load_selections` for why ``only`` is not optional in practice on a held-out run.
    """
    rows = load_selections(store_root, only)
    if not rows:
        return {"n_problems": 0, "mean_tokens": 0.0, "max_tokens": 0, "n_with_injection": 0}
    tokens = [int(r.get("n_tokens", 0) or 0) for r in rows.values()]
    return {
        "n_problems": len(tokens),
        "mean_tokens": sum(tokens) / len(tokens),
        "max_tokens": max(tokens),
        "n_with_injection": sum(1 for t in tokens if t > 0),
    }


def skill_usage(store_root: str | Path, only: set[int] | None = None) -> dict[str, dict]:
    """How often each skill was injected, and how often the problem was then solved.

    Counted from the selection log, not from each ``Skill``'s own counter: a skill a later
    curator cycle deleted no longer exists to hold a counter, while its earlier injections still
    happened and still cost tokens. Dropping those would understate both usage and cost.
    Ordered by frequency -- the question "which skills actually got used" is answered by the top
    of the list and by how long the tail of never-injected ones is.

    ``only`` scopes this to one phase, for the reason given in :func:`load_selections`; without it
    a held-out run reports the union of the skills two phases injected (measured: 15 rather than
    the 13 the held-out year actually saw).
    """
    usage: dict[str, dict] = defaultdict(lambda: {"n_selected": 0, "n_selected_success": 0})
    for row in load_selections(store_root, only).values():
        success = int(bool(row.get("success")))
        for sid in row.get("skill_ids") or []:
            usage[str(sid)]["n_selected"] += 1
            usage[str(sid)]["n_selected_success"] += success
    return dict(sorted(usage.items(), key=lambda kv: (-kv[1]["n_selected"], kv[0])))


def compare_arms(runs: dict[str, RunData]) -> dict:
    """Align the arms on the problems every one of them completed. See the module docstring."""
    present = {name: run for name, run in runs.items() if run.problems}
    if not present:
        return {"common": set(), "arms": {}}
    common: set[int] = set.intersection(*(run.completed for run in present.values()))
    return {
        "n_common": len(common),
        "n_total": max(len(run.problems) for run in present.values()),
        "dropped_by_arm": {name: sorted(set(run.problems) - common) for name, run in present.items()},
        "on_common": {name: overall(run, only=common) for name, run in present.items()},
        "on_own": {name: overall(run) for name, run in present.items()},
        "common": common,
    }


def transfer_cases(evo: RunData, baseline: RunData, store_root: str | Path,
                   limit: int = 10) -> dict[str, list[dict]]:
    """Surface the strongest candidates for a positive/negative transfer case study.

    Mechanical part only. A transfer *case* is an argument about what the model did differently
    with a skill in context, which needs the trajectory text and a human reading it; what can be
    automated is finding the problems worth reading. Those are the ones where the Evo arm had
    skills injected and its outcome differs from Baseline's on the same problem -- positive when
    Evo solved what Baseline did not, negative for the reverse.

    Both directions are returned because a negative case is as required as a positive one, and
    strictly more interesting: the assignment weights honest analysis of interference over a
    scoreboard.
    """
    selections = load_selections(store_root, only=set(evo.problems) or None)

    positive, negative = [], []
    for idx in sorted(evo.completed & baseline.completed):
        selected = selections.get(idx, {})
        skill_ids = selected.get("skill_ids") or selected.get("source_problems") or []
        if not skill_ids:
            continue  # nothing was injected, so nothing transferred either way
        evo_ok = int(evo.problems[idx].get("adapt/pass_final", 0))
        base_ok = int(baseline.problems[idx].get("adapt/pass_final", 0))
        if evo_ok == base_ok:
            continue
        case = {
            "problem_idx": idx,
            "topic": evo.problems[idx].get("topic", ""),
            "skill_ids": skill_ids,
            "injected_tokens": selected.get("n_tokens", 0),
            "evo_pass_final": evo_ok,
            "baseline_pass_final": base_ok,
            "trajectory": f"trajectories/problem_{idx:04d}.json",
        }
        (positive if evo_ok > base_ok else negative).append(case)
    return {"positive": positive[:limit], "negative": negative[:limit]}


# --- rendering ------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def render(adapt: dict[str, RunData], heldout: dict[str, RunData],
           store_roots: dict[str, str], window: int = 25) -> str:
    lines = ["# Task C results", ""]
    comparison = compare_arms(adapt)

    lines += ["## 1. Adaptation stream", ""]
    if comparison.get("arms") == {} and not comparison.get("on_common"):
        lines += ["_No adaptation runs found._", ""]
    else:
        lines += [f"Compared on the {comparison['n_common']} of {comparison['n_total']} problems "
                  f"every arm completed. Problems each arm lost: "
                  + ", ".join(f"{n} ({len(v)})" for n, v in comparison["dropped_by_arm"].items())
                  + ".", ""]
        lines += _table(
            ["arm", "n", "Pass@1 (round 0)", "final", "on own set (final)", "lost"],
            [[name, s["n"], _pct(s["pass1_round0"]), _pct(s["pass_final"]),
              _pct(comparison["on_own"][name]["pass_final"]), comparison["on_own"][name]["n_lost"]]
             for name, s in comparison["on_common"].items()])

        lines += [f"### Over time (windows of {window})", ""]
        for name, run in adapt.items():
            curve = adaptation_curve(run, window=window, only=comparison["common"])
            if curve:
                lines += [f"**{name}**", ""]
                lines += _table(["window", "n", "Pass@1", "final"],
                                [[c["window"], c["n"], _pct(c["pass1_round0"]), _pct(c["pass_final"])]
                                 for c in curve])

    lines += ["## 2. Held-out set (final frozen harness)", ""]
    if any(r.problems for r in heldout.values()):
        held = compare_arms(heldout)
        lines += _table(["arm", "n", "Pass@1 (round 0)", "final", "lost"],
                        [[name, s["n"], _pct(s["pass1_round0"]), _pct(s["pass_final"]),
                          held["on_own"][name]["n_lost"]]
                         for name, s in held["on_common"].items()])
    else:
        lines += ["_Not run yet._", ""]

    lines += ["## 3. By topic", ""]
    for phase, runs in (("adaptation", adapt), ("held-out", heldout)):
        for name, run in runs.items():
            breakdown = by_topic(run)
            if breakdown:
                lines += [f"**{phase} / {name}**", ""]
                lines += _table(["topic", "n", "Pass@1", "final"],
                                [[t, s["n"], _pct(s["pass1_round0"]), _pct(s["pass_final"])]
                                 for t, s in breakdown.items()])

    lines += ["## 4. Harness size, length and growth", ""]
    for name, run in adapt.items():
        growth = harness_growth(run)
        if growth:
            lines += [f"**{name}**", ""]
            lines += _table(["after problem", "general", "topic", "total tokens", "mean/skill"],
                            [[g["step"], g["n_general"], g["n_topic"], g["total_tokens"],
                              f"{g['mean_skill_tokens']:.0f}"] for g in growth])

    lines += ["## 5. Injected context and model calls", ""]
    rows = []
    for name, run in adapt.items():
        if not run.problems:
            continue  # an arm that has not run is absent, not an arm that cost nothing
        c = cost(run)
        inj = injected_context(store_roots[name], set(run.problems)) if name in store_roots else {}
        rows.append([name, c["calls_solver"], c["calls_mgmt"],
                     f"{c['calls_per_problem']:.1f}",
                     f"{c['tokens_in'] / 1e6:.2f}M", f"{c['tokens_out'] / 1e6:.2f}M",
                     f"{inj.get('mean_tokens', 0):.0f}", c["calls_unscoped"]])
    lines += _table(["arm", "solver calls", "mgmt calls", "calls/problem", "tokens in",
                     "tokens out", "mean injected tok", "unscoped"], rows)
    lines += ["Management calls are reported separately from solver calls: an arm that wins on "
              "accuracy while spending materially more calls has not obviously won. A non-zero "
              "`unscoped` column is an instrumentation bug, not a cost category.", ""]

    lines += ["## 6. Skill usage frequency", ""]
    for name in ("evo", "raw"):
        if name not in store_roots or not adapt.get(name, RunData(name)).problems:
            continue
        usage = skill_usage(store_roots[name], set(adapt[name].problems))
        if not usage:
            continue
        lines += [f"**{name}** — {len(usage)} skill(s) were injected at least once", ""]
        lines += _table(
            ["skill", "times injected", "solved when injected", "share of problems"],
            [[sid, u["n_selected"], u["n_selected_success"],
              _pct(u["n_selected"] / max(len(adapt[name].problems), 1))]
             for sid, u in usage.items()])
    lines += ["\"Solved when injected\" is an association, not an attribution -- a skill picked "
              "for easy problems scores well without helping. Section 7 pairs the same problems "
              "against Baseline, which is where a causal claim can start.", ""]

    lines += ["## 7. Transfer cases", ""]
    if "evo" in adapt and "baseline" in adapt and "evo" in store_roots:
        cases = transfer_cases(adapt["evo"], adapt["baseline"], store_roots["evo"])
        for label, key in (("Positive (Evo solved, Baseline did not)", "positive"),
                           ("Negative (Baseline solved, Evo did not)", "negative")):
            lines += [f"**{label}** — {len(cases[key])} candidate(s)", ""]
            if cases[key]:
                lines += _table(["problem", "topic", "skills injected", "tokens", "trajectory"],
                                [[c["problem_idx"], c["topic"], ",".join(map(str, c["skill_ids"])),
                                  c["injected_tokens"], c["trajectory"]] for c in cases[key]])
        lines += ["These are candidates, not conclusions: a transfer case is an argument about "
                  "what the model did differently, which needs the trajectory read alongside the "
                  "injected skill text. The tables above say which files to open.", ""]
    else:
        lines += ["_Needs both the Evo and Baseline adaptation runs._", ""]

    return "\n".join(lines)


def main(out_dir: str = "./outputs/harness/report", root: str = "./outputs/harness",
         window: int = 25) -> None:
    """``python -m alphaapollo.core.harness.analysis --out_dir ./outputs/harness/report``"""
    logging.basicConfig(level=logging.INFO)
    root_path = Path(root)
    adapt = {arm: load_run(root_path / f"adapt-{arm}", arm) for arm in ARMS}
    heldout = {arm: load_run(root_path / f"heldout-{arm}", arm) for arm in ARMS}
    store_roots = {"evo": str(root_path / "adapt-evo" / "store"),
                   "raw": str(root_path / "adapt-raw" / "pool")}

    report = render(adapt, heldout, store_roots, window=window)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(report, encoding="utf-8")

    payload = {
        "adaptation": {a: overall(r) for a, r in adapt.items() if r.problems},
        "heldout": {a: overall(r) for a, r in heldout.items() if r.problems},
        "comparison": {k: v for k, v in compare_arms(adapt).items() if k != "common"},
        "by_topic": {a: by_topic(r) for a, r in adapt.items() if r.problems},
        "curve": {a: adaptation_curve(r, window) for a, r in adapt.items() if r.problems},
        "cost": {a: cost(r) for a, r in adapt.items() if r.problems},
        "growth": {a: harness_growth(r) for a, r in adapt.items() if r.harness_series},
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                                      encoding="utf-8")
    print(f"[analysis] wrote {out / 'results.md'} and {out / 'results.json'}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
