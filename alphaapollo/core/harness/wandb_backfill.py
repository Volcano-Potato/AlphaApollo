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
"""Put the numbers ``analysis.py`` computes onto the wandb runs that produced them.

`tracker.py` forwards to wandb exactly what it writes to ``metrics.jsonl``: per-problem 0/1
outcomes, per-batch harness scale, cumulative call/token counters. Every *reported* number --
accuracy over the stream, the per-topic breakdown, cost per problem, skill usage, transfer-case
candidates -- is derived from those rows after the fact by `analysis.py`, and none of it was ever
sent to wandb. A finished run's dashboard therefore shows a `adapt/pass_final` column that
alternates 0 and 1 and nothing a reader can conclude from.

This module closes that gap *without re-running anything*. `analysis.py` reads only
``metrics.jsonl`` and the selection logs, both of which the completed runs still have on disk, so
the derived metrics can be computed now and attached to the original wandb runs by resuming them
under the run id stored in ``<run_dir>/wandb_run_id.txt``.

Three design points worth stating, because each is a constraint wandb imposes rather than a
preference:

1. **The derived curves cannot reuse the original steps.** wandb's history step is monotonic; a
   resumed run continues from its last step and silently drops anything logged at an earlier one.
   So the per-problem curves are logged with a custom x-axis (``analysis/problem_idx``, declared
   via ``define_metric``) and left to take whatever global steps come after the originals. The
   charts read correctly against problem index; the raw ``_step`` axis is meaningless for them,
   which is what ``define_metric`` exists to express.
2. **Pushing history is therefore not repeatable**, since it is append-only -- a second push
   would draw a second set of points over the same x values, and wandb offers no way to remove
   the first. Two separate files keep that from happening, because they answer two different
   questions and collapsing them into one would make ``--force`` override both:

   * ``<run_dir>/wandb_backfill.json`` records *what has already been sent*, including a sticky
     ``series_sent`` that flips immediately before the first history row goes out and never back.
     A run whose history has been sent is refused history again **unconditionally** -- ``--force``
     cannot make a duplicated curve correct. Summary keys and tables overwrite by key, so
     ``--skip_series --force`` remains the way to correct a number on an already-pushed run.
     ``--force`` governs only this file, i.e. "yes, redo what I already did".
   * the harness's own pid lock (`resume.acquire_run_lock`) handles *concurrency*, is taken on
     every push, and is never bypassable. It also means a backfill cannot race a run still
     appending to the ``metrics.jsonl`` it reads.

   Both live in the run directory for the same reason the run id does: wiping the directory is how
   this repo says "that experiment is gone".
3. **Nothing here is allowed to be a source of truth.** ``metrics.jsonl`` and `analysis.py` stay
   the reproducible artifact; this is a one-way copy onto a dashboard. ``--dry_run`` writes the
   exact payload to ``<run_dir>/wandb_backfill_preview.json`` and touches no network, which is
   also how the tests check the numbers without wandb installed.

Usage::

    # look at what would be pushed, for all six runs, without touching wandb
    python -m alphaapollo.core.harness.wandb_backfill --root ./outputs/harness --dry_run

    # push it
    python -m alphaapollo.core.harness.wandb_backfill --root ./outputs/harness
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alphaapollo.core.harness import analysis
from alphaapollo.core.harness.resume import RunAlreadyActive, acquire_run_lock, release_run_lock

logger = logging.getLogger(__name__)

# The x-axis every derived per-problem series is plotted against. Logged as an ordinary metric and
# then named as the step metric, which is wandb's only mechanism for "this series is indexed by
# something other than the global step".
STEP_METRIC = "analysis/problem_idx"
MARKER_NAME = "wandb_backfill.json"
PREVIEW_NAME = "wandb_backfill_preview.json"

# Where an arm keeps its selection log, by convention rather than by config: the evo arm writes a
# skill store under `store/`, the raw arm a trajectory pool under `pool/`, and baseline neither.
# Probing for the file is deliberate -- reading it back out of the run's config would make this
# depend on the config still being on disk and still pointing where it pointed at run time.
STATE_DIRS = ("store", "pool")


@dataclass
class Payload:
    """Everything one run contributes to wandb, computed before any wandb call is made.

    Split three ways because wandb treats the three differently: ``series`` becomes history (one
    row per problem, append-only), ``summary`` becomes the run's summary columns (overwritable,
    and what the runs table sorts and compares by), and ``tables`` become logged
    ``wandb.Table``s (the breakdowns that have a string key and so cannot be a scalar column).
    """

    name: str
    series: list[dict] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    tables: dict[str, tuple[list[str], list[list[Any]]]] = field(default_factory=dict)

    def as_json(self) -> dict:
        return {"name": self.name, "series": self.series, "summary": self.summary,
                "tables": {k: {"columns": c, "data": d} for k, (c, d) in self.tables.items()}}


def find_state_root(run_dir: str | Path) -> Path | None:
    """The arm's selection log directory, or ``None`` for an arm that keeps none (baseline)."""
    run_dir = Path(run_dir)
    for name in STATE_DIRS:
        if (run_dir / name / "selection_log.jsonl").exists():
            return run_dir / name
    return None


def build_payload(run_dir: str | Path, *, name: str = "", window: int = 25,
                  state_root: str | Path | None = None) -> Payload:
    """Compute one run's derived metrics from its on-disk artifacts. No wandb, no network.

    Every number here comes from a function in `analysis.py` rather than being recomputed, so the
    dashboard and ``report/results.md`` cannot drift apart -- a discrepancy between the two would
    be worse than the dashboard staying empty.
    """
    run_dir = Path(run_dir)
    run = analysis.load_run(run_dir, name or run_dir.name)
    payload = Payload(name=run.name)
    if not run.problems:
        logger.warning("%s has no per-problem rows; nothing to backfill", run_dir)
        return payload

    # --- per-problem curves (history) -------------------------------------------------------
    for point in analysis.cumulative_curve(run, window=window):
        payload.series.append({
            STEP_METRIC: point["problem_idx"],
            "analysis/n_completed": point["n"],
            "analysis/cum_pass1_round0": point["cum_pass1_round0"],
            "analysis/cum_pass_final": point["cum_pass_final"],
            f"analysis/roll{window}_pass1_round0": point["roll_pass1_round0"],
            f"analysis/roll{window}_pass_final": point["roll_pass_final"],
            # How many problems the trailing mean is actually over. It only differs from `window`
            # for the first few points, which is exactly where the rolling curve is at its most
            # volatile and most likely to be over-read.
            f"analysis/roll{window}_n": point["n_roll"],
        })

    # --- headline scalars (summary) ---------------------------------------------------------
    head = analysis.overall(run)
    payload.summary.update({
        "analysis/overall/n": head["n"],
        "analysis/overall/n_lost": head["n_lost"],
        "analysis/overall/pass1_round0": head["pass1_round0"],
        "analysis/overall/pass_final": head["pass_final"],
    })

    # Per-topic accuracy as summary columns *and* as a table. The columns are what makes two arms
    # comparable in wandb's runs view (one row per run, one column per topic); the table is what
    # carries `n` alongside, without which a 100% topic of size 1 reads like a result.
    topics = analysis.by_topic(run)
    for topic, stat in topics.items():
        payload.summary[f"analysis/topic/{topic}/pass_final"] = stat["pass_final"]
        payload.summary[f"analysis/topic/{topic}/pass1_round0"] = stat["pass1_round0"]
        payload.summary[f"analysis/topic/{topic}/n"] = stat["n"]
    if topics:
        payload.tables["analysis/by_topic"] = (
            ["topic", "n", "pass1_round0", "pass_final"],
            [[t, s["n"], s["pass1_round0"], s["pass_final"]] for t, s in topics.items()],
        )

    # --- cost, reported the way the assignment asks: solver side apart from management side ---
    if run.accounting:
        c = analysis.cost(run)
        payload.summary.update({
            "analysis/cost/calls_solver": c["calls_solver"],
            "analysis/cost/calls_mgmt": c["calls_mgmt"],
            "analysis/cost/calls_per_problem": c["calls_per_problem"],
            "analysis/cost/tokens_in": c["tokens_in"],
            "analysis/cost/tokens_out": c["tokens_out"],
            "analysis/cost/calls_unscoped": c["calls_unscoped"],
        })
        for role, n in c["calls_by_role"].items():
            payload.summary[f"analysis/cost/calls_by_role/{role}"] = n

    # --- harness scale at the end of the run -------------------------------------------------
    growth = analysis.harness_growth(run)
    if growth:
        final = growth[-1]
        payload.summary.update({
            "analysis/harness/final_n_general": final["n_general"],
            "analysis/harness/final_n_topic": final["n_topic"],
            "analysis/harness/final_total_tokens": final["total_tokens"],
            "analysis/harness/final_mean_skill_tokens": final["mean_skill_tokens"],
        })
        payload.tables["analysis/harness_growth"] = (
            ["step", "n_general", "n_topic", "total_tokens", "mean_skill_tokens"],
            [[g["step"], g["n_general"], g["n_topic"], g["total_tokens"], g["mean_skill_tokens"]]
             for g in growth],
        )

    # --- injected context and skill usage, which live in the selection log, not in metrics ----
    state_root = Path(state_root) if state_root is not None else find_state_root(run_dir)
    if state_root is not None:
        # Scoped to this run's own stream positions. A frozen run inherits its state directory --
        # selection log included -- from the adaptation arm it was copied from, so the file holds
        # both phases' rows under colliding indices; see `analysis.load_selections`. Passing the
        # whole problem set rather than `completed` is deliberate: a crashed problem still had
        # skills selected into its context and still paid for them.
        own = set(run.problems)
        inj = analysis.injected_context(state_root, own)
        payload.summary.update({
            "analysis/injected/mean_tokens": inj["mean_tokens"],
            "analysis/injected/max_tokens": inj.get("max_tokens", 0),
            "analysis/injected/n_with_injection": inj["n_with_injection"],
            "analysis/injected/n_problems": inj["n_problems"],
        })
        usage = analysis.skill_usage(state_root, own)
        if usage:
            n_problems = max(len(run.problems), 1)
            payload.summary["analysis/usage/n_skills_injected"] = len(usage)
            payload.tables["analysis/skill_usage"] = (
                ["skill", "times_injected", "solved_when_injected", "share_of_problems"],
                [[sid, u["n_selected"], u["n_selected_success"], u["n_selected"] / n_problems]
                 for sid, u in usage.items()],
            )

    # The windowed table `results.md` prints, so the two can be checked against each other.
    curve = analysis.adaptation_curve(run, window=window)
    if curve:
        payload.tables["analysis/window_curve"] = (
            ["window", "n", "pass1_round0", "pass_final"],
            [[c["window"], c["n"], c["pass1_round0"], c["pass_final"]] for c in curve],
        )
    return payload


def add_cross_arm(payloads: dict[str, Payload], runs: dict[str, analysis.RunData],
                  state_roots: dict[str, Path]) -> None:
    """Attach the numbers that only exist relative to the other arms.

    Two arms' raw rates are computed over different problem sets whenever a rate-limit storm cost
    them different problems (see `analysis`'s module docstring), so the headline comparison must be
    taken on the intersection. Put on each arm's *own* wandb run, the intersection rate is what
    makes wandb's runs table a fair comparison instead of a misleading one -- otherwise a reader
    sorts by ``analysis/overall/pass_final`` and compares three different denominators.
    """
    present = {n: r for n, r in runs.items() if r.problems}
    if len(present) < 2:
        return
    comparison = analysis.compare_arms(present)
    common = comparison["common"]
    for name, stat in comparison["on_common"].items():
        if name not in payloads:
            continue
        payloads[name].summary.update({
            "analysis/common/n": stat["n"],
            "analysis/common/pass1_round0": stat["pass1_round0"],
            "analysis/common/pass_final": stat["pass_final"],
        })
        # The cumulative curve restricted to the same intersection: the unrestricted one above is
        # each arm on its own problems, which is the honest per-arm view but not a comparable one.
        # When no arm lost a problem the two sets coincide and the restricted curve would be a
        # byte-identical copy under a second name -- worth skipping, since history is append-only
        # and every redundant row is permanent.
        if runs[name].completed == common:
            continue
        for point in analysis.cumulative_curve(runs[name], only=common):
            payloads[name].series.append({
                STEP_METRIC: point["problem_idx"],
                "analysis/common/cum_pass_final": point["cum_pass_final"],
                "analysis/common/cum_pass1_round0": point["cum_pass1_round0"],
            })

    # Transfer-case candidates are a property of the Evo arm read against Baseline, so they are
    # attached to the Evo run. They are candidates for a human to read, never conclusions -- the
    # table carries the trajectory path for exactly that reason.
    if "evo" in present and "baseline" in present and "evo" in state_roots:
        cases = analysis.transfer_cases(present["evo"], present["baseline"], state_roots["evo"])
        for key in ("positive", "negative"):
            payloads["evo"].summary[f"analysis/transfer/n_{key}"] = len(cases[key])
            if cases[key]:
                payloads["evo"].tables[f"analysis/transfer_{key}"] = (
                    ["problem_idx", "topic", "skill_ids", "injected_tokens", "trajectory"],
                    [[c["problem_idx"], c["topic"], ",".join(map(str, c["skill_ids"])),
                      c["injected_tokens"], c["trajectory"]] for c in cases[key]],
                )


# --- wandb side -------------------------------------------------------------------------------


def read_run_id(run_dir: str | Path) -> str | None:
    path = Path(run_dir) / "wandb_run_id.txt"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def detect_project(run_dir: str | Path) -> str | None:
    """Recover the wandb project from the config wandb itself stored inside the run directory.

    Preferred over re-reading ``examples/configs/*.yaml``: the config on disk may have been edited
    since the run, while ``wandb/<run>/files/config.yaml`` is a snapshot of what this run actually
    used, and pushing to the wrong project is a silent failure that looks like success.
    """
    try:
        import yaml
    except Exception:
        return None
    for cfg in sorted(Path(run_dir).glob("wandb/*/files/config.yaml"), reverse=True):
        try:
            loaded = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        project = ((loaded.get("wandb") or {}).get("value") or {}).get("project")
        if project:
            return str(project)
    return None


def payload_digest(payload: Payload, *, include_series: bool) -> str:
    """A short content hash of what a push would send, recorded in the marker.

    Not a correctness mechanism -- it is what lets a reader of a marker tell "this run already has
    exactly these numbers" from "this run was pushed before the held-out scoping bug was fixed",
    which is otherwise unanswerable from the run directory alone.
    """
    body = {"series": payload.series if include_series else [], "summary": payload.summary,
            "tables": {k: v for k, v in sorted(payload.tables.items())}}
    blob = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _write_marker(path: Path, body: dict) -> None:
    """Best-effort marker write. A marker that cannot be written must not turn a completed push
    into a crash -- the push already happened, and raising here would report the opposite."""
    try:
        path.write_text(json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.error("%s: could not record the push marker (%s). The metrics WERE pushed; a later "
                     "run of this command will not know that.", path, exc)


def read_marker(run_dir: str | Path) -> dict:
    """The record of what this machine has already pushed for ``run_dir``, or ``{}``."""
    try:
        return json.loads((Path(run_dir) / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def history_already_sent(marker: dict) -> bool:
    """Whether appending history to this run risks drawing a second copy of an existing curve.

    ``series_sent`` is sticky: once a push has begun logging history rows it stays true through
    every later marker rewrite, including a ``--skip_series`` repair that itself sends none. A
    repair that reset it would hand the next ``--force`` a licence to duplicate exactly the curves
    the repair was careful not to touch.

    A marker written before this field existed is read conservatively via ``n_series``, and a
    marker with neither is assumed to have sent history: the cost of being wrong in that direction
    is a refusal the operator can resolve by looking at the run, and in the other direction it is
    a permanently doubled curve.
    """
    if not marker:
        return False
    if "series_sent" in marker:
        return bool(marker["series_sent"])
    return int(marker.get("n_series", 1)) > 0


def _check_marker(marker: dict, *, include_series: bool, force: bool) -> str | None:
    """Why this push must not proceed, or ``None``.

    Two independent questions, deliberately not collapsed into one flag:

    * **Has this already been done?** Any marker at all means yes, and ``--force`` is how an
      operator says they meant it.
    * **Is appending history safe?** Only if no history has ever been sent to this run. This one
      ``--force`` cannot answer, because no amount of intent makes a second copy of a curve into
      the right result -- wandb offers no way to remove the first. So it is an unconditional
      refusal, and the way out is ``--skip_series`` (summary and tables overwrite by key) or,
      for someone who has actually inspected the remote run, deleting the marker by hand.
    """
    if include_series and history_already_sent(marker):
        return ("history was already pushed to this run, and wandb history is append-only -- "
                "sending it again would draw a second set of points over the same x values, "
                "permanently. Use --skip_series to correct summary and tables (both overwrite by "
                f"key), or delete {MARKER_NAME} by hand if you have checked the remote run and "
                "know its history is missing.")
    if marker and not force:
        state = marker.get("state", "complete")
        if state == "complete":
            return (f"already backfilled at {marker.get('pushed_at', '?')} "
                    f"(digest {marker.get('digest', '?')}); pass --force to redo it.")
        return (f"a previous push left {MARKER_NAME} in state {state!r}, so part of this payload "
                f"may already be on the run. Inspect it, then pass --force.")
    return None


def push(run_dir: str | Path, payload: Payload, *, project: str | None = None,
         entity: str | None = None, force: bool = False, dry_run: bool = False,
         include_series: bool = True) -> str:
    """Send one payload to the wandb run that owns ``run_dir``.

    Returns the outcome as one of ``"pushed"``, ``"skipped"``, ``"failed"`` or ``"dry_run"``.
    The skipped/failed split is the whole point of returning a string rather than a bool: "there
    was nothing to do" and "authentication failed" are the same amount of data uploaded but very
    different things for the caller's exit code to say.

    ``include_series=False`` sends only the summary and tables. Both of those are overwritable by
    key, so a run whose derived *scalars* were wrong can be corrected without appending a second
    copy of curves that were already right -- which is the only safe way to repair a pushed run.

    Never raises: this is a reporting convenience run against finished experiments, and a wandb
    outage must not look like a lost result.
    """
    run_dir = Path(run_dir)
    if not payload.series and not payload.summary:
        logger.warning("%s: nothing to push", run_dir)
        return "skipped"

    if dry_run:
        try:
            (run_dir / PREVIEW_NAME).write_text(
                json.dumps(payload.as_json(), indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8")
        except OSError as exc:
            logger.error("%s: could not write the preview: %s", run_dir, exc)
            return "failed"
        logger.info("%s: dry run, wrote %s", run_dir, run_dir / PREVIEW_NAME)
        return "dry_run"

    run_id = read_run_id(run_dir)
    if not run_id:
        logger.error("%s: no wandb_run_id.txt, so there is no run to attach these metrics to. "
                     "This run was logged jsonl-only.", run_dir)
        return "skipped"

    project = project or detect_project(run_dir)
    if not project:
        logger.error("%s: could not determine the wandb project; pass --project", run_dir)
        return "failed"

    try:
        import wandb
    except Exception as exc:
        logger.error("wandb is not installed, so there is nothing to backfill into: %s", exc)
        return "failed"

    # Concurrency is a separate concern from "has this been done", and is held by a separate
    # file. `--force` overrides the record; it must never override the lock, or two operators who
    # both typed --force append two copies of everything. This is the same pid lock the harness
    # driver takes on a run directory, which also means a backfill cannot race a run that is
    # still appending to the very metrics.jsonl it reads.
    try:
        acquire_run_lock(run_dir)
    except RunAlreadyActive as exc:
        logger.error("%s: %s", run_dir, exc)
        return "failed"
    except OSError as exc:
        logger.error("%s: could not take the run lock: %s", run_dir, exc)
        return "failed"

    try:
        marker_path = run_dir / MARKER_NAME
        previous = read_marker(run_dir)
        refusal = _check_marker(previous, include_series=include_series, force=force)
        if refusal:
            logger.warning("%s: %s", run_dir, refusal)
            return "skipped"

        sending_series = bool(include_series and payload.series)
        digest = payload_digest(payload, include_series=include_series)
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        base = {"run_id": run_id, "project": project, "digest": digest,
                "n_series": len(payload.series) if include_series else 0,
                "n_summary": len(payload.summary), "tables": sorted(payload.tables),
                # Inherited only, for now. It flips at the point of no return -- immediately
                # before the first history row is logged, below -- and never back. Setting it
                # here would mean a push that failed to authenticate, having sent nothing, still
                # locked the run out of ever receiving its curves.
                "series_sent": history_already_sent(previous)}
        _write_marker(marker_path, {**base, "state": "in_progress", "started_at": started})

        def fail(message: str, *args) -> str:
            logger.error(f"%s: {message}", run_dir, *args)
            _write_marker(marker_path, {**base, "state": "failed", "started_at": started,
                                        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            return "failed"

        try:
            # `resume="must"` rather than "allow": the run exists by construction, and "allow"
            # would quietly create a fresh empty run under a mistyped id -- which looks like
            # success and leaves the real run untouched.
            wrun = wandb.init(project=project, entity=entity, id=run_id, resume="must",
                              dir=str(run_dir))
        except (Exception, KeyboardInterrupt) as exc:
            # KeyboardInterrupt for the same reason `tracker._try_init_wandb` catches it: it is
            # what wandb raises from its interactive credential prompt, and it is not an
            # `Exception`.
            return fail("could not resume wandb run %s in %s: %r", run_id, project, exc)

        try:
            if sending_series:
                # The point of no return: from the next line on, this run's history may contain
                # rows that nothing can remove. Recorded first, and flushed to disk before the
                # first row goes out, so a process killed mid-upload still leaves the fact behind.
                base["series_sent"] = True
                _write_marker(marker_path, {**base, "state": "in_progress",
                                            "started_at": started})
                # Declared before the first log so the very first point is already indexed by
                # problem index rather than by the global step it happened to land on.
                wrun.define_metric(STEP_METRIC)
                wrun.define_metric("analysis/*", step_metric=STEP_METRIC)
                for row in payload.series:
                    wrun.log(row)
            for key, (columns, data) in payload.tables.items():
                wrun.log({key: wandb.Table(columns=columns, data=data)})
            for key, value in payload.summary.items():
                wrun.summary[key] = value
            wrun.summary["analysis/backfilled_at"] = started
        except Exception as exc:
            try:
                wrun.finish(exit_code=1)
            except (Exception, KeyboardInterrupt, TypeError):
                pass
            return fail("backfill failed part-way, and the run may hold part of it: %s", exc)

        try:
            # Inside the success path, not a `finally`: wandb uploads on finish, so a finish that
            # raises means the payload may never have left this machine. Recording success first
            # and then discovering that would leave a marker asserting something untrue.
            wrun.finish()
        except (Exception, KeyboardInterrupt) as exc:
            return fail("wandb.finish() failed, so the upload cannot be assumed complete: %r", exc)

        _write_marker(marker_path, {**base, "state": "complete", "started_at": started,
                                    "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "included_series": sending_series})
        logger.info("%s: pushed %d series rows, %d summary keys, %d tables to %s/%s",
                    run_dir, len(payload.series) if sending_series else 0, len(payload.summary),
                    len(payload.tables), project, run_id)
        return "pushed"
    finally:
        release_run_lock(run_dir)


def preflight(selected: dict[str, Path], *, phases: list[str], wanted: set[str],
              root: Path, project: str | None) -> list[str]:
    """Everything that would make this backfill quietly incomplete, found before anything uploads.

    The failure this guards against is not a crash; it is a command that prints a cheerful summary
    having pushed five of six runs. A missing arm directory, an empty metrics file and a run that
    was never tracked by wandb all look like "nothing to do" one run at a time, and a per-run skip
    is invisible next to five successes. Checked up front and together, they are obviously wrong.

    Returns a list of problems; the caller decides whether ``--allow_missing`` forgives them.
    """
    problems: list[str] = []

    if wanted:
        # An explicit --only is its own completeness claim: these named runs, all of them.
        for name in sorted(wanted - {d.name for d in selected.values()}):
            problems.append(f"--only named {name}, which is not a run directory under {root}")
    else:
        for phase in phases:
            missing = [arm for arm in analysis.ARMS if f"{phase}-{arm}" not in
                       {d.name for d in selected.values()}]
            if missing == list(analysis.ARMS):
                problems.append(f"phase {phase!r} has no run directories under {root}")
            elif missing:
                problems.append(f"phase {phase!r} is missing the {', '.join(missing)} arm(s); "
                                f"the three arms are only comparable as a set")

    for key, run_dir in sorted(selected.items()):
        if not analysis.load_run(run_dir).problems:
            problems.append(f"{run_dir.name} has no per-problem rows in metrics.jsonl")
        if not read_run_id(run_dir):
            problems.append(f"{run_dir.name} has no wandb_run_id.txt, so it was never tracked by "
                            f"wandb and there is nothing to attach metrics to")
        elif not (project or detect_project(run_dir)):
            problems.append(f"{run_dir.name}: cannot determine its wandb project; pass --project")
    return problems


def main(root: str = "./outputs/harness", project: str | None = None, entity: str | None = None,
         window: int = 25, force: bool = False, dry_run: bool = False,
         phases: str = "adapt,heldout", only: str = "", skip_series: bool = False,
         allow_missing: bool = False) -> None:
    """Backfill every run under ``root`` that has both metrics and a wandb run id.

    ``python -m alphaapollo.core.harness.wandb_backfill --root ./outputs/harness --dry_run``

    ``--only heldout-evo,heldout-raw`` restricts the push to named run directories, and
    ``--skip_series`` sends only the overwritable summary and tables. Together with ``--force``
    they are the repair path: correcting a scalar on a run that was already pushed must not append
    a second copy of its curves. Cross-arm numbers are still computed from the whole phase, so a
    filtered run's ``analysis/common/*`` stays a comparison against the other arms.

    Exits non-zero if any run was attempted and failed, or if the preflight found the target set
    incomplete. A run that was deliberately skipped (already pushed) is not a failure; an
    authentication error is, and a shell that cannot tell the two apart will report a completely
    failed backfill as a success. ``--allow_missing`` downgrades the preflight to warnings, for a
    partly-run experiment or a jsonl-only run.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root_path = Path(root)
    wanted = {n.strip() for n in only.split(",") if n.strip()}
    phase_list = [p.strip() for p in phases.split(",") if p.strip()]

    # Discover first, check second, upload third. Nothing may be pushed before the whole target
    # set is known to be intact -- a preflight run per-arm inside the loop would already have
    # uploaded two arms by the time it found the third missing.
    found: dict[tuple[str, str], Path] = {}
    for phase in phase_list:
        for arm in analysis.ARMS:
            d = root_path / f"{phase}-{arm}"
            if (d / "metrics.jsonl").exists():
                found[(phase, arm)] = d
    selected = {f"{p}-{a}": d for (p, a), d in found.items() if not wanted or d.name in wanted}

    problems = preflight({k: v for k, v in selected.items()}, phases=phase_list, wanted=wanted,
                         root=root_path, project=project)
    if problems:
        for problem in problems:
            (logger.warning if (allow_missing or dry_run) else logger.error)("preflight: %s",
                                                                             problem)
        if not allow_missing and not dry_run:
            logger.error("refusing to push a partial backfill; fix the above or pass "
                         "--allow_missing to accept it")
            raise SystemExit(1)

    tally: dict[str, int] = defaultdict(int)
    for phase in phase_list:
        dirs = {arm: d for (p, arm), d in found.items() if p == phase}
        if not dirs:
            continue

        payloads = {arm: build_payload(d, name=arm, window=window) for arm, d in dirs.items()}
        runs = {arm: analysis.load_run(d, arm) for arm, d in dirs.items()}
        state_roots = {arm: r for arm, d in dirs.items() if (r := find_state_root(d)) is not None}
        # Phase-scoped on purpose: the adaptation stream and the held-out year are different
        # problem sets, so intersecting across them would compare nothing. Computed over every
        # arm found, not just the selected ones, so a --only repair keeps its comparison.
        add_cross_arm(payloads, runs, state_roots)

        for arm, d in sorted(dirs.items()):
            if d.name not in selected:
                continue
            tally[push(d, payloads[arm], project=project, entity=entity, force=force,
                       dry_run=dry_run, include_series=not skip_series)] += 1

    if dry_run:
        print(f"[backfill] dry run: wrote {PREVIEW_NAME} into {tally['dry_run']} run director"
              f"{'y' if tally['dry_run'] == 1 else 'ies'} under {root_path}")
    else:
        print(f"[backfill] pushed {tally['pushed']}, skipped {tally['skipped']}, "
              f"failed {tally['failed']}")
    if tally["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
