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
"""Crash recovery for a multi-hour adaptation run, at **batch** granularity.

An adaptation stream is ~15 hours of billed calls. Without this module a single interruption --
a dropped connection, a rate-limit storm, a stray Ctrl-C, an agent session tearing down its
background tasks -- costs the entire run, because ``SkillStore`` reloads its accumulated skills
from disk at construction while ``run_stream`` restarts the problem stream at index 0. That
combination is worse than losing the run: problem 0 would be solved with skills compiled from
problems it has not seen yet, which is exactly the future-information leak the assignment
forbids ("一道题自己产生的新技能只能影响后面的题"). A resumed run must therefore either be
protocol-exact or refuse to start.

**Why batches and not problems.** ``arm.end_batch()`` is the only point at which the harness may
change; skills and ``harness_log`` rows appear on disk only there. A crash midway through batch
5 leaves the store holding exactly what batch 4 produced, while ``metrics.jsonl`` and
``selection_log.jsonl`` already carry rows for however many of batch 5's problems finished. The
protocol-exact resume point is therefore the *start of batch 5*, never "the next unfinished
problem": re-running the whole batch reproduces the state the batch was meant to see. It costs
at most ``batch_size`` problems of duplicated compute, which is the cheapest correct option.

**Why the partial rows must go.** Those already-written rows describe problems that are about to
run again. Left in place they double-count in every success rate and every token total, and the
duplicates are indistinguishable from genuine records after the fact. :func:`truncate_jsonl` drops
them before the rerun. ``harness_log.jsonl`` and ``skills/`` need no truncation -- nothing writes
to them outside ``end_batch``, so they are already consistent with the resume point.

**Why a fingerprint.** Pointing a resume at a different stream, arm, or batch size would silently
produce a run whose first half and second half came from different experiments. The recorded
fingerprint makes that a startup error instead (:class:`ResumeMismatch`).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROGRESS_FILENAME = "progress.json"

# The config fields that change what a run *is*. A mismatch on any of them means the progress
# marker belongs to a different experiment. `max_workers` is deliberately absent: it is a
# throughput knob that changes wall-clock only, never which problem sees which harness state
# (see evolving_harness_main's module docstring on batch_size vs max_workers), so resuming with
# a different worker count is legitimate -- useful when the first attempt died to rate limits.
_FINGERPRINT_FIELDS = ("stream_path", "arm", "batch_size", "seed", "frozen")


class ResumeMismatch(RuntimeError):
    """Raised when a run directory holds progress from a materially different run."""


@dataclass(frozen=True)
class ResumePlan:
    """What a resumed run must do before it starts calling the model.

    ``start_batch`` is fed to ``run_stream``; ``first_problem_idx`` is the stream position that
    batch begins at, and the cutoff every partial-row truncation uses.
    """

    start_batch: int
    first_problem_idx: int

    @property
    def is_fresh(self) -> bool:
        return self.start_batch == 0


def fingerprint(harness_cfg: dict) -> dict:
    """The identity of a run, for comparing a progress marker against the current config."""
    return {k: harness_cfg.get(k) for k in _FINGERPRINT_FIELDS}


def read_progress(run_dir: str | Path) -> dict | None:
    """Return the recorded progress, or ``None`` when there is none to resume from.

    A corrupt marker returns ``None`` rather than raising: the file is written atomically, so
    corruption means the very first write was interrupted, which is indistinguishable from a run
    that never completed a batch. Both resume from scratch.
    """
    path = Path(run_dir) / PROGRESS_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("ignoring unreadable progress marker at %s: %s", path, exc)
        return None


def write_progress(run_dir: str | Path, *, completed_batches: int, fp: dict) -> Path:
    """Record that ``completed_batches`` batches are durably done.

    Written to a temporary file and then ``os.replace``d, which is atomic on POSIX. A plain
    in-place write that is interrupted halfway leaves a truncated JSON file, and this marker is
    the one artifact whose corruption would make a resume silently start from the wrong batch --
    the failure mode this whole module exists to prevent.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / PROGRESS_FILENAME
    tmp = run_dir / f".{PROGRESS_FILENAME}.tmp"
    tmp.write_text(json.dumps({"completed_batches": completed_batches, "fingerprint": fp},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def truncate_jsonl(path: str | Path, *, key: str, first_dropped: int) -> int:
    """Drop rows whose ``row[key] >= first_dropped``; return how many were dropped.

    Rewrites through a temporary file and ``os.replace`` so an interruption during truncation
    cannot leave a half-written log -- the same reasoning as :func:`write_progress`.

    Rows without ``key``, and rows whose ``key`` is not an integer, are **kept**. The logs this
    runs against mix per-problem rows with per-batch rows (``metrics.jsonl`` carries harness-scale
    and call-accounting lines keyed by the batch's last problem, plus per-problem pass/fail lines);
    a row that does not carry a comparable position is not one this function can judge, and
    silently discarding it would lose accounting the rerun does not regenerate.
    """
    path = Path(path)
    if not path.exists():
        return 0

    kept: list[str] = []
    dropped = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)  # a torn final line is evidence; do not quietly delete it
                continue
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                kept.append(line)
                continue
            if value >= first_dropped:
                dropped += 1
            else:
                kept.append(line)

    if dropped:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(kept), encoding="utf-8")
        os.replace(tmp, path)
    return dropped


def plan_resume(run_dir: str | Path, *, fp: dict, batch_size: int) -> ResumePlan:
    """Decide where a run starts, and refuse outright when the marker is from another run.

    Raises :class:`ResumeMismatch` rather than starting fresh on a fingerprint mismatch. Starting
    fresh would overwrite a real run's artifacts; starting where the marker says would splice two
    experiments together. Neither is recoverable after the fact, so this is one of the places the
    harness deliberately refuses to degrade.
    """
    progress = read_progress(run_dir)
    if progress is None:
        return ResumePlan(start_batch=0, first_problem_idx=0)

    recorded = progress.get("fingerprint") or {}
    if recorded != fp:
        differing = {k: (recorded.get(k), fp.get(k)) for k in set(recorded) | set(fp)
                     if recorded.get(k) != fp.get(k)}
        raise ResumeMismatch(
            f"{Path(run_dir) / PROGRESS_FILENAME} records a different run; refusing to resume. "
            f"Differences (recorded -> current): {differing}. Point --config's harness.run_dir at "
            f"a fresh directory, or delete the progress marker to restart this one from zero."
        )

    completed = int(progress.get("completed_batches") or 0)
    return ResumePlan(start_batch=completed, first_problem_idx=completed * batch_size)


LOCK_FILENAME = "run.lock"


class RunAlreadyActive(RuntimeError):
    """Raised when another live process already owns this run directory."""


def _pid_is_alive(pid: int) -> bool:
    """``os.kill(pid, 0)`` raises if the pid is gone, returns if it exists.

    ``PermissionError`` counts as alive: the process exists, it just belongs to someone else.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_run_lock(run_dir: str | Path) -> Path:
    """Claim ``run_dir`` for this process, or refuse if a live process already holds it.

    Two processes writing one run directory interleave their ``metrics.jsonl`` rows and their
    progress markers, and the result is quiet: the file stays valid JSONL, the run stays
    "successful", and the only trace is duplicated problem indices that every downstream average
    then double-counts. It happened here during testing, and the path to it in a real run is
    short -- a long run looks stalled, gets relaunched, and the two processes race with the
    original still alive.

    A stale lock (the recorded pid is gone) is taken over rather than treated as an error: a run
    killed by SIGKILL never gets to clean up, and refusing to restart after a crash would break
    the recovery this module exists to provide.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / LOCK_FILENAME

    if path.exists():
        try:
            holder = int(path.read_text(encoding="utf-8").strip() or 0)
        except (ValueError, OSError):
            holder = 0
        if holder and holder != os.getpid() and _pid_is_alive(holder):
            raise RunAlreadyActive(
                f"{path} is held by live process {holder}. Two processes writing one run "
                f"directory interleave their metrics and corrupt every average computed from "
                f"them, without failing. Wait for it, kill it, or point harness.run_dir "
                f"somewhere else."
            )
        if holder:
            logger.info("taking over a stale lock from pid %s", holder)

    path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    return path


def release_run_lock(run_dir: str | Path) -> None:
    """Drop this process's claim. Never raises -- a run that finished its work must not fail in
    cleanup, and a lock left behind is taken over by the next process anyway."""
    path = Path(run_dir) / LOCK_FILENAME
    try:
        if path.exists() and path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        logger.warning("could not release %s; the next run will take it over as stale", path)


def prepare_resume(run_dir: str | Path, *, fp: dict, batch_size: int,
                   partial_logs: dict[str, str]) -> ResumePlan:
    """Plan the resume and drop every partial row the aborted batch left behind.

    ``partial_logs`` maps a path to the row field holding that log's stream position --
    ``metrics.jsonl`` keys on ``"step"``, the store's ``selection_log.jsonl`` on ``"problem_idx"``.
    """
    plan = plan_resume(run_dir, fp=fp, batch_size=batch_size)
    if plan.is_fresh:
        return plan

    for path, key in partial_logs.items():
        dropped = truncate_jsonl(path, key=key, first_dropped=plan.first_problem_idx)
        if dropped:
            logger.info("resume: dropped %d partial row(s) from %s (problem_idx >= %d)",
                        dropped, path, plan.first_problem_idx)

    logger.info("resume: %d batch(es) already complete; restarting at batch %d (problem %d)",
                plan.start_batch, plan.start_batch, plan.first_problem_idx)
    return plan
