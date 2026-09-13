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
"""Metrics tracking for the cross-problem skill harness: wandb + jsonl dual-write.

The `informal_math_evolving` / evo path has no wandb integration at all today -- the only
`wandb` references in this repo are in the `rl_*`/`sft_*` verl-trainer configs and in
`setup.py`'s dependency list, neither of which this harness touches. Task C requires reporting
several curves (adaptation success over time, harness growth, token/call overhead) that need
*some* run-tracking layer, so this module builds a minimal one from scratch, purpose-built for
the harness rather than reused from the verl trainer's wandb usage (which is wired directly into
its own training loop and assumes a very different lifecycle).

Two hard requirements drive the design:

1. wandb must never be a point of failure. A multi-hour adaptation-stream run must not die
   because wandb isn't installed, the user isn't logged in, the network is down, or the wandb
   cache directory isn't writable. Every wandb interaction (import, `wandb.init()`, and every
   individual `wandb.log()`/`wandb.finish()` call) is therefore wrapped in a broad
   ``except Exception`` and degrades to "just keep writing jsonl" rather than propagating.
2. jsonl is the reproducible artifact. wandb dashboards are not something this repo can ship or
   version; every chart in the README/report must be re-plottable offline from
   ``<run_dir>/metrics.jsonl`` alone. wandb, when available, only ever gets a *copy* of what was
   already written to that file.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# wandb is strictly optional here -- the assignment never asks for it, and `metrics.jsonl` is the
# real record. But an *enabled* wandb on a machine that has never run `wandb login` is the default
# state of a fresh checkout, and it must not be able to hurt the run. Measured on wandb 0.30 under
# a real PTY: `wandb.init()` without credentials blocks ~4s on an interactive API-key prompt and
# then raises `KeyboardInterrupt` -- a BaseException that `except Exception` does not catch. The
# tracker constructor propagated it and killed the process before one metrics.jsonl line existed.
_WANDB_NETRC_HOST = "api.wandb.ai"
# Modes that log locally and never authenticate, so they are legitimate without credentials.
_OFFLINE_MODES = frozenset({"offline", "dryrun", "disabled"})


def _netrc_has_wandb() -> bool:
    """Is there a stored `wandb login` credential? Checked by reading the netrc file directly
    rather than via `netrc.netrc()`, whose parser raises on entries it dislikes -- on a shared
    machine that would turn an unrelated netrc line into a wandb outage."""
    for name in (".netrc", "_netrc"):
        path = Path.home() / name
        try:
            if path.exists() and _WANDB_NETRC_HOST in path.read_text():
                return True
        except OSError:
            continue
    return False


def _wandb_can_authenticate() -> bool:
    """Whether reaching ``wandb.init()`` could succeed without a terminal prompt."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    if str(os.environ.get("WANDB_MODE", "")).strip().lower() in _OFFLINE_MODES:
        return True
    return _netrc_has_wandb()


def _json_default(value: Any) -> Any:
    """Fallback encoder for ``json.dumps`` so metric values that aren't plain Python types
    (most commonly numpy scalars such as ``np.float64``/``np.int64``, which real accuracy and
    token-average computations naturally produce) never crash the jsonl writer.

    numpy scalar types all expose a zero-argument ``.item()`` that returns the equivalent
    plain Python ``int``/``float``; that is tried first since it round-trips exactly. Anything
    else (a ``Path``, an arbitrary object with a useful ``__str__``) falls back to ``str()``
    rather than raising, since a metrics writer must never be the reason a run aborts.
    """
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


class HarnessTracker:
    """Dual-writes run metrics to ``<run_dir>/metrics.jsonl`` and, best-effort, to wandb.

    jsonl is unconditional and always succeeds if the filesystem does; wandb is opportunistic
    and any failure at any stage (import, init, log, finish) silently falls back to jsonl-only
    for the rest of the tracker's lifetime.
    """

    def __init__(self, run_dir: str | Path, *, project: str | None = None,
                 run_name: str | None = None, config: dict | None = None, enabled: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

        self.wandb_run = None
        if enabled:
            self.wandb_run = self._try_init_wandb(project, run_name, config)

    def _try_init_wandb(self, project: str | None, run_name: str | None, config: dict | None):
        """Best-effort wandb init. Returns ``None`` (never raises) on any failure: wandb not
        installed, not logged in, network unreachable, or the run dir not writable by wandb's
        own cache -- all observed as different exception types depending on wandb version and
        transport, so the catch is deliberately broad rather than enumerated.
        """
        try:
            import wandb
        except Exception as exc:
            logger.info("wandb unavailable, logging to jsonl only: %s", exc)
            return None

        # Checked BEFORE calling init: catching the interrupt is not enough, because merely
        # reaching `wandb.init()` means blocking on a terminal prompt and dumping a signup banner
        # into the run log. With no key, no netrc entry and no offline mode there is nothing to
        # authenticate against, so there is no reason to ask.
        if not _wandb_can_authenticate():
            logger.warning("wandb is enabled but no credentials were found (WANDB_API_KEY, netrc, "
                           "or WANDB_MODE=offline); logging to jsonl only. Run `wandb login` to enable it.")
            return None

        try:
            return wandb.init(project=project, name=run_name, config=config, dir=str(self.run_dir))
        except (Exception, KeyboardInterrupt) as exc:
            # KeyboardInterrupt is listed explicitly: it is what wandb raises when its interactive
            # prompt gets no input, and it is a BaseException, so `except Exception` misses it.
            # Losing a multi-hour run to an optional logging backend is the worse failure; a real
            # Ctrl-C during this sub-second call is vanishingly unlikely.
            logger.warning("wandb.init() failed, logging to jsonl only: %r", exc)
            return None

    def log(self, step: int, metrics: dict) -> None:
        """Append one jsonl row ``{"step": step, **metrics}`` and, if wandb is live, forward
        the same metrics to it. The jsonl append happens unconditionally and first, so a wandb
        failure on this call can never cost the jsonl record.
        """
        row = {"step": step, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

        if self.wandb_run is not None:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as exc:
                logger.warning("wandb.log() failed at step %s, continuing jsonl-only: %s", step, exc)

    def log_harness_state(self, step: int, store) -> None:
        """Compute harness-scale metrics live from ``store`` (never cached, since the harness
        mutates between batches): general/topic skill counts, per-topic counts, total and mean
        skill token length, and every skill's usage (hit count) and utility.

        The harness has a hard capacity of 5 general + 5-per-topic skills, so
        ``harness/n_general``/``harness/n_topic_total`` saturate and flatten after the first
        few adaptation batches -- ``harness/total_tokens`` and ``harness/mean_skill_tokens`` are
        what keep carrying signal afterward, tracking the harness's "refinement" phase where
        MERGE/REVISE change skill length without changing skill count.
        """
        skills = store.all()
        general = [s for s in skills if s.level == "general"]
        topic_skills = [s for s in skills if s.level != "general"]

        per_topic: dict[str, int] = {}
        for s in topic_skills:
            per_topic[s.topic] = per_topic.get(s.topic, 0) + 1

        total_tokens = sum(s.n_tokens for s in skills)

        metrics: dict = {
            "harness/n_general": len(general),
            "harness/n_topic_total": len(topic_skills),
            "harness/total_tokens": total_tokens,
            "harness/mean_skill_tokens": (total_tokens / len(skills)) if skills else 0,
        }
        for topic, n in per_topic.items():
            metrics[f"harness/n_topic/{topic}"] = n
        for s in skills:
            metrics[f"usage/skill_hit/{s.id}"] = s.n_selected
            metrics[f"usage/skill_utility/{s.id}"] = s.utility()

        self.log(step, metrics)

    def log_accounting(self, step: int, accountant) -> None:
        """Forward ``CallAccountant.snapshot()`` verbatim -- that dict is already exactly the
        ``calls/<role>`` / ``tokens/<role>_in`` / ``tokens/<role>_out`` /
        ``calls/solver_side_total`` / ``calls/mgmt_side_total`` shape this needs, so this method
        does no reshaping of its own.
        """
        self.log(step, accountant.snapshot())

    def finish(self) -> None:
        """Close the wandb run, if one is open. Safe to call more than once: the first call
        tears down ``self.wandb_run`` and sets it to ``None``, so every subsequent call is a
        no-op regardless of whether the first ``.finish()`` itself raised.
        """
        if self.wandb_run is None:
            return
        run = self.wandb_run
        self.wandb_run = None
        try:
            run.finish()
        except (Exception, KeyboardInterrupt) as exc:
            # finish() runs in the driver's `finally`; anything escaping here would mask whatever
            # actually ended the run. Same reasoning as `_try_init_wandb`'s catch.
            logger.warning("wandb.finish() failed: %r", exc)
