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
"""Persist each problem's full rollout, so the write-up's case studies do not require a rerun.

``metrics.jsonl`` keeps two integers per problem (``pass1_round0``, ``pass_final``). That is
enough to plot every curve the assignment asks for, and not remotely enough for the thing it asks
for next: "至少 2-3 个具体的正/负迁移案例". A transfer case is an argument about *what the model
did differently* when a skill was injected -- which needs the rollout text, not its verdict.

Upstream already serialises exactly this payload (``utils.save_problem_outputs``), but from its
own ``run()``, which this driver replaces; calling ``run_problem`` directly skips it. Rather than
adopt upstream's output path (``<project>/outputs/<dataset>/<tag>/<model>/test_<n>/``, which
scatters one run's artifacts across a tree keyed by things this experiment does not vary), the
sink here writes under the run's own ``run_dir`` so that a single directory holds everything one
run produced -- what makes the run both resumable and backup-able as a unit.

Only upstream's ``convert_to_serializable`` is reused, because the payload contains objects
(numpy scalars, env wrappers) that plain ``json.dumps`` rejects.

**This file is written for the analyst, not for the method.** Nothing in the harness reads it
back; no selection, reflection, or curation decision may depend on it. It is also the one
artifact here that contains the ground truth (upstream stamps ``ground_truth`` onto the payload
and onto every ``step_outputs`` entry), which is precisely why it stays out of the method's
reach -- see ``reflect.build_reflect_context``'s whitelist signature for the boundary that keeps
it out.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DIRNAME = "trajectories"


def trajectory_path(run_dir: str | Path, problem_idx: int) -> Path:
    """Zero-padded so a plain directory listing sorts in stream order."""
    return Path(run_dir) / DIRNAME / f"problem_{problem_idx:04d}.json"


def save_trajectory(run_dir: str | Path, problem_idx: int, payload: dict) -> Path | None:
    """Write one problem's rollout; return the path, or ``None`` if it could not be written.

    Never raises. A full disk or an unserialisable payload must not abort an adaptation run whose
    real product -- the solver results and the compiled harness -- is already safely on disk. The
    failure is logged loudly enough to notice, and the run continues.
    """
    from alphaapollo.core.generation.evolving.utils.utils import convert_to_serializable

    path = trajectory_path(run_dir, problem_idx)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(convert_to_serializable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
    except Exception:  # noqa: BLE001 -- an analysis artifact may never break the run
        logger.exception("could not save the trajectory for problem %s; continuing", problem_idx)
        return None
