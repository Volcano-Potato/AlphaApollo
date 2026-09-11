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
"""Persistent, cross-problem skill store.

This is the infrastructure half of Task A: a skill lives on disk under ``<root>/skills/*.md``
(one file per skill, in the markdown-with-frontmatter form defined by ``schema.skill_to_markdown``
/ ``skill_from_markdown``) plus two append-only JSONL logs that are deliberately kept separate:

- ``harness_log.jsonl`` -- the harness's own edit history: every add/merge/revise/delete, and
  every candidate that was proposed but *rejected*. This is what lets the exported harness be
  audited later ("what changed, and why").
- ``selection_log.jsonl`` -- the per-problem selection record: which skills were injected into
  which problem's context, at what token cost, alongside the resulting pass/fail signal. This is
  a different axis (retrieval, not editing) and must not be interleaved with the edit history --
  see ``test_selection_log_is_a_separate_file``, which asserts that logging a selection never
  creates ``harness_log.jsonl`` at all.

This module only implements persistence and logging. Skill *selection* (budget-aware retrieval
for a given problem) and *edit application* (turning a ``SkillEdit`` into an ADD/MERGE/REVISE/
DELETE on the store) are later tasks layered on top of ``SkillStore`` -- they are not present
here even though ``Caps``/``Budget`` (their configuration) are already defined, because later
tasks depend on the exact field names below.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alphaapollo.core.harness.render import count_tokens
from alphaapollo.core.harness.schema import Skill, skill_from_markdown, skill_to_markdown


@dataclass
class Caps:
    general: int = 5
    per_topic: int = 5


@dataclass
class Budget:
    b: int = 6
    general_max: int = 3
    topic_max: int = 4
    tokens: int = 800


class SkillStore:
    def __init__(self, root: str | Path, caps: Caps | None = None, budget: Budget | None = None):
        self.root = Path(root)
        self.caps = caps or Caps()
        self.budget = budget or Budget()
        self.skills_dir = self.root / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.harness_log = self.root / "harness_log.jsonl"
        self.selection_log = self.root / "selection_log.jsonl"
        self._skills: dict[str, Skill] = {}
        self.reload()

    # ---- persistence -------------------------------------------------
    def reload(self) -> None:
        """Rebuild in-memory state from disk. Used both at construction time and by tests /
        callers that want to confirm a write actually survived a fresh read, independent of
        whatever this process's in-memory dict currently holds."""
        self._skills = {}
        for path in sorted(self.skills_dir.glob("*.md")):
            skill = skill_from_markdown(path.read_text(encoding="utf-8"))
            self._skills[skill.id] = skill

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def snapshot(self) -> list[Skill]:
        """A deep copy of the current skill set, for freezing the harness at the start of a
        batch. A batch of problems must all see the same harness state regardless of what any
        problem *within* that batch produces -- a problem's own new skills may only affect
        later problems, never itself (and, by extension, never its own batch-mates evaluated
        concurrently). Returning a reference here would let a later ``_write_skill`` in the
        same batch leak into an already-snapshotted view; a deep copy makes that impossible."""
        return copy.deepcopy(self.all())

    @staticmethod
    def _validate_level_topic(skill: Skill) -> None:
        """Enforce the invariant ``level == "general" <=> topic is None``. A general-level
        skill is by definition not scoped to any one topic, and a topic-level skill is by
        definition scoped to one; a violation here is a caller bug, not a recoverable data
        state, so it raises rather than silently coercing one field to match the other."""
        if skill.level == "general" and skill.topic is not None:
            raise ValueError(f"skill {skill.id!r} has level='general' but topic={skill.topic!r} (must be None for a general skill)")
        if skill.level == "topic" and not skill.topic:
            raise ValueError(f"skill {skill.id!r} has level='topic' but topic is missing (a topic skill must set a non-empty topic)")

    def _write_skill(self, skill: Skill) -> None:
        self._validate_level_topic(skill)
        skill.n_tokens = count_tokens(f"{skill.trigger}\n{skill.lesson}\n{skill.failure_mode}")
        (self.skills_dir / f"{skill.name}.md").write_text(skill_to_markdown(skill), encoding="utf-8")
        self._skills[skill.id] = skill

    def _delete_skill(self, skill_id: str) -> None:
        skill = self._skills.pop(skill_id, None)
        if skill is not None:
            (self.skills_dir / f"{skill.name}.md").unlink(missing_ok=True)

    def _next_id(self) -> str:
        """Derived from the ids currently on disk (``self._skills``, populated by ``reload()``)
        rather than an in-memory counter, so numbering stays monotonic across process restarts:
        a fresh ``SkillStore`` over the same root must never hand out an id that collides with
        (and overwrites) a skill file already on disk."""
        used = [int(sid.split("_")[1]) for sid in self._skills if sid.startswith("sk_")]
        return f"sk_{(max(used) + 1 if used else 1):04d}"

    # ---- logging -----------------------------------------------------
    @staticmethod
    def _append(path: Path, payload: dict) -> None:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_event(self, **fields) -> None:
        self._append(self.harness_log, fields)

    def log_selection(self, **fields) -> None:
        self._append(self.selection_log, fields)
