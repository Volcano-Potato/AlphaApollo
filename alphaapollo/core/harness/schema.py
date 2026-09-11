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
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

OPS = frozenset({"ADD", "MERGE", "REVISE", "DELETE", "SKIP"})
LEVELS = frozenset({"general", "topic"})
SCOPE_HINTS = frozenset({"general", "topic"})
ACTION_HINTS = frozenset({"NEW", "ENHANCE"})


@dataclass
class Skill:
    id: str
    name: str
    level: str
    topic: str | None
    trigger: str
    lesson: str
    failure_mode: str
    evidence: list[str] = field(default_factory=list)
    created_at: int = 0
    revised_at: list[int] = field(default_factory=list)
    n_selected: int = 0
    n_selected_success: int = 0
    n_tokens: int = 0

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {sorted(LEVELS)}, got {self.level!r}")

    def utility(self) -> float:
        return (self.n_selected_success + 1) / (self.n_selected + 2)


@dataclass
class CandidateMemory:
    trigger: str
    lesson: str
    failure_mode: str
    scope_hint: str
    topic: str | None
    evidence: list[str] = field(default_factory=list)
    # Paper Appendix E.1: the proposal step decides NEW / ENHANCE / NONE.
    # NONE is represented by returning no CandidateMemory at all.
    action_hint: str = "NEW"
    target_id: str | None = None

    def __post_init__(self) -> None:
        if self.scope_hint not in SCOPE_HINTS:
            raise ValueError(f"scope_hint must be one of {sorted(SCOPE_HINTS)}, got {self.scope_hint!r}")
        if self.action_hint not in ACTION_HINTS:
            raise ValueError(f"action_hint must be one of {sorted(ACTION_HINTS)}, got {self.action_hint!r}")


@dataclass
class SkillEdit:
    op: str
    actor: str
    reason: str
    skill_id: str | None = None
    payload: CandidateMemory | None = None

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"op must be one of {sorted(OPS)}, got {self.op!r}")


_FRONTMATTER_KEYS = ("id", "name", "level", "topic", "evidence", "created_at",
                     "revised_at", "n_selected", "n_selected_success", "n_tokens")


def skill_to_markdown(skill: Skill) -> str:
    meta: dict[str, Any] = {k: getattr(skill, k) for k in _FRONTMATTER_KEYS}
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    lines += [
        "---",
        "## When to use",
        skill.trigger.strip(),
        "",
        "## Strategy",
        skill.lesson.strip(),
        "",
        "## Avoid",
        skill.failure_mode.strip(),
        "",
    ]
    return "\n".join(lines)


def skill_from_markdown(text: str) -> Skill:
    _, frontmatter, body = text.split("---", 2)
    meta = {}
    for line in frontmatter.strip().splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = json.loads(v.strip())

    sections: dict[str, list[str]] = {}
    current = None
    for line in body.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    def section(name: str) -> str:
        return "\n".join(sections.get(name, [])).strip()

    return Skill(
        trigger=section("When to use"),
        lesson=section("Strategy"),
        failure_mode=section("Avoid"),
        **meta,
    )
