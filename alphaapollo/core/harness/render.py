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
"""Render a bounded set of :class:`Skill` objects into the policy's system-message text.

Two things this module deliberately does NOT do:

1. It never falls back to an empty string. Upstream (``alphaapollo/core/generation/evolving/
   utils/agent.py``, ``if self.system_prompt:``) treats an empty string as "no system message at
   all" -- the message disappears rather than becoming an empty one. Since the three experiment
   arms (baseline / raw-experience / evo-harness) must differ in exactly one variable, and since
   the Evo arm's own harness starts empty and grows over time, ``render_harness([])`` returns
   ``NEUTRAL_SYSTEM_PROMPT`` so a system message with the same *shape* is always present,
   independent of how many skills happen to be selected for a given problem.
2. It never emits any of the bookkeeping fields on ``Skill`` (``id``, ``evidence``,
   ``created_at``, ``revised_at``, ``n_selected*``, ``n_tokens``). Those are internal harness
   state for selection/pruning (see ``schema.skill_to_markdown`` for the on-disk form that does
   include them) -- the policy model only ever sees the three natural-language sections.

``count_tokens`` is a plain regex-based heuristic, not a real tokenizer: it exists solely to
drive budget control (how many skills fit in the injected context), never to report the actual
token cost of a call -- that number comes from the API response's ``usage`` field.
"""

from __future__ import annotations

import math
import re

from alphaapollo.core.harness.schema import Skill

NEUTRAL_SYSTEM_PROMPT = "You are a competition mathematics solver."

_TOKEN = re.compile(r"\w+|[^\w\s]")

_HEADER = ("You are a competition mathematics solver. The following reusable skills were "
           "distilled from your own past attempts on other problems. Apply the ones that fit; "
           "ignore the ones that do not.")


def count_tokens(text: str) -> int:
    """Heuristic token-count estimate for budget control (not a real tokenizer)."""
    return math.ceil(len(_TOKEN.findall(text)) * 1.3)


def _render_one(skill: Skill) -> str:
    return (f"### {skill.name}\n"
            f"When to use: {skill.trigger}\n"
            f"Strategy:\n{skill.lesson}\n"
            f"Avoid: {skill.failure_mode}")


def render_harness(skills: list[Skill]) -> str:
    """Render ``skills`` (general-level first, then topic-level) into system-message text.

    Returns ``NEUTRAL_SYSTEM_PROMPT`` for an empty list -- see module docstring for why this
    must never be an empty string.
    """
    if not skills:
        return NEUTRAL_SYSTEM_PROMPT

    general = [s for s in skills if s.level == "general"]
    topic = [s for s in skills if s.level != "general"]

    parts = [_HEADER]
    if general:
        parts.append("## General strategies\n" + "\n\n".join(_render_one(s) for s in general))
    if topic:
        parts.append("## Topic-specific procedures\n" + "\n\n".join(_render_one(s) for s in topic))
    return "\n\n".join(parts)
