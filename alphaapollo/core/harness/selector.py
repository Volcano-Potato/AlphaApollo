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
"""Model-driven harness selection -- the ``Select(x, H; b)`` of paper Algorithm 1, line 5.

Appendix F states it plainly: *"For harness selection, we use Claude Sonnet 4.5 across all
experiments to retrieve relevant skills from the current harness before task execution."*
Selection is a model call that reads the question and the harness, not a lexical filter. An
earlier version of this package used deterministic content-word overlap plus a hard topic filter;
that was chosen for reproducibility, but it also silently required every problem to carry a topic
label, which is a dependency the method does not have.

**The budget is enforced here, not delegated to the model.** Task A requires injected content to
respect a configurable count/token budget, and a model told about a budget will sometimes exceed
it. The model chooses *which* skills and in *what order*; this module then applies the count cap,
the per-level quotas and the cumulative token cap deterministically. That split is what keeps the
budget requirement testable rather than hoped-for, and it means a badly-behaved reply degrades
into a smaller selection instead of an oversized prompt.

**Failure degrades to an empty selection.** Assignment Task B: a skill-mechanism failure must
never break the underlying AlphaApollo baseline. If the selector call raises, the problem simply
runs with the neutral system prompt -- structurally identical to the Baseline arm -- rather than
taking the run down with it.
"""

from __future__ import annotations

import logging
import re

from alphaapollo.core.harness.accounting import role_scope
from alphaapollo.core.harness.reflect import _strip_reasoning
from alphaapollo.core.harness.schema import Skill
from alphaapollo.core.harness.store import Budget

logger = logging.getLogger(__name__)

SELECT_PROMPT = """You are about to solve a competition mathematics problem. Choose which of your \
existing skills are worth loading into your context for THIS problem.

Problem:
{question}

## Your skill harness
{harness}

Reply with the ids of the skills to load, most useful first, separated by commas. Reply with \
exactly NONE if none of them apply.

Choose at most {b} of them. Prefer a smaller, sharper set: an irrelevant skill costs context and \
can mislead. Do not explain your choice."""

_ID_RE = re.compile(r"\bsk_\d+\b")


def _render_harness(skills: list[Skill]) -> str:
    lines = []
    for skill in skills:
        scope = "general" if skill.level == "general" else f"topic:{skill.topic}"
        lines.append(f"- [{skill.id}] ({scope}) {skill.trigger}")
    return "\n".join(lines)


def parse_selection(text: str) -> list[str]:
    """Pull skill ids out of a selector reply, in the order the model gave them, de-duplicated.

    ``<think>`` blocks are stripped first. A reasoning model rehearses candidate ids while
    deliberating ("Maybe sk_0002? No, that is geometry."), and a bare id scan over the raw reply
    would collect the rehearsal along with -- or instead of -- the actual answer. This is the same
    hijack already handled in ``reflect`` and the curators.
    """
    cleaned = _strip_reasoning(text or "")
    seen: list[str] = []
    for match in _ID_RE.findall(cleaned):
        if match not in seen:
            seen.append(match)
    return seen


def apply_budget(skills: list[Skill], budget: Budget) -> list[Skill]:
    """Trim an ordered selection to the count cap, per-level quotas and cumulative token cap.

    A skill whose tokens would overflow the cap is skipped rather than ending the walk, so a
    smaller lower-ranked skill can still use the remaining headroom; one larger than the whole
    budget is simply never admitted.
    """
    picked: list[Skill] = []
    used_tokens = n_general = n_topic = 0
    for skill in skills:
        if len(picked) >= budget.b:
            break
        if skill.level == "general":
            if n_general >= budget.general_max:
                continue
        elif n_topic >= budget.topic_max:
            continue
        if used_tokens + skill.n_tokens > budget.tokens:
            continue
        picked.append(skill)
        used_tokens += skill.n_tokens
        if skill.level == "general":
            n_general += 1
        else:
            n_topic += 1
    return picked


def select_skills(agent, question: str, skills: list[Skill], budget: Budget) -> list[Skill]:
    """Ask ``agent`` which skills to load for ``question``, then enforce ``budget`` on the answer.

    Returns ``[]`` for an empty harness (without spending a call), for a reply that names nothing
    usable, and for a failed call. An empty selection is a legitimate answer, never back-filled
    with "best available": with an irrelevant harness, injecting nothing is the correct choice.
    """
    if not skills:
        return []

    prompt = SELECT_PROMPT.format(question=question, harness=_render_harness(skills), b=budget.b)
    try:
        with role_scope("selector"):
            reply = agent.get_action_from_gpt(prompt)
    except Exception:
        logger.warning("harness selection failed; this problem runs with no injected skills", exc_info=True)
        return []

    by_id = {skill.id: skill for skill in skills}
    # A hallucinated id is dropped rather than fabricated into a skill.
    chosen = [by_id[sid] for sid in parse_selection(reply) if sid in by_id]
    return apply_budget(chosen, budget)
