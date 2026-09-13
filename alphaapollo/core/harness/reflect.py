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
"""Reflect: turn one completed (and, in practice, failed) problem attempt into a candidate
cross-problem skill (paper Appendix E.1's proposal step).

This is a *whitelist* boundary, not a "please don't peek" convention. ``build_reflect_context()``
takes the raw materials Reflect is allowed to see as individually named keyword arguments -- there
is no ``question`` parameter, no answer-string parameter, and no ``**kwargs``/dict escape hatch
through which either could be smuggled in. If a caller wants to leak the problem statement or the
correct answer into a skill, it cannot do so through this function's signature; the type system
of the call site, not developer discipline, is what blocks it. For the same reason this module
must never define or import anything named with the literal identifier this docstring is
carefully avoiding (see ``tests/harness/test_reflect.py``, which scans this module's source via
``inspect.getsource`` for that exact identifier) -- the one place in this package allowed to touch
that value is ``guard.py``, which only ever *rejects* a candidate, never constructs one.

The other leak this module closes is more mundane: AlphaApollo's own Python verification tool
(``core/tools/informalmath_verify.py:107,176``) writes a line like ``Matches ground truth: True``
or ``Matches GT: False`` directly into the tool's return text, which ``env.py:106`` then wraps in
a ``<tool_response>`` block and feeds back to the policy -- and from there it ends up in whatever
verifier/tool-error text this module is handed. ``sanitize_feedback()`` strips that channel by
deleting the whole offending line (it is emitted as an independent line by the verifier tool, so
line-level deletion is precise and never touches surrounding text).

``feedback_level`` implements the design doc's Section 11 ablation: ``"minimal"`` gives Reflect
only the tool-error text (no verifier report at all -- the pass/fail outcome itself is still
visible via the ``{outcome}`` slot in the prompt, since that is a top-level context field, not
part of the withheld report); ``"standard"`` additionally includes the full verifier feedback.
This exists because the paper's Table 4 shows that letting a self-evaluating LLM verifier's
report drive skill compilation can make the harness worse over successive rounds, and AlphaApollo's
verifier is itself an LLM -- the ablation is how Task C's arms measure that risk instead of
assuming it away.

``parse_reflection()`` never raises on malformed model output; it returns ``None``. A model call
that itself raises (a network error, a provider error) is a different failure mode and is left to
propagate -- the caller (the Task B orchestration loop) decides whether to degrade gracefully
there. Conflating "the model refused/timed out" with "the model answered but not in the format we
asked for" would hide two very different failure classes behind one return value.
"""

from __future__ import annotations

import re

from alphaapollo.core.harness.accounting import role_scope
from alphaapollo.core.harness.schema import ACTION_HINTS, SCOPE_HINTS, CandidateMemory, Skill

# The two textual shapes AlphaApollo's own verifier tool emits for the answer-matching line
# (see module docstring). Matched as a whole line, case-insensitively, so that indentation or
# a trailing remark on the same physical line still gets removed in one piece -- the channel is
# always emitted as its own independent line, never interleaved with other content mid-line.
_GT_CHANNEL = re.compile(r"^.*(matches ground truth|matches gt)\s*:.*$", re.IGNORECASE | re.MULTILINE)

# `accounting.install_accounting` (preserving upstream `utils/agent.py` behavior) prepends a
# reasoning model's raw `reasoning_content`/`reasoning` field to every reply, wrapped in exactly
# this tag, whenever the served model is a reasoning model. `_THINK_BLOCK` removes a
# well-formed `<think>...</think>` pair; the body is non-greedy so consecutive blocks
# ("<think>A</think>text<think>B</think>") are each matched and removed individually rather than
# one match spanning both and swallowing the text between them. `_THINK_UNCLOSED` is a second,
# separate pass for the case no `</think>` ever appears: everything from that tag to the end of
# the string is dropped, on the theory that a formatted answer was never actually produced yet,
# so guessing where "real" content might resume would be worse than discarding it.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_UNCLOSED = re.compile(r"<think>.*\Z", re.IGNORECASE | re.DOTALL)

REFLECT_PROMPT = """You just failed a competition mathematics problem. Distill ONE reusable \
lesson that would help you on FUTURE, DIFFERENT problems.

Answer you produced: {final_answer_given}
Outcome: {outcome}
Rounds used: {round_count}
{feedback_block}
Your reasoning (excerpt):
{reasoning_excerpt}

## Related skills already in your harness
{related_skills}

## Topics already in your harness
{existing_topics}

Write in English. Output EXACTLY this format and nothing else:

ACTION: NEW | ENHANCE | NONE
TARGET: <existing skill id>          (only when ACTION is ENHANCE)
TOPIC: <a broad topic name, lowercase, 1-3 words>
SCOPE: general | topic
TRIGGER: <one sentence naming the situation where this applies>
LESSON:
- <bullet, imperative, at most 3 bullets, 60 words total>
AVOID: <one sentence naming the failure mode>

Filter aggressively. Output ACTION: NONE rather than proposing:
- generic advice such as "read carefully" or "double-check the work";
- basic tool usage that any solver already knows;
- an exact replay of this task, its statement, or its numeric answer;
- anything that would not help on a problem you have never seen.

Other rules:
- If a listed existing skill already covers this lesson, use ACTION: ENHANCE with its id.
- Describe the METHOD, never the problem statement or its numeric answer.
- TOPIC: reuse one of the topics listed above whenever it fits; only name a new one when none do.
- SCOPE: general only if it would help on a DIFFERENT mathematical topic."""

_NO_RELATED_SKILLS = "(none yet)"
_NO_EXISTING_TOPICS = "(none yet -- you are naming the first one)"

# A topic name is a bucket key, so two spellings of the same idea are two buckets. The harness
# caps skills per topic, so unchecked fragmentation ("Number Theory" / "number theory" /
# "number-theory") ends with every bucket holding one skill and no layer at all. Canonicalisation
# is intentionally shallow -- case, separators, surrounding punctuation and a trailing "problems"
# -- because anything cleverer would start merging topics the model meant to keep apart. Real
# synonym collapsing is the prompt's job: it is shown the existing topic names and told to reuse
# one where it fits.
_TOPIC_RE = re.compile(r"^[ \t]*TOPIC[ \t]*:[ \t]*(.+)$", re.IGNORECASE | re.MULTILINE)
_TOPIC_STRIP = re.compile(r"[^a-z0-9]+")
_TOPIC_SUFFIX = re.compile(r"_(problems?|questions?|tasks?)$")
_MAX_TOPIC_WORDS = 3


def normalise_topic(raw: str | None) -> str | None:
    """Canonicalise a model-chosen topic name into a stable bucket key, or ``None`` if unusable.

    Names longer than three words are rejected rather than truncated: past that length the model
    has written a description of this one problem ("counting lattice paths under a divisibility
    constraint") instead of a topic, and truncating it would produce a bucket key that looks
    legitimate while still matching nothing else.
    """
    if raw is None:
        return None
    name = _TOPIC_STRIP.sub("_", str(raw).strip().lower()).strip("_")
    name = _TOPIC_SUFFIX.sub("", name)
    if not name or name in {"none", "n_a", "na", "unknown", "general"}:
        return None
    if len(name.split("_")) > _MAX_TOPIC_WORDS:
        return None
    return name

# Field-label regexes. Leading/trailing whitespace around a label and its colon is restricted to
# spaces/tabs (`[ \t]*`), never the full `\s*` class -- `\s` also matches newlines, and a leading
# `\s*` placed right after a `^` anchor (or a trailing `\s*` placed right before a greedy `.*`)
# would be able to swallow across a line boundary the caller never intended it to cross. Using
# `[ \t]*` keeps every one of these anchored to a single physical line, which is what makes the
# LESSON/AVOID boundary search below well-defined.
_ACTION_RE = re.compile(r"^[ \t]*ACTION[ \t]*:[ \t]*(\S+)", re.IGNORECASE | re.MULTILINE)
_TARGET_RE = re.compile(r"^[ \t]*TARGET[ \t]*:[ \t]*(\S+)", re.IGNORECASE | re.MULTILINE)
_SCOPE_RE = re.compile(r"^[ \t]*SCOPE[ \t]*:[ \t]*(\S+)", re.IGNORECASE | re.MULTILINE)
_TRIGGER_RE = re.compile(r"^[ \t]*TRIGGER[ \t]*:[ \t]*(.+)$", re.IGNORECASE | re.MULTILINE)
_LESSON_RE = re.compile(r"^[ \t]*LESSON[ \t]*:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
_AVOID_RE = re.compile(r"^[ \t]*AVOID[ \t]*:[ \t]*(.+)$", re.IGNORECASE | re.MULTILINE)


def _strip_reasoning(text: str) -> str:
    """Remove every ``<think>...</think>`` reasoning block from ``text`` (see the ``_THINK_BLOCK``
    / ``_THINK_UNCLOSED`` comment above for why this exists and how the two passes divide the
    work). Applied before any field parsing, and before the GT-channel line scrub, since a
    reasoning model's think-aloud text can itself contain either an answer-matching line (if it
    is quoting/paraphrasing a tool response while thinking) or field-label-shaped text (if it is
    rehearsing candidate answers, e.g. "...should this be ACTION: NONE? No..."). Every regex
    downstream of this function uses ``re.search``/``re.match``, which return the *first* match
    by position -- an unstripped think block sitting before the real formatted answer would
    silently win that first match instead of raising anything, which is exactly the failure mode
    this function exists to prevent."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_UNCLOSED.sub("", text)
    return text


def sanitize_feedback(text: str) -> str:
    """Strip AlphaApollo's answer-matching channel (see module docstring) from ``text``, after
    first removing any ``<think>...</think>`` reasoning block -- the verifier's report text
    flows through the same ``Agent.get_action_from_gpt`` path as everything else in this module,
    so it can carry the same reasoning-model wrapper. Offending lines are deleted in full rather
    than edited around."""
    text = _strip_reasoning(text)
    return "\n".join(line for line in text.split("\n") if not _GT_CHANNEL.match(line))


def _render_related(related_skills: list[Skill]) -> str:
    """Render already-selected skills for the "related skills already in your harness" prompt
    section. Every one of ``related_skills`` has already passed ``guard.validate_skill()`` (it
    came from ``store.select()``), so surfacing its id and trigger here is safe -- and doing so
    is what keeps Reflect from repeatedly re-proposing something the harness already has (see
    the ``related_skills`` parameter note below). An empty list still renders a non-empty
    placeholder, never an empty string: an empty string would leave a dangling, content-free
    "## Related skills already in your harness" heading in the prompt.
    """
    if not related_skills:
        return _NO_RELATED_SKILLS
    return "\n".join(f"- [{skill.id}] {skill.trigger}" for skill in related_skills)


def _render_topics(existing_topics: list[str]) -> str:
    seen: list[str] = []
    for topic in existing_topics:
        name = normalise_topic(topic)
        if name and name not in seen:
            seen.append(name)
    if not seen:
        return _NO_EXISTING_TOPICS
    return "\n".join(f"- {name}" for name in sorted(seen))


def build_reflect_context(*, existing_topics: list[str], final_answer_given: str, outcome: str, verifier_feedback: str, tool_errors: str, reasoning_excerpt: str, round_count: int, related_skills: list[Skill]) -> dict:
    """Assemble the whitelisted materials Reflect is allowed to see, as a plain dict ready for
    ``reflect()``. Every keyword here is listed explicitly (no ``**kwargs``, no dict parameter) --
    see the module docstring for why that is the actual anti-leakage mechanism, not just style.

    ``related_skills`` mirrors paper Appendix E.1's proposal-step input "related existing
    skills": pass exactly the skills the selector already chose for this problem. Doing so costs
    nothing extra (they are already in hand) and, per the paper, keeps Reflect from proposing
    lessons the harness already holds under a different id -- without them the curator spends its
    fixed budget on near-duplicates instead of genuinely new coverage.

    ``existing_topics`` is NOT the current problem's topic -- there is no such input. Paper
    Appendix E.1 lists the proposal step's inputs as "evaluation result, verifier details or
    rubric feedback, trajectory signals, compressed trajectory, and related existing skills", and
    instructs the model to *"Propose a reusable skill. Choose a broad topic..."*. The topic is
    therefore something the model names on the way out, from what it just experienced -- not a
    dataset label handed in on the way. An earlier version of this module had that backwards,
    which manufactured a dependency on per-problem topic annotations that the method does not
    actually have.

    The existing topic names are passed purely so the model can *reuse* one rather than coining a
    synonym. Without that, free-form naming fragments the buckets ("number theory" / "modular
    arithmetic" / "divisibility"), and since the harness caps skills per topic, every bucket ends
    up holding one skill and the topic layer stops being a layer.

    ``verifier_feedback`` and ``tool_errors`` are passed through ``sanitize_feedback()`` here, at
    context-build time, so that every downstream consumer of this dict (today: ``reflect()``; any
    future logging of the context itself) sees already-scrubbed text rather than having to
    remember to scrub it again.
    """
    return {
        "existing_topics": _render_topics(existing_topics),
        "final_answer_given": final_answer_given,
        "outcome": outcome,
        "verifier_feedback": sanitize_feedback(verifier_feedback),
        "tool_errors": sanitize_feedback(tool_errors),
        "reasoning_excerpt": reasoning_excerpt,
        "round_count": round_count,
        "related_skills": _render_related(related_skills),
    }


def _feedback_block(context: dict, feedback_level: str) -> str:
    """Render the verifier/tool-error section of the prompt, gated by ``feedback_level`` (design
    doc Section 11's required ablation -- see module docstring). The pass/fail outcome itself is
    always visible via the prompt's separate ``{outcome}`` slot regardless of this setting; what
    ``"minimal"`` withholds is specifically the verifier's own *report* text.
    """
    tool_errors = context["tool_errors"].strip() or "(none)"
    if feedback_level == "minimal":
        return f"Tool errors: {tool_errors}"
    verifier_feedback = context["verifier_feedback"].strip() or "(none)"
    return f"Verifier feedback: {verifier_feedback}\nTool errors: {tool_errors}"


def reflect(agent, context: dict, *, feedback_level: str = "standard") -> CandidateMemory | None:
    """Run one Reflect call: build the prompt from ``context`` (as produced by
    ``build_reflect_context()``), invoke ``agent`` under the ``"reflect"`` accounting role (see
    ``accounting.MGMT_ROLES``, so this call is counted as skill-management cost, never solver
    cost), and parse the reply into a candidate.

    Returns ``None`` if the model's reply cannot be parsed into a well-formed candidate (see
    ``parse_reflection``). Does not catch exceptions raised by ``agent.get_action_from_gpt`` --
    those propagate to the caller, which is a different failure mode than "the model answered but
    not in the requested format" (see module docstring).
    """
    prompt = REFLECT_PROMPT.format(
        existing_topics=context["existing_topics"],
        final_answer_given=context["final_answer_given"],
        outcome=context["outcome"],
        round_count=context["round_count"],
        feedback_block=_feedback_block(context, feedback_level),
        reasoning_excerpt=context["reasoning_excerpt"],
        related_skills=context["related_skills"],
    )
    with role_scope("reflect"):
        reply = agent.get_action_from_gpt(prompt)

    # No topic backfill: the model names the topic itself (paper Appendix E.1), so
    # `parse_reflection` reads it straight off the reply.
    return parse_reflection(reply)


def parse_reflection(text: str) -> CandidateMemory | None:
    """Parse a Reflect model reply into a ``CandidateMemory``, or ``None`` if there is nothing
    reusable to propose or the reply does not conform to the requested format.

    Deliberately conservative: a reply that is missing any of the four required fields (SCOPE,
    TRIGGER, LESSON, AVOID) yields ``None``, never a partially-populated candidate. The one
    field this is lenient about is ACTION: an absent ``ACTION:`` line defaults to ``"NEW"`` (for
    backward compatibility with a model that omits it), but an ``ACTION:`` line that names
    anything other than NEW/ENHANCE/NONE is treated the same as a missing required field, since
    at that point the reply is not reliably following the requested format either.

    ``ACTION: NONE`` and ``SCOPE: none`` are both accepted as "nothing to propose" (models do not
    always follow the ACTION: NONE convention exactly), and both short-circuit to ``None`` before
    any of the four required fields are even checked.

    A ``<think>...</think>`` reasoning block, if present, is stripped before any of the field
    regexes run (see ``_strip_reasoning``) -- otherwise a reasoning model's think-aloud text,
    which comes *before* the formatted answer and routinely rehearses field-label-shaped text
    while reasoning out loud, would win every ``re.search``'s first-match-by-position semantics
    instead of the real, later answer.
    """
    text = _strip_reasoning(text or "")
    if not text.strip():
        return None

    action_match = _ACTION_RE.search(text)
    if action_match is None:
        action = "NEW"
    else:
        action = action_match.group(1).upper()
        if action == "NONE":
            return None
        if action not in ACTION_HINTS:
            return None

    scope_match = _SCOPE_RE.search(text)
    if scope_match is None:
        return None
    scope = scope_match.group(1).lower()
    if scope == "none":
        return None
    if scope not in SCOPE_HINTS:
        return None

    trigger_match = _TRIGGER_RE.search(text)
    if trigger_match is None:
        return None
    trigger = trigger_match.group(1).strip()
    if not trigger:
        return None

    lesson_header = _LESSON_RE.search(text)
    if lesson_header is None:
        return None
    # LESSON is multi-line; its end boundary is the next line that starts with "AVOID:" --
    # searched for starting right after the LESSON header line, never a bare substring search,
    # so a bullet that merely *contains* the word "avoid" mid-sentence can never be mistaken for
    # the boundary (see test_lesson_bullet_containing_the_word_avoid_does_not_confuse_the_boundary).
    avoid_match = _AVOID_RE.search(text, lesson_header.end())
    if avoid_match is None:
        return None
    lesson_body = text[lesson_header.end() : avoid_match.start()]
    lesson = (lesson_header.group(1) + "\n" + lesson_body).strip()
    if not lesson:
        return None

    failure_mode = avoid_match.group(1).strip()
    if not failure_mode:
        return None

    target_id = None
    if action == "ENHANCE":
        target_match = _TARGET_RE.search(text)
        if target_match is not None:
            target_id = target_match.group(1).strip()

    # The model names its own topic (paper Appendix E.1: "Choose a broad topic"). A topic-scoped
    # candidate with no usable topic name has no bucket to live in, so it is rejected outright
    # rather than filed under a placeholder that matches nothing.
    #
    # The name is kept even when SCOPE is general. Algorithm 1 feeds the *same* proposal pool to
    # CompileTaskType and to the cross-task compile step, so a proposal that arose while doing
    # number theory still belongs in that topic's curator call even if the model judged the
    # lesson itself to be cross-task. `evolver._bind_payloads` is what finally nulls the topic
    # for anything the GeneralCurator accepts, since the store rejects a general skill carrying
    # a topic.
    topic_match = _TOPIC_RE.search(text)
    topic = normalise_topic(topic_match.group(1)) if topic_match else None
    if scope == "topic" and not topic:
        return None

    return CandidateMemory(
        trigger=trigger,
        lesson=lesson,
        failure_mode=failure_mode,
        scope_hint=scope,
        topic=topic,
        action_hint=action,
        target_id=target_id,
    )
