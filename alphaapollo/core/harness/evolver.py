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
"""Curate: turn a batch of Reflect-produced candidate skills into harness edits (paper Appendix
E.2/E.3's curation step).

Two curators sit at two different layers of the harness (see ``schema.Skill.level`` /
``store.Caps``):

- ``TopicCurator`` manages the skills of a single mathematical topic (one ``Caps.per_topic``
  slot budget). It sees only that topic's existing skills and only this batch's candidates for
  that topic.
- ``GeneralCurator`` manages the cross-topic layer (``Caps.general``). It sees the general
  skills and this batch's *entire* candidate pool (candidates from every topic), because a
  general skill is, by definition, a pattern that recurs across topics -- something a
  single-topic view could never establish.

Both curators are a thin LLM call plus a strict text parser (``parse_curator_output``): render a
prompt, invoke ``agent`` under the appropriate accounting role (``"topic_curator"`` /
``"general_curator"``, see ``accounting.MGMT_ROLES``), parse the reply into ``SkillEdit``s, and
bind each parsed edit's ``ADD``/``MERGE``/``REVISE`` payload back to the concrete candidate or
existing skill it refers to. Layer fields (``scope_hint``/``topic``) on every payload produced
here are then overwritten unconditionally by the calling curator's own layer -- the model's
opinion on which layer a skill belongs to is never trusted, only its content.

Failure handling is the same hard requirement in both curators, not an incidental detail: the
design doc (Section 5.5) and the assignment's Task B both require that a skill-update failure
degrade to a no-op rather than ever propagate up into (and break) the underlying, frozen
solver/verifier loop this harness is layered on top of. Concretely: an empty candidate batch
short-circuits to ``[]`` before any model call is made (also saves a call and avoids an empty
list rendering into the prompt); any exception raised by ``agent.get_action_from_gpt`` -- a
network error, a provider error, anything -- is caught, logged as a warning, and turned into
``[]``; and any candidate reference the parser cannot resolve (an out-of-range or non-numeric
candidate number, a payload missing one of its three sections) is dropped rather than turned
into a half-built ``SkillEdit``.
"""

from __future__ import annotations

import logging
import re

from alphaapollo.core.harness.accounting import role_scope
from alphaapollo.core.harness.reflect import _strip_reasoning
from alphaapollo.core.harness.schema import CandidateMemory, Skill, SkillEdit
from alphaapollo.core.harness.store import Caps

logger = logging.getLogger(__name__)

_COMMON_FORMAT = """For each decision output ONE block. Use EXACTLY these forms:

ADD: <candidate number>
REASON: <brief>

MERGE: <candidate number> INTO <existing skill id>
REASON: <brief>
TRIGGER: <one sentence>
LESSON:
- <bullets, 60 words total>
AVOID: <one sentence>

REVISE: <existing skill id>
REASON: <brief>
TRIGGER: <one sentence>
LESSON:
- <bullets, 60 words total>
AVOID: <one sentence>

DELETE: <existing skill id>
REASON: <brief>

SKIP: <candidate number>
REASON: <brief>

Write in English. Be SPECIFIC and ACTIONABLE, never generic advice like "read carefully"."""

TOPIC_CURATOR_PROMPT = """You curate topic-specific skills for a competition mathematics solver.

## Current skills for topic "{topic}" ({used}/{cap} slots used)
{existing}

## Candidate lessons from this batch
{candidates}

Decision criteria:
- GENERALIZABILITY TEST: keep a candidate only if it would be useful on MULTIPLE
  problems you have never seen. Judging that it accurately describes what just
  went wrong is NOT sufficient.
- Overlaps an existing skill -> MERGE (preferred over ADD).
- A candidate marked ENHANCE already names a target; prefer MERGE into that target.
- Budget full ({used}/{cap}) -> only MERGE, REVISE, DELETE or SKIP are allowed.
- One skill = one specific procedure. Few broad skills beat many narrow ones.
- Require a clear trigger description; keep content short and actionable.
- Low confidence -> SKIP.

{fmt}

If no candidate is worth keeping, output: NO_PROPOSALS"""

GENERAL_CURATOR_PROMPT = """You are a meta-learning curator. Analyse failure patterns ACROSS \
different topics to distil general skills that help on ANY mathematics problem.

## Current general skills ({used}/{cap} slots used)
{existing}

## Candidate lessons from this batch ({n} failed problems)
{candidates}

Your job:
- Find failure patterns that repeat across DIFFERENT problems and DIFFERENT topics.
- Each general skill must address a pattern seen in at least 2 (2+) different problems.
- GENERALIZABILITY TEST: a general skill must be useful on MULTIPLE unseen problems,
  and must encode procedures for planning, verification, recovery or tool use.
- General skills MUST avoid context-specific references. Reject any candidate that
  names a particular problem type, formula, or mathematical object.
- Do NOT create general skills for topic-specific procedures.
- Prefer REVISE over ADD when an existing general skill already covers the pattern.

{fmt}

If no cross-topic pattern is present, output: NO_PATTERNS"""

# One instruction "head" line, e.g. "ADD: 3" or "MERGE: 2 INTO sk_0007". Anchored to the start
# of a physical line (re.MULTILINE's ``^``) so that a verb-shaped substring appearing mid-sentence
# elsewhere in the reply (e.g. inside a REASON:) can never be mistaken for a real instruction --
# see parse_curator_output's use of finditer() below, which relies on every match being a genuine
# line start to delimit instruction blocks.
_HEAD = re.compile(r"^(ADD|MERGE|REVISE|DELETE|SKIP):\s*(\S+)(?:\s+INTO\s+(\S+))?\s*$",
                   re.MULTILINE)

# Payload sub-fields inside a MERGE/REVISE block. Restricted to ``[ \t]*`` around the label (never
# the newline-eating ``\s*``) for the same reason reflect.py's field regexes are: an anchored `^`
# right after a bare `\s*` would happily consume across a line boundary the caller never intended.
_REASON_RE = re.compile(r"^[ \t]*REASON[ \t]*:[ \t]*(.+)$", re.IGNORECASE | re.MULTILINE)
_TRIGGER_RE = re.compile(r"^[ \t]*TRIGGER[ \t]*:[ \t]*(.+)$", re.IGNORECASE | re.MULTILINE)
_LESSON_RE = re.compile(r"^[ \t]*LESSON[ \t]*:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
_AVOID_RE = re.compile(r"^[ \t]*AVOID[ \t]*:[ \t]*(.+)$", re.IGNORECASE | re.MULTILINE)


def _parse_block_payload(block: str) -> tuple[str, str, str, str] | None:
    """Parse ``REASON`` + ``TRIGGER`` + ``LESSON`` + ``AVOID`` out of one instruction block's
    body text (``block`` starts right after the head line). Returns ``None`` if any of the four
    is missing -- a partially-specified MERGE/REVISE is not safe to turn into a ``SkillEdit``
    payload, since ``Skill``/``CandidateMemory`` have no notion of an optional trigger/lesson/
    failure_mode.

    The LESSON body's end boundary is "the next line that starts with AVOID:", searched via
    ``_AVOID_RE.search(block, lesson_header.end())`` rather than a bare substring scan -- exactly
    the same reasoning as ``reflect.parse_reflection``'s LESSON/AVOID boundary: a lesson bullet
    that merely contains the word "avoid" mid-sentence must never be mistaken for the boundary.
    """
    reason_match = _REASON_RE.search(block)
    if reason_match is None:
        return None
    reason = reason_match.group(1).strip()

    trigger_match = _TRIGGER_RE.search(block)
    if trigger_match is None:
        return None
    trigger = trigger_match.group(1).strip()
    if not trigger:
        return None

    lesson_header = _LESSON_RE.search(block)
    if lesson_header is None:
        return None
    avoid_match = _AVOID_RE.search(block, lesson_header.end())
    if avoid_match is None:
        return None
    lesson_body = block[lesson_header.end() : avoid_match.start()]
    lesson = (lesson_header.group(1) + "\n" + lesson_body).strip()
    if not lesson:
        return None

    avoid = avoid_match.group(1).strip()
    if not avoid:
        return None

    return reason, trigger, lesson, avoid


def _parse_reason_only(block: str) -> str | None:
    """Parse just a ``REASON:`` line out of a DELETE/SKIP block's body -- these two ops carry
    no payload, so the block need not (and, per the format, does not) contain TRIGGER/LESSON/
    AVOID at all."""
    reason_match = _REASON_RE.search(block)
    if reason_match is None:
        return None
    return reason_match.group(1).strip()


def parse_curator_output(text: str, actor: str) -> list[SkillEdit]:
    """Parse a curator model reply into a list of ``SkillEdit``s.

    ``NO_PROPOSALS`` / ``NO_PATTERNS`` (or any reply with no recognizable instruction head at
    all -- pure noise, a refusal, chatty prose with nothing formatted) yield an empty list, never
    an exception: these are normal, expected outcomes of a curation call, not error states.

    A ``<think>...</think>`` reasoning-model prefix is stripped first, via
    ``reflect._strip_reasoning`` -- the exact same defense task 8 needed for ``parse_reflection``,
    and for the same underlying reason: every regex here uses first-match-by-position semantics,
    and a reasoning model's think-aloud text routinely rehearses instruction-head-shaped lines
    ("I could just say DELETE: sk_0003, but...") before the real formatted answer. Left
    unstripped, that rehearsal would be indistinguishable from a real instruction and would turn
    into a live edit against the harness -- worse here than in Reflect, since what is silently
    hijacked is the edit instruction itself, not just a classification field.

    Instruction blocks are delimited by consecutive ``_HEAD`` matches: each match starts a new
    block, and a block's body runs from the end of its head line to the start of the next head
    match (or the end of the text, for the last block) -- so a REASON/TRIGGER/AVOID sentence that
    happens to *contain* a verb-shaped substring (e.g. "REASON: better than a plain ADD: 1") can
    never be misread as a second head, because the substring never starts its own physical line
    beyond what the regex already excludes via ``^``/``re.MULTILINE`` anchoring... except when it
    does start its own line; that risk is why every payload sub-field regex used here is also
    line-anchored and searched only within the current block's slice, not the whole text.
    """
    text = _strip_reasoning(text or "")
    heads = list(_HEAD.finditer(text))
    if not heads:
        return []

    edits: list[SkillEdit] = []
    for i, head in enumerate(heads):
        op = head.group(1).upper()
        operand = head.group(2)
        into_id = head.group(3)
        block_end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        block = text[head.end():block_end]

        if op in ("DELETE",):
            reason = _parse_reason_only(block)
            if reason is None:
                continue
            edits.append(SkillEdit(op="DELETE", actor=actor, reason=reason, skill_id=operand))
            continue

        if op == "SKIP":
            reason = _parse_reason_only(block)
            if reason is None:
                continue
            edits.append(SkillEdit(op="SKIP", actor=actor, reason=reason, skill_id=None,
                                    payload=_CandidateRef(int_or_none(operand))))
            continue

        if op == "ADD":
            reason = _parse_reason_only(block)
            if reason is None:
                continue
            idx = int_or_none(operand)
            if idx is None:
                continue
            edits.append(SkillEdit(op="ADD", actor=actor, reason=reason, skill_id=None,
                                    payload=_CandidateRef(idx)))
            continue

        if op == "REVISE":
            parsed = _parse_block_payload(block)
            if parsed is None:
                continue
            reason, trigger, lesson, avoid = parsed
            edits.append(SkillEdit(op="REVISE", actor=actor, reason=reason, skill_id=operand,
                                    payload=_PayloadRef(trigger, lesson, avoid)))
            continue

        if op == "MERGE":
            if into_id is None:
                continue
            parsed = _parse_block_payload(block)
            if parsed is None:
                continue
            reason, trigger, lesson, avoid = parsed
            idx = int_or_none(operand)
            if idx is None:
                continue
            edits.append(SkillEdit(op="MERGE", actor=actor, reason=reason, skill_id=into_id,
                                    payload=_MergeRef(idx, trigger, lesson, avoid)))
            continue

    return edits


def int_or_none(token: str) -> int | None:
    """Parse ``token`` as a plain (non-negative, 1-based) candidate ordinal. Anything that is
    not purely digits -- including a leading ``-`` (Python's ``int()`` would happily accept
    ``"-1"``), a non-numeric word, or a decimal -- is rejected outright. ``0`` is syntactically a
    valid non-negative integer but is rejected by the caller (there is no candidate #0 under
    1-based numbering), not here, so that "not a number at all" and "a number but out of range"
    stay distinguishable failure modes if a future caller ever wants to log them differently.
    """
    if not token.isdigit():
        return None
    return int(token)


class _CandidateRef:
    """Placeholder carried inside a ``SkillEdit.payload`` slot between parsing and binding: the
    parser only knows a 1-based candidate ordinal, not the concrete ``CandidateMemory`` it refers
    to (that requires the caller's own candidate list, which the parser is never given). Bound to
    a real ``CandidateMemory`` by ``_bind_payloads`` before any edit reaches its caller; nothing
    outside this module ever observes a ``_CandidateRef``."""

    __slots__ = ("index",)

    def __init__(self, index: int | None) -> None:
        self.index = index


class _PayloadRef:
    """Placeholder for a REVISE's new content: parsed trigger/lesson/avoid text, not yet a
    ``CandidateMemory`` (it needs no candidate list -- REVISE rewrites an existing skill in
    place -- but still needs ``evidence``/``scope_hint``/``topic`` filled in by the caller's
    layer, which only ``_bind_payloads`` knows)."""

    __slots__ = ("trigger", "lesson", "avoid")

    def __init__(self, trigger: str, lesson: str, avoid: str) -> None:
        self.trigger = trigger
        self.lesson = lesson
        self.avoid = avoid


class _MergeRef(_PayloadRef):
    """Placeholder for a MERGE's new content: same three text fields as ``_PayloadRef`` plus the
    1-based candidate ordinal being merged in (so ``_bind_payloads`` can attach that candidate's
    ``evidence`` to the resulting skill)."""

    __slots__ = ("index",)

    def __init__(self, index: int | None, trigger: str, lesson: str, avoid: str) -> None:
        super().__init__(trigger, lesson, avoid)
        self.index = index


def _resolve_candidate(index: int | None, candidates: list[CandidateMemory]) -> CandidateMemory | None:
    """1-based ordinal -> the candidate it names, or ``None`` if the ordinal is missing, zero,
    negative, or out of range for ``candidates``. Bounds-checked here (rather than trusting the
    model to only ever cite a number that exists) because an out-of-range or garbage ordinal must
    make the whole edit disappear, never produce a half-built ``SkillEdit`` with a ``None``
    payload where a real one was expected."""
    if index is None or index < 1 or index > len(candidates):
        return None
    return candidates[index - 1]


def _bind_payloads(
    edits: list[SkillEdit],
    candidates: list[CandidateMemory],
    *,
    level: str,
    topic: str | None,
) -> list[SkillEdit]:
    """Resolve every parsed edit's placeholder payload (``_CandidateRef``/``_PayloadRef``/
    ``_MergeRef``) into a real ``CandidateMemory``, dropping any edit whose reference cannot be
    resolved (out-of-range candidate ordinal) rather than emitting a half-built edit.

    Every resulting payload's ``scope_hint``/``topic`` is then force-overwritten to ``level``/
    ``topic`` regardless of what the source candidate or the model's own text said -- the model
    is never trusted on which layer a skill belongs to; that is the calling curator's own layer,
    known only here, not by the parser.
    """
    bound: list[SkillEdit] = []
    for edit in edits:
        payload = edit.payload

        if payload is None:
            bound.append(edit)
            continue

        if isinstance(payload, _CandidateRef):
            source = _resolve_candidate(payload.index, candidates)
            if source is None:
                continue
            new_payload = CandidateMemory(
                trigger=source.trigger, lesson=source.lesson, failure_mode=source.failure_mode,
                scope_hint=level, topic=topic, evidence=list(source.evidence),
                action_hint=source.action_hint, target_id=source.target_id,
            )
            bound.append(SkillEdit(op=edit.op, actor=edit.actor, reason=edit.reason,
                                    skill_id=edit.skill_id, payload=new_payload))
            continue

        if isinstance(payload, _MergeRef):
            source = _resolve_candidate(payload.index, candidates)
            if source is None:
                continue
            new_payload = CandidateMemory(
                trigger=payload.trigger, lesson=payload.lesson, failure_mode=payload.avoid,
                scope_hint=level, topic=topic, evidence=list(source.evidence),
            )
            bound.append(SkillEdit(op=edit.op, actor=edit.actor, reason=edit.reason,
                                    skill_id=edit.skill_id, payload=new_payload))
            continue

        if isinstance(payload, _PayloadRef):
            # REVISE: no candidate ordinal to resolve, just the new text.
            new_payload = CandidateMemory(
                trigger=payload.trigger, lesson=payload.lesson, failure_mode=payload.avoid,
                scope_hint=level, topic=topic, evidence=[],
            )
            bound.append(SkillEdit(op=edit.op, actor=edit.actor, reason=edit.reason,
                                    skill_id=edit.skill_id, payload=new_payload))
            continue

        # Unknown payload shape: never seen in practice, but dropping rather than raising keeps
        # this function's no-throw contract intact even if a future op is added upstream without
        # a matching branch here.
        continue

    return bound


def _render_existing(existing: list[Skill]) -> str:
    if not existing:
        return "(none yet)"
    return "\n".join(f"- [{s.id}] {s.trigger} -- {s.lesson}" for s in existing)


def _render_candidates(candidates: list[CandidateMemory]) -> str:
    if not candidates:
        return "(none)"
    lines = []
    for i, c in enumerate(candidates, start=1):
        hint = f" [{c.action_hint} -> {c.target_id}]" if c.action_hint == "ENHANCE" and c.target_id else ""
        lines.append(f"{i}. {c.trigger}{hint}\n   Lesson: {c.lesson}\n   Avoid: {c.failure_mode}")
    return "\n".join(lines)


class _BaseCurator:
    """Shared curation flow for ``TopicCurator``/``GeneralCurator``: render a prompt, call the
    model under the appropriate accounting role, parse the reply, bind payloads to candidates,
    and force-overwrite every payload's layer fields. Subclasses supply only the prompt template,
    the accounting role name, and the layer (``level``/``topic``) they operate at.
    """

    role: str
    level: str

    def _build_prompt(self, *, existing: list[Skill], candidates: list[CandidateMemory], caps: Caps, topic: str | None) -> str:
        raise NotImplementedError

    def curate(self, agent, *, existing: list[Skill], candidates: list[CandidateMemory], caps: Caps, topic: str | None = None) -> list[SkillEdit]:
        if not candidates:
            return []

        prompt = self._build_prompt(existing=existing, candidates=candidates, caps=caps, topic=topic)

        try:
            with role_scope(self.role):
                reply = agent.get_action_from_gpt(prompt)
        except Exception:
            logger.warning("%s call failed; degrading to no-op for this batch", self.role, exc_info=True)
            return []

        try:
            edits = parse_curator_output(reply, actor=self.role)
            return _bind_payloads(edits, candidates, level=self.level, topic=topic if self.level == "topic" else None)
        except Exception:
            logger.warning("%s reply could not be parsed; degrading to no-op for this batch", self.role, exc_info=True)
            return []


class TopicCurator(_BaseCurator):
    """Manages one mathematical topic's skill slot (``Caps.per_topic``). Sees only that topic's
    existing skills and this batch's candidates already scoped to that topic."""

    role = "topic_curator"
    level = "topic"

    def _build_prompt(self, *, existing: list[Skill], candidates: list[CandidateMemory], caps: Caps, topic: str | None) -> str:
        used = sum(1 for s in existing if s.topic == topic)
        return TOPIC_CURATOR_PROMPT.format(
            topic=topic, used=used, cap=caps.per_topic,
            existing=_render_existing(existing), candidates=_render_candidates(candidates),
            fmt=_COMMON_FORMAT,
        )

    def curate(self, agent, *, existing: list[Skill], candidates: list[CandidateMemory], topic: str, caps: Caps) -> list[SkillEdit]:
        return super().curate(agent, existing=existing, candidates=candidates, caps=caps, topic=topic)


class GeneralCurator(_BaseCurator):
    """Manages the cross-topic skill layer (``Caps.general``). Sees the general skills and this
    batch's entire candidate pool (every topic at once), since a general skill is, by definition,
    a pattern that recurs across topics -- see the module docstring."""

    role = "general_curator"
    level = "general"

    def _build_prompt(self, *, existing: list[Skill], candidates: list[CandidateMemory], caps: Caps, topic: str | None) -> str:
        used = len(existing)
        return GENERAL_CURATOR_PROMPT.format(
            used=used, cap=caps.general, n=len(candidates),
            existing=_render_existing(existing), candidates=_render_candidates(candidates),
            fmt=_COMMON_FORMAT,
        )

    def curate(self, agent, *, existing: list[Skill], candidates: list[CandidateMemory], caps: Caps) -> list[SkillEdit]:
        return super().curate(agent, existing=existing, candidates=candidates, caps=caps, topic=None)
