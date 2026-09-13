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
"""Task C's three cross-problem arms, wired on top of Task A/B's infrastructure.

The three arms answer three different questions and must be **structurally identical** in
everything except their cross-problem mechanism (assignment Task C: "identical model, task
order, in-problem evolution rounds, tools, and generation params"):

- ``BaselineArm``       -- no cross-problem mechanism at all. The raw performance floor.
- ``RawExperienceArm``  -- stores a short free-text summary of every completed trajectory
                           (success AND failure) and injects the most relevant ones verbatim.
                           This is the "why not just paste history into the context?" strawman,
                           and it must be a *serious* one: same injection budget as the Evo arm,
                           same batch-boundary discipline, same non-empty-prompt guarantee.
- ``EvoHarnessArm``     -- the Task A/B mechanism: a persistent, budget-capped ``SkillStore``
                           that only learns from *failures* (paper eq. 6) via Reflect + the two
                           curators, and whose harness is injected via ``render_harness``.

Two invariants apply to every arm, not just one:

1. **Non-empty system prompt, always.** ``utils/agent.py:37`` is ``if self.system_prompt:`` --
   an empty string deletes the system message outright rather than sending an empty one. If the
   baseline arm ever returned ``""`` while the other two returned real text, the three arms would
   differ in message *structure*, not just content -- a confound Task C's "fair, order-matched
   comparison" explicitly rules out. ``CrossProblemArm``'s default `system_prompt_for` therefore
   returns ``NEUTRAL_SYSTEM_PROMPT`` (see ``render.py``), never ``""``, and both ``RawExperienceArm``
   and ``EvoHarnessArm`` fall back to the same constant whenever their pool/harness is empty (e.g.
   the very first batch), so the cold-start shape is identical across all three.

2. **A problem's own outcome can never affect the prompt it itself just saw.** Every stateful
   arm freezes its cross-problem state at ``begin_batch()`` (a deep-ish copy) and every
   ``system_prompt_for()`` call for the rest of that batch reads *only* that frozen copy, never
   the live, mutable store/pool. Only ``end_batch()`` is allowed to write to the live state. This
   is what makes "no future-problem information, and a problem's new skills only affect later
   problems" (assignment Task B) a structural guarantee instead of a call-order convention: even
   a caller who mutates the live store mid-batch (bypassing this class entirely) cannot change
   what an in-flight batch's problems see, because they were never reading the live state to
   begin with.
"""

from __future__ import annotations

import logging
import re

from alphaapollo.core.harness.accounting import role_scope
from alphaapollo.core.harness.evolver import GeneralCurator, TopicCurator
from alphaapollo.core.harness.reflect import _strip_reasoning, build_reflect_context, reflect, sanitize_feedback
from alphaapollo.core.harness.render import NEUTRAL_SYSTEM_PROMPT, count_tokens, render_harness
from alphaapollo.core.harness.schema import CandidateMemory, Skill, SkillEdit
from alphaapollo.core.harness.store import Budget, Caps, SkillStore

logger = logging.getLogger(__name__)


def has_trajectory(result: dict) -> bool:
    """Did this problem actually produce a rollout to learn from?

    ``run_stream`` degrades any problem whose ``run_problem`` raised into an all-zero result so
    that one flaky call cannot abort an entire adaptation stream. That degradation is correct, but
    it makes a dead problem indistinguishable from a genuine wrong answer if the only thing a
    cross-problem arm checks is ``pass_final == 0`` -- and the two must be treated differently.
    A wrong answer is precisely the signal this mechanism exists to compile; a problem that never
    ran carries no signal at all, and feeding its empty context to Reflect just prompts the model
    to invent a confident, unfalsifiable skill out of nothing and merge it into the harness.

    The first real end-to-end run made this concrete: all four problems died on
    ``KeyError: 'data_source'``, and the harness still came back with four fluent skills compiled
    from four empty trajectories -- a silent, expensive, and completely undetectable-from-metrics
    corruption of the experiment.

    ``round_count`` is the discriminator rather than any text field: it is non-zero as soon as the
    solver completed one evolving round, so a degenerate-but-real rollout (wrong answer, no
    salvageable reasoning excerpt) still counts as a trajectory, while the all-zero shape never
    does.
    """
    return int(result.get("round_count", 0) or 0) > 0


# --- verbatim contract constants (mandated wording, not implementation detail) --------------

_RAW_HEADER = ("You are a competition mathematics solver. Summaries of your own previous "
               "attempts on other problems follow. Use them if relevant.")

_SUMMARY_PROMPT = """Summarise, in at most 40 English words, what went right or wrong in this \
attempt in a way that could help on a DIFFERENT problem. Do not restate the problem or its answer.

Topic: {topic}
Outcome: {outcome}
Verifier feedback: {verifier_feedback}
Reasoning excerpt: {reasoning_excerpt}"""


# --- shared, dependency-free lexical-overlap scoring ----------------------------------------
#
# Both RawExperienceArm's raw-summary ranking and EvoHarnessArm's snapshot-local skill ranking
# need the exact same "how relevant is this text to the current question" primitive, and neither
# can call ``SkillStore.select()`` directly for it (see ``_select_from_snapshot`` below for why).
# Rather than reach into ``store.py``'s underscore-prefixed helpers from a sibling module, this
# is a small, self-contained reimplementation -- deliberately the same dumb, model-free formula
# (fraction of one side's content words that also appear on the other), so a difference in
# selection quality is never itself the reason two arms diverge.

_WORD_RE = re.compile(r"[A-Za-z']+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that the "
    "this to was were when where which with you your".split()
)


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def _overlap(a: str, b: str) -> float:
    """Fraction of ``a``'s content words that also appear in ``b``, in [0, 1]. 0.0 for an
    empty-after-stopwords ``a`` rather than raising a division-by-zero."""
    words_a = _content_words(a)
    if not words_a:
        return 0.0
    return len(words_a & _content_words(b)) / len(words_a)


def _select_from_snapshot(skills: list[Skill], question: str, topic: str | None, budget: Budget) -> list[Skill]:
    """``SkillStore.select()``'s exact algorithm (score, per-level quotas, token cap), but run
    over an explicit, already-frozen ``skills`` list instead of the store's own live
    ``self._skills``.

    This duplication is deliberate, not an oversight: ``SkillStore.select()`` only ever reads
    live store state, so calling it directly from ``EvoHarnessArm.system_prompt_for`` would mean
    a ``record_usage()`` call for one problem in a batch (which mutates the live store's utility
    counters) could change what a *later* problem in the same batch sees -- exactly the
    within-batch leak invariant (2) in the module docstring forbids. Selecting from a
    ``store.snapshot()`` taken once at ``begin_batch()`` closes that hole structurally.
    """
    candidates = [s for s in skills if s.level == "general" or s.topic == topic]
    ranked = sorted(candidates, key=lambda s: (-(0.3 * s.utility() + 0.7 * _overlap(s.trigger, question)), s.id))

    picked: list[Skill] = []
    used_tokens = 0
    n_general = n_topic = 0
    for skill in ranked:
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


def _select_raw(pool: list[dict], question: str, budget: Budget) -> list[dict]:
    """Pick up to ``budget.b`` raw-experience entries under the cumulative ``budget.tokens``
    cap, ranked by lexical overlap between the entry's summary text and ``question`` -- the same
    ``b``/``tokens`` fields ``EvoHarnessArm`` reads off its own ``Budget``, so the two arms'
    injection sizes are never a confound (see the module docstring's "serious opponent" note).
    ``entry["problem_idx"]`` is a pure tiebreaker, exactly as ``Skill.id`` is in
    ``_select_from_snapshot``, so the ranking is total and reproducible."""
    ranked = sorted(pool, key=lambda e: (-_overlap(e["summary"], question), e.get("problem_idx", 0)))
    picked: list[dict] = []
    used_tokens = 0
    for entry in ranked:
        if len(picked) >= budget.b:
            break
        if used_tokens + entry["n_tokens"] > budget.tokens:
            continue
        picked.append(entry)
        used_tokens += entry["n_tokens"]
    return picked


def _render_raw(entries: list[dict]) -> str:
    if not entries:
        return NEUTRAL_SYSTEM_PROMPT
    lines = [_RAW_HEADER, ""]
    lines.extend(f"{i}. {entry['summary']}" for i, entry in enumerate(entries, start=1))
    return "\n".join(lines)


# --- the protocol base -----------------------------------------------------------------------


class CrossProblemArm:
    """Protocol base for a cross-problem mechanism. Every method is a safe, neutral default --
    a new arm only overrides what it actually needs; ``BaselineArm`` needs none of them, which is
    exactly the point (see module docstring invariant 1 for why the default `system_prompt_for`
    must be ``NEUTRAL_SYSTEM_PROMPT`` and never ``""``)."""

    def begin_batch(self, batch_idx: int) -> None:
        """Called once before a batch of problems is solved. Freeze whatever cross-problem
        state exists into a batch-local view; see invariant 2."""
        return None

    def system_prompt_for(self, problem: dict) -> str:
        """The system-message text to inject for this ``problem``, computed from the frozen
        view established by the most recent ``begin_batch()``. Never returns ``""``."""
        return NEUTRAL_SYSTEM_PROMPT

    def record_selection(self, problem: dict, result: dict) -> None:
        """Called once ``result`` (the problem's outcome) is known, so the arm can update usage
        statistics and log which cross-problem material was injected. Must be safe to call even
        if ``system_prompt_for`` was never called for this ``problem`` (nothing was selected)."""
        return None

    def observe(self, problem: dict, result: dict) -> None:
        """Accumulate evidence from a completed problem. Must never mutate any state that
        ``system_prompt_for`` reads -- only ``end_batch()`` may do that."""
        return None

    def end_batch(self, batch_idx: int) -> list[dict]:
        """Compile whatever was ``observe()``-d this batch into the arm's persistent state, and
        return one record per attempted update (empty list if nothing was updated)."""
        return []


class BaselineArm(CrossProblemArm):
    """No cross-problem mechanism whatsoever: the raw performance floor. Answers "what does the
    frozen policy/verifier loop already get right or wrong, with only its existing in-problem
    memory?" Needs no overrides -- every ``CrossProblemArm`` default already does the right
    thing (a constant neutral prompt, no state to freeze or compile)."""


class RawExperienceArm(CrossProblemArm):
    """Stores a short free-text summary of every completed trajectory -- success AND failure,
    unlike ``EvoHarnessArm`` -- and injects the most relevant ones verbatim, under the exact same
    ``b``/``tokens`` budget as the Evo arm. This is the "why not just paste history in?" arm; the
    only thing that may differ from ``EvoHarnessArm`` is *what* gets stored (a raw summary vs. a
    curated skill), never *how much* gets injected.

    Callers running the Task C experiment must pass the *same* ``Budget`` values (or object) to
    this arm and to ``EvoHarnessArm`` -- both default independently to ``Budget()``, which is
    sufficient unless either arm is given a custom budget.
    """

    def __init__(self, *, agent, budget: Budget | None = None):
        self.agent = agent
        self.budget = budget or Budget()
        self._pool: list[dict] = []
        self._frozen_pool: list[dict] = []
        self._pending: list[tuple[dict, dict]] = []

    def begin_batch(self, batch_idx: int) -> None:
        self._frozen_pool = [dict(e) for e in self._pool]
        self._pending = []

    def system_prompt_for(self, problem: dict) -> str:
        picked = _select_raw(self._frozen_pool, problem.get("question", ""), self.budget)
        return _render_raw(picked)

    def observe(self, problem: dict, result: dict) -> None:
        # Dropped at the observe boundary, not inside end_batch, so a dead problem never even
        # enters this arm's pending list -- see `has_trajectory`.
        if not has_trajectory(result):
            logger.warning("problem %s produced no trajectory; not summarizing it into the raw pool",
                           problem.get("problem_idx"))
            return
        self._pending.append((problem, result))

    def end_batch(self, batch_idx: int) -> list[dict]:
        added: list[dict] = []
        for problem, result in self._pending:
            try:
                prompt = _SUMMARY_PROMPT.format(
                    topic=problem.get("topic", ""),
                    outcome="passed" if result.get("pass_final") else "failed",
                    verifier_feedback=sanitize_feedback(result.get("verifier_feedback", "")),
                    reasoning_excerpt=result.get("reasoning_excerpt", ""),
                )
                # "raw_summarizer", never "summarizer" -- see accounting.py's MGMT_ROLES comment.
                # This is this arm's entire cross-problem management overhead (one call per
                # completed problem), and Task C requires it reported as management-side cost,
                # not folded into the upstream in-problem summariser's solver-side bucket.
                with role_scope("raw_summarizer"):
                    reply = self.agent.get_action_from_gpt(prompt)
                summary = _strip_reasoning(reply or "").strip()
                if not summary:
                    raise ValueError("summarizer returned no usable text")
            except Exception:
                logger.warning("raw-experience summarizer failed for problem %s; skipping",
                                problem.get("problem_idx"), exc_info=True)
                continue
            entry = {
                "problem_idx": problem.get("problem_idx"),
                "topic": problem.get("topic"),
                "batch": batch_idx,
                "summary": summary,
                "n_tokens": count_tokens(summary),
            }
            self._pool.append(entry)
            added.append(entry)
        self._pending = []
        return added


class EvoHarnessArm(CrossProblemArm):
    """The Task A/B skill-compilation loop: select from a persistent ``SkillStore`` -> inject via
    ``render_harness`` -> run to completion (outside this class) -> Reflect *only the failures*
    (paper eq. 6) -> curate per-topic then cross-topic -> apply under the store's hard capacity
    bounds. ``frozen=True`` makes ``end_batch`` a complete no-op (no Reflect call, no curator
    call, no log line, no store mutation) -- this is what a held-out evaluation run relies on to
    guarantee the harness never changes underneath it.
    """

    def __init__(self, *, store_root, agent, caps: Caps | None = None, budget: Budget | None = None,
                 feedback_level: str = "standard", frozen: bool = False):
        self.store = SkillStore(store_root, caps=caps, budget=budget)
        self.agent = agent
        self.feedback_level = feedback_level
        self.frozen = frozen
        self._frozen_snapshot: list[Skill] = []
        self._pending: list[tuple[dict, dict]] = []
        self._selections: dict[object, list[Skill]] = {}

    def begin_batch(self, batch_idx: int) -> None:
        self._frozen_snapshot = self.store.snapshot()
        self._pending = []
        self._selections = {}

    def system_prompt_for(self, problem: dict) -> str:
        picked = _select_from_snapshot(self._frozen_snapshot, problem.get("question", ""),
                                        problem.get("topic"), self.store.budget)
        self._selections[problem.get("problem_idx")] = picked
        return render_harness(picked)

    def record_selection(self, problem: dict, result: dict) -> None:
        picked = self._selections.get(problem.get("problem_idx"), [])
        success = bool(result.get("pass_final"))
        self.store.record_usage(picked, success)
        self.store.log_selection(
            problem_idx=problem.get("problem_idx"),
            topic=problem.get("topic"),
            skill_ids=[s.id for s in picked],
            n_tokens=sum(s.n_tokens for s in picked),
            success=success,
        )

    def observe(self, problem: dict, result: dict) -> None:
        # Dropped at the observe boundary, not inside end_batch, so a dead problem can never
        # reach Reflect, contribute a `p_<idx>` evidence tag, or spend a management call on an
        # empty context -- see `has_trajectory`.
        if not has_trajectory(result):
            logger.warning("problem %s produced no trajectory; not reflecting on it",
                           problem.get("problem_idx"))
            return
        self._pending.append((problem, result))

    def end_batch(self, batch_idx: int) -> list[dict]:
        if self.frozen:
            return []

        failed = [(p, r) for p, r in self._pending if not r.get("pass_final")]
        self._pending = []
        if not failed:
            return []

        # 1. Reflect -- only on failures (paper eq. 6). Each candidate remembers which topic it
        # came from, for the grouping in step 2.
        candidates: list[CandidateMemory] = []

        existing_topics = sorted({s.topic for s in self._frozen_snapshot if s.level == "topic" and s.topic})
        for problem, result in failed:
            try:
                related = self._selections.get(problem.get("problem_idx"), [])
                context = build_reflect_context(
                    existing_topics=existing_topics,
                    problem_shape=problem.get("problem_shape", ""),
                    final_answer_given=str(result.get("final_answer_given", "")),
                    outcome="failed",
                    verifier_feedback=result.get("verifier_feedback", ""),
                    tool_errors=result.get("tool_errors", ""),
                    reasoning_excerpt=result.get("reasoning_excerpt", ""),
                    round_count=result.get("round_count", 0),
                    related_skills=related,
                )
                candidate = reflect(self.agent, context, feedback_level=self.feedback_level)
            except Exception:
                logger.warning("reflect failed for problem %s; skipping candidate",
                                problem.get("problem_idx"), exc_info=True)
                continue
            if candidate is None:
                continue
            # Provenance tag, never the question/answer itself -- see the anti-leakage
            # constraint in the module docstring and the guard's own evidence convention.
            candidate.evidence = list(candidate.evidence) + [f"p_{problem.get('problem_idx')}"]
            candidates.append(candidate)

        if not candidates:
            return []

        # 2. Group by the topic the MODEL named (paper Appendix E.1), not by any dataset label --
        # a candidate that named no topic is cross-task only and has no bucket to be curated in.
        edits: list[SkillEdit] = []
        by_topic: dict[str, list[CandidateMemory]] = {}
        for cand in candidates:
            if cand.topic:
                by_topic.setdefault(cand.topic, []).append(cand)

        # `self.store.all()` (live), not `self._frozen_snapshot`, is deliberate and safe here:
        # nothing between `begin_batch()` and the `store.apply()` call below ever mutates a
        # skill's trigger/lesson/failure_mode (the only fields `_render_existing` reads) --
        # `record_selection()` can run concurrently with an open batch, but it only touches
        # `n_selected`/`n_selected_success` via `record_usage()`, which `_render_existing` never
        # looks at. So the two views are provably identical in content at this point, and using
        # the live one avoids an extra deep copy. This equivalence is *not* enforced by a type or
        # test, though -- it would silently break if a future change let something else mutate
        # skill content before `apply()` runs in the same `end_batch()`.
        for topic, topic_candidates in by_topic.items():
            existing = [s for s in self.store.all() if s.level == "topic" and s.topic == topic]
            edits.extend(TopicCurator().curate(self.agent, existing=existing,
                                                candidates=topic_candidates, caps=self.store.caps,
                                                topic=topic))

        # 3. All of this batch's candidates -> GeneralCurator (cross-topic layer).
        existing_general = [s for s in self.store.all() if s.level == "general"]
        edits.extend(GeneralCurator().curate(self.agent, existing=existing_general,
                                              candidates=candidates, caps=self.store.caps))

        if not edits:
            return []

        # `ground_truth` is read here for exactly one purpose: passed straight through to
        # `store.apply()`, whose guard uses it only to *reject* a leaking candidate. It is never
        # read for any other purpose and never touches a prompt (see module docstring / global
        # constraints).
        question_texts = [p.get("question", "") for p, _ in failed]
        ground_truths = [p.get("ground_truth", "") for p, _ in failed]
        try:
            # `problem_idx=batch_idx`, not any one failed problem's own index: `store.apply()`
            # takes a single `problem_idx` per call, but this call's edits are typically drawn
            # from *multiple* failed problems in the batch (see step 1) -- there is no single
            # problem this batch-level compilation step belongs to. `batch` is already logged
            # separately for that purpose; `problem_idx`/`created_at` here is read-side
            # provenance metadata only (never a decision input elsewhere), so tagging it with the
            # batch's own index is the least misleading choice among several arbitrary ones.
            return self.store.apply(edits, problem_idx=batch_idx, batch=batch_idx,
                                     question_texts=question_texts, ground_truths=ground_truths)
        except Exception:
            logger.exception("store.apply failed for batch %s; harness left unchanged", batch_idx)
            return []


_ARM_REGISTRY = {
    "baseline": BaselineArm,
    "raw": RawExperienceArm,
    "evo": EvoHarnessArm,
}


def build_arm(name: str, **kwargs) -> CrossProblemArm:
    try:
        cls = _ARM_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown arm name: {name!r} (expected one of {sorted(_ARM_REGISTRY)})") from None
    return cls(**kwargs)
