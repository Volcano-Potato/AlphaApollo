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
  every candidate that was proposed but *rejected*, including one the curator neither accepted
  nor explicitly skipped but simply never mentioned (``evolver._bind_payloads`` turns those into
  synthetic SKIPs precisely so this promise holds). Each line also carries ``source_problems``,
  the problems whose failures produced that candidate -- ``problem_idx`` on these lines is the
  *batch* index and cannot name them. This is what lets the exported harness be audited later
  ("what changed, why, and on whose evidence").
- ``selection_log.jsonl`` -- the per-problem selection record: which skills were injected into
  which problem's context, at what token cost, alongside the resulting pass/fail signal. This is
  a different axis (retrieval, not editing) and must not be interleaved with the edit history --
  see ``test_selection_log_is_a_separate_file``, which asserts that logging a selection never
  creates ``harness_log.jsonl`` at all.

This module implements persistence, logging, and edit application (``apply()``, turning a batch
of curator-issued ``SkillEdit``s into ADD/MERGE/REVISE/DELETE mutations under a hard capacity
bound -- see ``apply()`` for the two-phase ordering this requires), plus per-problem skill
*selection* (``select()``/``record_usage()``, budget-aware retrieval under ``Budget``).

``select()`` here is the deterministic, model-free retriever this module originally shipped. It
is **no longer the path the Evo-Harness arm uses**: selection now happens in ``selector.py`` as a
model call, matching paper Appendix F ("For harness selection, we use Claude Sonnet 4.5 across all
experiments to retrieve relevant skills from the current harness before task execution"). The
deterministic version is kept because it is a useful, dependency-free reference implementation and
several store-level tests exercise ranking without a model -- but a reader tracing what the
experiment actually does should follow ``selector.select_skills``, not this method.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from alphaapollo.core.harness.guard import validate_skill
from alphaapollo.core.harness.render import count_tokens
from alphaapollo.core.harness.schema import CandidateMemory, Skill, SkillEdit, skill_from_markdown, skill_to_markdown

# ``Skill.name`` is not a trusted, developer-controlled constant: later tasks have a curator
# (an LLM) mint it, e.g. as a kebab-case slug of the trigger. Two independent problems follow
# from ever using it as a filename verbatim:
#   1. Collision: two skills with different ids can end up with the same (or same-looking)
#      name, silently overwriting one another's file on disk while the in-memory dict (keyed
#      by id) still holds both -- an in-memory/on-disk split that only surfaces on the next
#      reload, with no error anywhere.
#   2. Path traversal: a malformed name like "../../pwned" writes outside ``skills_dir``
#      entirely, and can never be found again by ``reload()`` (which only globs inside
#      ``skills_dir``), i.e. a silent, unrecoverable write to the wrong place.
# The fix: uniqueness is the id's job, not the name's. The filename is a sanitized,
# human-readable name prefix plus the skill's own (already-unique) id, so a collision in
# ``name`` can never collide on disk, and a malformed ``name`` can never escape the directory.
_NAME_DISALLOWED = re.compile(r"[^a-z0-9-]+")
_DASH_RUN = re.compile(r"-+")
_MAX_NAME_PREFIX_LEN = 60

# Used to mint ``Skill.name`` itself (see ``_slugify_trigger``), which is a distinct concern
# from ``_safe_filename`` above: ``name`` is not just a filename-prefix source, it is also what
# ``render.py`` puts directly into the policy's system message (``### {skill.name}``). Naming a
# skill after its own bookkeeping fields (e.g. ``f"{level}-{id}"`` -- an earlier draft of this
# module did exactly that) would render literal internal ids like "topic-sk_0007" into the
# model's context as pure noise, and would produce unreadable on-disk filenames when a human
# later wants to cite a specific skill in the README's case-study section. A short, readable
# kebab-case slug of the trigger text solves both: it is what the skill is *about*, not how the
# store happens to have filed it.
_SLUG_WORD = re.compile(r"[a-z0-9]+")
_MAX_SLUG_WORDS = 8

# Used by `SkillStore.select()`'s lexical-overlap term: a deliberately dumb, dependency-free
# word-overlap heuristic (not a real IDF/BM25 implementation) so that selection needs neither a
# model call nor a tokenizer download -- see the module docstring addendum on `select()` below
# for why zero-model-call selection is a hard requirement here, not just an optimization.
# Provenance tags written by `EvoHarnessArm.observe` onto every candidate's `evidence` list, in
# the form `p_<problem_idx>`. Anchored at both ends so a free-form note the guard may also have
# appended to `evidence` cannot be mistaken for a problem reference.
_EVIDENCE_TAG = re.compile(r"^p_(\d+)$")


def _source_problems(edit: SkillEdit) -> list[int]:
    """Which problems' failures produced the candidate behind ``edit``.

    A ``harness_log.jsonl`` line's ``problem_idx`` is the *batch* index, not a problem's: a single
    ``apply()`` call carries edits pooled from every failed problem in the batch, so one value
    cannot name them. Without this field the log cannot answer "what did problem N propose, and
    was it accepted?" -- something the assignment requires, and something that matters most for
    *rejected* candidates, which exist nowhere else once the call returns (an accepted one keeps
    its evidence in the skill file).

    Returns an empty list for edits with no candidate behind them (``DELETE`` is harness
    maintenance, not a response to a proposal), rather than omitting the key -- every log line
    keeps the same shape for downstream analysis.
    """
    payload = getattr(edit, "payload", None)
    evidence = getattr(payload, "evidence", None) or []
    found: list[int] = []
    for tag in evidence:
        match = _EVIDENCE_TAG.match(str(tag))
        if match:
            idx = int(match.group(1))
            if idx not in found:
                found.append(idx)
    return sorted(found)


_SELECT_WORD = re.compile(r"[A-Za-z']+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that the "
    "this to was were when where which with you your".split()
)


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _SELECT_WORD.findall(text) if w.lower() not in _STOPWORDS}


def _slugify_trigger(trigger: str) -> str:
    """First few words of ``trigger``, lowercased and hyphenated, e.g. "Counting integers
    subject to divisibility conditions." -> "counting-integers-subject-to-divisibility-
    conditions". Falls back to the literal string "skill" if the trigger contains no
    alphanumeric words at all (e.g. it was pure punctuation) -- this can only ever affect
    cosmetics (the rendered header, the filename prefix), never correctness: uniqueness on
    disk is always guaranteed by the id via ``_safe_filename``, never by this slug."""
    words = _SLUG_WORD.findall(trigger.lower())[:_MAX_SLUG_WORDS]
    slug = "-".join(words)[:_MAX_NAME_PREFIX_LEN].strip("-")
    return slug or "skill"


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
        self._max_issued_seq = 0
        self.reload()

    # ---- persistence -------------------------------------------------
    def reload(self) -> None:
        """Rebuild in-memory state from disk. Used both at construction time and by tests /
        callers that want to confirm a write actually survived a fresh read, independent of
        whatever this process's in-memory dict currently holds.

        Also re-derives the id high-water-mark used by ``_next_id()`` from whatever is on
        disk, taking the max against whatever this instance had already issued rather than
        overwriting it -- see ``_next_id`` for why a plain "max of currently-held ids" is not
        enough once ``apply()`` can delete a skill and then mint a replacement in the very
        same batch."""
        self._skills = {}
        for path in sorted(self.skills_dir.glob("*.md")):
            skill = skill_from_markdown(path.read_text(encoding="utf-8"))
            self._skills[skill.id] = skill
        on_disk_max = max((int(sid.split("_")[1]) for sid in self._skills if sid.startswith("sk_")), default=0)
        self._max_issued_seq = max(getattr(self, "_max_issued_seq", 0), on_disk_max)

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

    @staticmethod
    def _safe_filename(skill: Skill) -> str:
        """Filename for ``skill``: a sanitized, truncated ``name`` prefix (for human
        readability, e.g. when eyeballing ``skills/`` or citing a file in the README) plus the
        skill's own globally-unique ``id`` as the actual uniqueness/collision guarantee, so two
        skills can never contend for the same path regardless of what ``name`` either of them
        was given. Disallowed characters (anything outside ``[a-z0-9-]``, including path
        separators, dots, whitespace, and non-ASCII) are replaced with ``-``; runs of ``-`` are
        collapsed and leading/trailing ``-`` stripped, so ``"../../pwned"`` sanitizes to
        ``"pwned"`` rather than escaping ``skills_dir``. If the sanitized prefix is empty (e.g.
        the name was entirely disallowed characters), the id alone is used as the filename."""
        prefix = _DASH_RUN.sub("-", _NAME_DISALLOWED.sub("-", skill.name.lower())).strip("-")
        prefix = prefix[:_MAX_NAME_PREFIX_LEN].strip("-")
        return f"{prefix}--{skill.id}.md" if prefix else f"{skill.id}.md"

    def _write_skill(self, skill: Skill) -> None:
        self._validate_level_topic(skill)
        skill.n_tokens = count_tokens(f"{skill.trigger}\n{skill.lesson}\n{skill.failure_mode}")
        (self.skills_dir / self._safe_filename(skill)).write_text(skill_to_markdown(skill), encoding="utf-8")
        self._skills[skill.id] = skill

    def _delete_skill(self, skill_id: str) -> None:
        skill = self._skills.pop(skill_id, None)
        if skill is not None:
            (self.skills_dir / self._safe_filename(skill)).unlink(missing_ok=True)

    def _next_id(self) -> str:
        """Monotonic id issuance, seeded from whatever is on disk at ``reload()``/construction
        time and only ever incremented afterwards -- never recomputed from ``self._skills``
        alone. That would look sufficient (and was the original design) but breaks the moment
        a batch deletes a skill and then mints a replacement in the same ``apply()`` call:
        once the victim is gone from ``self._skills``, "max of currently-held ids" drops back
        down and reissues the id that was *just freed*, silently aliasing two unrelated skills
        (different trigger/lesson/evidence, same id) across the harness log's history. Tracking
        a high-water-mark instead of recomputing it keeps both guarantees: a fresh ``SkillStore``
        over the same root still never collides with a skill file already on disk (``reload()``
        maxes the mark against on-disk state), and a same-session delete-then-add never reuses
        the deleted id either."""
        self._max_issued_seq += 1
        return f"sk_{self._max_issued_seq:04d}"

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

    # ---- editing -------------------------------------------------------
    def _occupancy(self, level: str, topic: str | None) -> tuple[int, int]:
        """Current used-slot count and cap for ``level`` (and, for a topic-level skill,
        ``topic``). Always computed live from ``self._skills`` -- never cached -- so that a
        DELETE applied earlier in the same ``apply()`` call is immediately visible to an ADD
        considered later in that same call (see ``apply()``'s two-phase ordering below)."""
        if level == "general":
            used = sum(1 for s in self._skills.values() if s.level == "general")
            return used, self.caps.general
        used = sum(1 for s in self._skills.values() if s.level != "general" and s.topic == topic)
        return used, self.caps.per_topic

    def _materialize(self, payload: CandidateMemory, problem_idx: int) -> Skill:
        level = payload.scope_hint
        sid = self._next_id()
        return Skill(
            id=sid, name=_slugify_trigger(payload.trigger), level=level,
            topic=None if level == "general" else payload.topic,
            trigger=payload.trigger, lesson=payload.lesson,
            failure_mode=payload.failure_mode, evidence=list(payload.evidence),
            created_at=problem_idx,
        )

    def apply(
        self,
        edits: list[SkillEdit],
        *,
        problem_idx: int,
        batch: int,
        question_texts: list[str],
        ground_truths: list[str],
    ) -> list[dict]:
        """Apply a batch of curator-issued edits and return one result dict per edit (same
        order as ``edits``), while writing exactly one ``harness_log.jsonl`` line per edit
        (accepted or rejected) as a side effect.

        Two-phase, not one pass in list order: everything that frees or rewrites an existing
        slot (DELETE / MERGE / REVISE / SKIP) is applied first; ADDs are only checked against
        occupancy *after* that -- regardless of where they appear in the input list. This is
        not an optimization, it is required for a curator that emits "DELETE the weak skill,
        ADD a better one" in the same cycle: without phase separation, a full harness could
        only ever grow, never turn over, because the ADD would be checked against
        pre-deletion occupancy.
        """
        phase1 = [e for e in edits if e.op in ("DELETE", "MERGE", "REVISE", "SKIP")]
        phase2 = [e for e in edits if e.op == "ADD"]

        results: list[dict] = []
        for edit in phase1 + phase2:
            try:
                record = self._apply_one(edit, problem_idx, question_texts, ground_truths)
            except Exception as exc:
                # A single malformed/unexpected edit must never abort the whole batch --
                # doing so would risk breaking the underlying baseline run this harness is
                # layered on top of. Log it as a rejection and keep processing the rest.
                record = {"skill_id": edit.skill_id, "accepted": False,
                          "reject_reason": f"internal_error:{type(exc).__name__}", "guard_note": None}
            record.update(problem_idx=problem_idx, batch=batch, op=edit.op,
                          actor=edit.actor, reason=edit.reason,
                          source_problems=_source_problems(edit))
            self.log_event(**record)
            results.append(record)
        return results

    def _apply_one(self, edit: SkillEdit, problem_idx: int, question_texts: list[str], ground_truths: list[str]) -> dict:
        if edit.op == "SKIP":
            return {"skill_id": edit.skill_id, "accepted": False, "reject_reason": "skipped",
                    "guard_note": None}

        if edit.op == "DELETE":
            existed = edit.skill_id in self._skills
            self._delete_skill(edit.skill_id)
            return {"skill_id": edit.skill_id, "accepted": existed,
                    "reject_reason": None if existed else "unknown_skill_id",
                    "guard_note": None}

        if edit.payload is None:
            return {"skill_id": edit.skill_id, "accepted": False,
                    "reject_reason": "missing_payload", "guard_note": None}

        ok, note = validate_skill(edit.payload, question_texts=question_texts,
                                  ground_truths=ground_truths)
        if not ok:
            return {"skill_id": edit.skill_id, "accepted": False, "reject_reason": note,
                    "guard_note": None}
        # `note` may instead be an advisory tag on an *accepted* candidate (currently only
        # "numeric_coincidence"): keep it and log it, it is not a rejection -- its count is
        # what quantifies how many skills a bare GT-equality rule would have thrown away.
        guard_note = note

        if edit.op in ("REVISE", "MERGE"):
            target = self._skills.get(edit.skill_id)
            if target is None:
                return {"skill_id": edit.skill_id, "accepted": False,
                        "reject_reason": "unknown_skill_id", "guard_note": None}
            # A curator may only edit skills in its own layer -- and, for the topic layer, its
            # own topic bucket. Observed in a real run: GeneralCurator, which is shown only
            # general skills, hallucinated a topic skill's id and this method rewrote it, because
            # it looked the target up by id and never checked what layer the target was in. Both
            # curators then accumulated content and evidence into the same skill, so the two
            # layers stopped being independent -- which would silently invalidate the
            # General-Only / Topic-Only ablation the paper reports.
            #
            # `_bind_payloads` forces every payload's scope_hint/topic to the issuing curator's
            # own layer, so the payload is a reliable statement of who issued this edit.
            if target.level != edit.payload.scope_hint or (
                target.level == "topic" and target.topic != edit.payload.topic
            ):
                return {"skill_id": edit.skill_id, "accepted": False,
                        "reject_reason": "wrong_layer", "guard_note": guard_note}
            target.trigger = edit.payload.trigger
            target.lesson = edit.payload.lesson
            target.failure_mode = edit.payload.failure_mode
            target.evidence = sorted(set(target.evidence) | set(edit.payload.evidence))
            target.revised_at = sorted(set(target.revised_at) | {problem_idx})
            self._write_skill(target)
            return {"skill_id": target.id, "accepted": True, "reject_reason": None,
                    "guard_note": guard_note}

        # A general skill must rest on a pattern seen in at least two DIFFERENT problems.
        # GENERAL_CURATOR_PROMPT already says so; nothing enforced it, and the model ignored it --
        # the first real 12-problem batch produced three "general" skills whose evidence was
        # p_0, p_2 and p_3 respectively: three single-problem lessons filed as cross-task
        # patterns, two of them verbatim duplicates of topic skills minted from the same
        # candidate in the same batch. Left unchecked the general layer stops being a layer and
        # becomes a second copy of the topic layer, with both copies competing for the same
        # injection budget.
        #
        # Enforced here rather than by asking the model more firmly, for exactly the reason the
        # token budget is enforced in code: an instruction the model may ignore is not a
        # constraint. Checked on ADD only -- a MERGE reinforces a skill that already cleared this
        # bar, and re-checking the incoming payload alone would reject every legitimate
        # reinforcement.
        if edit.payload.scope_hint == "general" and len(set(_source_problems(edit))) < 2:
            return {"skill_id": None, "accepted": False,
                    "reject_reason": "general_needs_two_problems", "guard_note": guard_note}

        # ADD. `CandidateMemory.__post_init__` only validates that scope_hint is one of
        # {"general", "topic"}; it does NOT enforce the scope_hint=="topic" => topic-is-set
        # invariant that `Skill`/`_write_skill._validate_level_topic` assumes. Left unchecked,
        # a topic-scoped candidate with a missing topic would reach `_write_skill` and raise
        # ValueError there instead of failing this ADD cleanly -- caught above by the
        # try/except in `apply()`, but a dedicated reject_reason is far more useful for the
        # experiment log than a generic internal_error.
        level = edit.payload.scope_hint
        if level == "topic" and not edit.payload.topic:
            return {"skill_id": None, "accepted": False, "reject_reason": "missing_topic",
                    "guard_note": guard_note}
        used, cap = self._occupancy(level, edit.payload.topic)
        if used >= cap:
            return {"skill_id": None, "accepted": False, "reject_reason": "capacity_full",
                    "guard_note": guard_note}
        skill = self._materialize(edit.payload, problem_idx)
        self._write_skill(skill)
        return {"skill_id": skill.id, "accepted": True, "reject_reason": None,
                "guard_note": guard_note}

    # ---- selection -----------------------------------------------------
    @staticmethod
    def _lexical_overlap(trigger: str, question: str) -> float:
        """Fraction of `trigger`'s content words that also appear in `question`, in [0, 1].
        Deliberately not a real IDF/BM25 score (see the module docstring): it needs no corpus
        statistics, no tokenizer, and no network, so it can run inline in `select()` at zero
        marginal cost. An empty-after-stopwords trigger (all punctuation, or nothing but
        stopwords) scores 0.0 rather than raising a division-by-zero."""
        trigger_words = _content_words(trigger)
        if not trigger_words:
            return 0.0
        return len(trigger_words & _content_words(question)) / len(trigger_words)

    def _score(self, skill: Skill, question: str) -> float:
        return 0.3 * skill.utility() + 0.7 * self._lexical_overlap(skill.trigger, question)

    def select(self, question: str, topic: str | None) -> list[Skill]:
        """Deterministically pick up to `self.budget.b` skills relevant to `question` (and, if
        given, scoped `topic`), under fixed per-level quotas and a total token cap.

        Design is intentionally a single fixed-quota pass, not "reserve N general slots, let the
        rest compete globally": at `budget.b == 6` with `general_max == 3`, a global-competition
        scheme can starve topic skills down to a single slot whenever general skills happen to
        score well on a given question. Quotas here are hard caps, never reallocated across
        levels -- so a topic with no matching skills at all does *not* let general skills spill
        into its unused slots to reach `b`; the resulting selection can legitimately be smaller
        than `b`. See `test_adversarial_empty_topic_bucket_does_not_borrow_general_slack`.

        Candidates are ranked once, across both levels together, by
        `(-score, id)` -- `id` is a pure tiebreaker (never influences ranking otherwise) that
        makes the sort total and hence independent of dict/insertion order, which is what makes
        this fully reproducible for the Task C experiment arms. The single pass below then walks
        that ranked list applying, in order: the overall count cap `b`, the per-level quota
        (`general_max` / `topic_max`), and the cumulative token cap `tokens` (a skill that would
        push the running total over budget is skipped, not treated as a hard stop, so a smaller
        lower-ranked skill still gets a chance to fit in the remaining headroom -- see
        `test_adversarial_token_budget_boundary_equal_vs_one_over`). A single skill whose own
        `n_tokens` exceeds the entire budget is simply never admitted; it can never stall or loop
        the selection, since the surrounding `for` is over a fixed, finite list.
        """
        candidates = [s for s in self._skills.values() if s.level == "general" or s.topic == topic]
        ranked = sorted(candidates, key=lambda s: (-self._score(s, question), s.id))

        picked: list[Skill] = []
        used_tokens = 0
        n_general = n_topic = 0
        for skill in ranked:
            if len(picked) >= self.budget.b:
                break
            if skill.level == "general":
                if n_general >= self.budget.general_max:
                    continue
            elif n_topic >= self.budget.topic_max:
                continue
            if used_tokens + skill.n_tokens > self.budget.tokens:
                continue
            picked.append(skill)
            used_tokens += skill.n_tokens
            if skill.level == "general":
                n_general += 1
            else:
                n_topic += 1
        return picked

    def record_usage(self, selected: list[Skill], success: bool) -> None:
        """Update `n_selected`/`n_selected_success` (the counters `Skill.utility()` reads) for
        each skill in `selected` and persist the change. A skill id that no longer exists in the
        store (e.g. deleted by a curator edit that ran between `select()` and this call) is
        silently skipped rather than raising -- a stale usage callback must never be able to
        crash the run this harness is layered on top of; see
        `test_adversarial_record_usage_on_a_deleted_skill_does_not_raise`."""
        for picked in selected:
            skill = self._skills.get(picked.id)
            if skill is None:
                continue
            skill.n_selected += 1
            skill.n_selected_success += int(success)
            self._write_skill(skill)
