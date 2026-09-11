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
"""Anti-leakage / format guard for candidate skills.

This is the only module in ``alphaapollo/core/harness/`` allowed to touch ``ground_truth``
values: it exists purely to *reject* candidates, never to generate or persist them, so a
ground-truth string is compared against and then discarded -- it never flows into a return
value, a log line, or an exception message. Every other harness module is checked by CI for
the absence of this identifier; this file is the deliberate, documented exception.

Note: this module must never import the in-problem memory package (the analogue this
cross-problem mechanism is kept separate from) -- see tests/harness/test_smoke.py, which
enforces that boundary via an AST scan of this package.

``validate_skill()`` returns ``(accepted, note)``:
  - ``accepted is False``: ``note`` is one of the fixed reject-reason strings
    (``"empty_section"``, ``"non_english"``, ``"lesson_too_long"``, ``"skill_too_long"``,
    ``"question_overlap"``, ``"answer_leak"``) -- these are a stable external contract,
    written verbatim into ``harness_log.jsonl``'s ``reject_reason`` field by later stages.
  - ``accepted is True``: ``note`` is either ``None``, or the advisory tag
    ``"numeric_coincidence"`` -- the candidate is kept, but a number in it happened to equal
    a ground truth without being asserted as an answer (see below). Later stages persist
    this into ``harness_log.jsonl``'s ``guard_note`` field for observability; it is not a
    rejection.
"""

from __future__ import annotations

import re

from alphaapollo.core.harness.schema import CandidateMemory

_WORD = re.compile(r"[A-Za-z0-9']+")
_NUMBER = re.compile(r"(?<![\w.])\d+(?![\w.])")
_NGRAM = 8
_NON_ASCII_RATIO = 0.05

# Phrases that assert a value IS the answer, as opposed to using it as a bound,
# a modulus, or a step count. Kept deliberately tight: every added cue trades a
# caught leak for false rejections of ordinary method text.
_ASSERTION_CUE = re.compile(
    r"\b(answers?|solutions?|equals?|results?|is\s+exactly|turns?\s+out\s+to\s+be)\b",
    re.IGNORECASE,
)


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def validate_skill(
    candidate: CandidateMemory,
    *,
    question_texts: list[str],
    ground_truths: list[str],
    max_words: int = 200,
    max_lesson_words: int = 60,
) -> tuple[bool, str | None]:
    sections = (candidate.trigger, candidate.lesson, candidate.failure_mode)
    if any(not s.strip() for s in sections):
        return False, "empty_section"

    blob = "\n".join(sections)

    non_ascii = sum(1 for ch in blob if ord(ch) > 127)
    if blob and non_ascii / len(blob) > _NON_ASCII_RATIO:
        return False, "non_english"

    if len(_words(candidate.lesson)) > max_lesson_words:
        return False, "lesson_too_long"
    if len(_words(blob)) > max_words:
        return False, "skill_too_long"

    skill_ngrams = _ngrams(_words(blob), _NGRAM)
    for q in question_texts:
        if skill_ngrams & _ngrams(_words(q), _NGRAM):
            return False, "question_overlap"

    gt_numbers = set()
    for gt in ground_truths:
        gt_numbers.update(_NUMBER.findall(gt))

    # A number equal to a ground truth is only a leak when the text asserts it
    # AS an answer. AIME answers are 0-999, and 4% of the adaptation pool are
    # exactly the round numbers a skill uses as a bound, so a bare equality test
    # rejects 21-28% of "enumerate a small range first" skills -- the very skills
    # this project exists to learn. Coincidence is kept and flagged instead.
    coincidence = False
    for line in blob.splitlines():
        hits = gt_numbers & set(_NUMBER.findall(line))
        if not hits:
            continue
        if _ASSERTION_CUE.search(line):
            return False, "answer_leak"
        coincidence = True

    return True, "numeric_coincidence" if coincidence else None
