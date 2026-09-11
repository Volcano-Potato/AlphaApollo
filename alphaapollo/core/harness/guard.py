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
"""

from __future__ import annotations

import re

from alphaapollo.core.harness.schema import CandidateMemory

_WORD = re.compile(r"[A-Za-z0-9']+")
_NUMBER = re.compile(r"(?<![\w.])\d+(?![\w.])")
_NGRAM = 8
_NON_ASCII_RATIO = 0.05


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
    if gt_numbers & set(_NUMBER.findall(blob)):
        return False, "answer_leak"

    return True, None
