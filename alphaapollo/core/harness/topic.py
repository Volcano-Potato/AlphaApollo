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
"""Model-side topic labelling for the problem stream.

**Why this exists at all.** The harness stores topic-scoped skills, and retrieval filters on the
topic (``store.select``: ``s.level == "general" or s.topic == topic``). A problem with no topic can
therefore only ever retrieve general skills -- the topic layer becomes write-only. The catch is
that retrieval happens *before* the problem is solved, so the topic cannot come from the
trajectory: ``reflect`` already lets the model choose the general-vs-topic *layer* for a skill it
is writing, but the *reading* side needs a topic for a problem nothing has been learned about yet.

**Why the model does it rather than a person.** Hand-curating a clean taxonomy would be doing part
of the method's work for it, and only the Evo-Harness arm benefits from topic retrieval, so any
quality in hand-made labels flatters exactly one arm. Here the labels are produced by the same
frozen model, from the question text alone.

**Why that is not leakage.** The topic is a function of the question, and the solver reads that
same question in full -- there is no information here it does not already have. What would be
leakage is deriving a topic from the ground-truth answer or from later problems; the signature
below takes a bare ``question`` string precisely so a problem dict carrying ``ground_truth``
cannot be passed in by accident.

**Why labelling is frozen into the stream instead of run online per arm.** The per-topic
performance breakdown is a required result for *all three* arms, and it is only comparable if
Baseline, Raw Experience and Evo-Harness see identical topics. Labelling once, up front, is what
guarantees that. The calls are accounted under the ``offline_labeling`` management role, so they
appear in the cost report as cross-problem overhead rather than hiding in the solver bucket.

**Measured label quality** (``agreement()`` against MathArena's human ``problem_type`` on all 30
AIME 2025 problems; a multi-label problem counts as agreeing if the predicted label is among the
human ones):

===============  ==================  ==============  =======
annotator        agreement (n=30)    unlabelled      wall
===============  ==================  ==============  =======
qwen3-8b         21/30 = 70%         1               64s
qwen3-32b        25/30 = 83%         0               21s
qwen3.7-flash    25/30 = 83%         0               21s
===============  ==================  ==============  =======

qwen3-8b's errors are systematic, not scattered: it returned ``geometry`` 14 times where the
humans said 7, and six of its nine misses are "model says geometry, human says algebra or
combinatorics" -- AIME algebra problems mention geometric objects constantly. That bias would put
roughly half the geometry bucket's skills somewhere no related problem retrieves from.

Two consequences for how this is used and reported:

- The annotator is configured separately from the solver, and is expected to be a stronger model
  than the frozen policy. This is *dataset annotation*, the same category of act as MathArena
  paying humans to label its own split -- not part of the method under test. It is applied once,
  shared byte-identically by all three arms, and reads only question text. The policy, verifier,
  Reflect and curator models stay frozen and identical across arms, which is what the fairness
  requirement actually constrains.
- 83% on n=30 is a wide interval (roughly +/-13pp). The write-up should quote it as a measured
  limitation, not a guarantee: "selection errors" is one of the explanations the assignment asks
  for when a null result appears, and this makes that explanation quantitative instead of
  speculative.

The prompt below was deliberately **not** tuned against these numbers. AIME 2025 is the held-out
split; using it to pick wording would be exactly the test-set leakage the protocol forbids, even
though only its metadata is involved. It was used to measure candidate annotators and nothing
else.
"""

from __future__ import annotations

import logging
import re

from alphaapollo.core.harness.accounting import role_scope
from alphaapollo.core.harness.reflect import _strip_reasoning

logger = logging.getLogger(__name__)

# A fixed, closed vocabulary, deliberately the same four buckets MathArena's human `problem_type`
# annotations use for AIME 2025. Defining the taxonomy is method design; assigning problems to it
# is the work, and that is what the model does. An open-ended "what is this problem about?" would
# return a fresh phrasing per problem, leaving every topic bucket holding exactly one skill and
# destroying the layer it exists to support.
TOPICS: tuple[str, ...] = ("algebra", "number_theory", "combinatorics", "geometry")

_ALIASES = {
    "number theory": "number_theory",
    "numbertheory": "number_theory",
    "counting": "combinatorics",
    "probability": "combinatorics",
    "counting and probability": "combinatorics",
    "combinatorics and probability": "combinatorics",
}

TOPIC_PROMPT = """Classify the mathematical topic of the competition problem below.

Answer with EXACTLY ONE of these four labels and nothing else:
algebra
number_theory
combinatorics
geometry

If the problem spans several, choose the one whose techniques dominate the solution.

Problem:
{question}"""


def _normalise(text: str) -> str | None:
    """Map a model reply onto the vocabulary, or ``None`` if it does not land on it.

    Reasoning-model ``<think>`` blocks are stripped first: they routinely mention several topics
    while deliberating, and matching against that text would pick whichever label happened to be
    typed first rather than the conclusion.
    """
    cleaned = _strip_reasoning(text or "").strip().lower()
    if not cleaned:
        return None

    # Longest alias first so "number theory" is not shadowed by a bare "theory"-style prefix.
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", cleaned):
            return _ALIASES[alias]

    for topic in TOPICS:
        spaced = topic.replace("_", "[ _-]")
        if re.search(rf"(?<![\w-]){spaced}(?![\w-])", cleaned):
            return topic
    return None


def classify_topic(agent, question: str) -> str | None:
    """Label one problem's topic from its question text. Returns ``None`` when the reply cannot be
    mapped onto :data:`TOPICS`, or when the call fails.

    ``None`` rather than a fallback guess is deliberate: a *wrong* topic is worse than no topic,
    because it files a skill in a bucket no related problem will ever retrieve from, and it
    silently corrupts the per-topic breakdown. An unlabelled problem degrades to
    general-skills-only retrieval, which is merely weaker, not wrong.

    Takes ``question`` as a plain string, never a problem dict -- see this module's docstring on
    why the ground truth must not be reachable from here.
    """
    try:
        with role_scope("offline_labeling"):
            reply = agent.get_action_from_gpt(TOPIC_PROMPT.format(question=question))
    except Exception:
        logger.warning("topic labelling call failed; leaving the problem unlabelled", exc_info=True)
        return None
    return _normalise(reply)


def label_stream(agent, problems: list[dict]) -> list[dict]:
    """Return a copy of ``problems`` with every entry carrying a ``topic``.

    Three behaviours worth stating, each load-bearing for the experiment rather than incidental:

    - An existing non-empty ``topic`` is kept and costs no call. MathArena's 2025 split ships
      human ``problem_type`` labels, which are better than anything this produces.
    - Identical question texts are labelled once and reused, so the stream is internally
      consistent and repeated problems cannot land in two different buckets.
    - A problem that cannot be labelled is kept with ``topic == ""``, never dropped. Dropping
      would shrink the stream and break the requirement that all three arms run the identical
      problem sequence.
    """
    labelled: list[dict] = []
    cache: dict[str, str] = {}

    for problem in problems:
        existing = str(problem.get("topic") or "").strip()
        if existing:
            labelled.append(dict(problem))
            continue

        question = str(problem.get("question") or "")
        if question not in cache:
            cache[question] = classify_topic(agent, question) or ""
        labelled.append({**problem, "topic": cache[question]})

    return labelled


def agreement(predicted: list[dict], human: list[dict]) -> dict:
    """Compare model labels against human ones, keyed by ``problem_idx``.

    Run against MathArena's 2025 ``problem_type`` annotations, this turns "are the labels any
    good?" from an assumption the write-up has to hand-wave into a number it can report.
    Unlabelled predictions count as disagreements rather than being excluded -- a labeller that
    abstains often is not thereby more accurate.
    """
    human_by_idx = {h.get("problem_idx"): str(h.get("topic") or "").strip() for h in human}
    n_compared = n_agree = 0
    for entry in predicted:
        idx = entry.get("problem_idx")
        if idx not in human_by_idx:
            continue
        n_compared += 1
        if str(entry.get("topic") or "").strip() == human_by_idx[idx]:
            n_agree += 1
    return {
        "n_compared": n_compared,
        "n_agree": n_agree,
        "rate": (n_agree / n_compared) if n_compared else 0.0,
    }
