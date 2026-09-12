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
"""Per-role call accounting and seed injection for the cross-problem skill harness.

The assignment requires the three experiment arms to report "solver calls" and
"cross-problem skill-management calls" *separately* -- not just a final accuracy number. This
module is the only source of that split. It has nothing to do with in-problem memory (that
boundary is structural: see ``tests/harness/test_smoke.py``, which AST-scans every file directly
under this package for an import of ``alphaapollo.core.environments.memory``) -- it simply counts
how many LLM calls of each *role* were made, and how many tokens each cost.

Why a monkey-patch of ``Agent.get_action_from_gpt`` rather than a wrapper object: the summarizer
(``evolving_main.py:607``) and the verifier-report aggregator (``evolving_main.py:180``) both
construct their own ``Agent`` instance from a config dict *inside* the function that uses it --
the caller never sees that instance, so nothing external can hold a reference to wrap. Only a
class-level patch reaches every call site (solver, summarizer, aggregator, and whatever the
Task B reflect/curate stages construct) without touching a single upstream file.

``install_accounting()`` also has to reconstruct, not just observe, the request: the upstream
``get_action_from_gpt`` discards ``response.usage`` entirely (it only ever returns
``reasoning_text + action``), so there is no token count to intercept -- the patched method has to
issue the request itself and re-derive the return value using the exact same logic as the
original (including the ``<think>...</think>`` wrapping of a ``reasoning_content``/``reasoning``
field), so that swapping the patch in or out never changes what downstream parsing sees.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Iterator

from alphaapollo.core.generation.evolving.utils.agent import Agent

logger = logging.getLogger(__name__)

# Solver-side calls: the frozen policy/verifier loop that already exists in
# `informal_math_evolving` today, including the two helper agents the baseline itself
# constructs on the fly (summarizer, aggregator) -- neither is part of the cross-problem
# skill-management machinery this project adds, so both must be counted as solver-side cost,
# not management overhead.
SOLVER_ROLES = ("solver", "summarizer", "aggregator")

# Management-side calls: every LLM call the Task B compilation loop makes to turn a completed
# trajectory into harness updates (reflect / curate), plus any offline labeling pass, plus the
# RawExperience arm's per-problem trajectory summariser. That summariser call is deliberately
# its own role, "raw_summarizer", rather than reusing "summarizer": "summarizer" already names
# the upstream, in-problem helper agent the frozen baseline constructs on the fly
# (evolving_main.py:607, see SOLVER_ROLES above) to compress a single problem's own history --
# it is solver-side cost that exists independent of this project. RawExperienceArm's summariser
# is a *different* call, made once per completed problem specifically to produce the raw,
# cross-problem "experience" note that arm injects into later problems -- i.e. it is that arm's
# entire cross-problem management overhead, the direct analogue of what reflect()/the curators
# are for EvoHarnessArm. Folding it into "summarizer" would silently count it as solver-side and
# make RawExperience look like it has near-zero management overhead in Task C's cost report,
# when in fact it makes one such call per problem (denser than EvoHarness's per-batch
# reflect/curate calls).
MGMT_ROLES = ("reflect", "topic_curator", "general_curator", "raw_summarizer", "offline_labeling")

# A ContextVar (not a plain module-level global) because problems within a batch may run
# concurrently across threads; each thread must see only the role its own call stack set, never
# a role set by a sibling thread that happens to interleave.
_current_role: contextvars.ContextVar[str] = contextvars.ContextVar("harness_role", default="unscoped")


@contextmanager
def role_scope(role: str) -> Iterator[None]:
    """Tag every ``Agent.get_action_from_gpt`` call made while inside this block as ``role``."""
    token = _current_role.set(role)
    try:
        yield
    finally:
        _current_role.reset(token)


class CallAccountant:
    """Per-role call/token counters.

    All three counters are plain ``dict[str, int]`` (backed by ``defaultdict(int)`` for
    convenient increment, but never exposed in a way that lets a mere read auto-vivify a
    stray zero-count key -- ``snapshot()`` and any internal lookup use ``.get()``). A lock
    guards every mutation because a batch of problems may be solved concurrently across
    threads, all incrementing the same accountant.
    """

    def __init__(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.tokens_in: dict[str, int] = defaultdict(int)
        self.tokens_out: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(self, role: str, tokens_in: int, tokens_out: int) -> None:
        with self._lock:
            self.calls[role] += 1
            self.tokens_in[role] += tokens_in
            self.tokens_out[role] += tokens_out

    def snapshot(self) -> dict:
        """Flatten the per-role counters into a single dict for logging/reporting.

        Keys: ``calls/<role>``, ``tokens/<role>_in``, ``tokens/<role>_out`` for every role that
        has recorded at least one call, plus two fixed aggregates -- ``calls/solver_side_total``
        and ``calls/mgmt_side_total`` -- which are the direct source of the "solver calls vs.
        skill-management calls, reported separately" requirement.
        """
        with self._lock:
            calls = dict(self.calls)
            tokens_in = dict(self.tokens_in)
            tokens_out = dict(self.tokens_out)

        result: dict = {}
        for role, n in calls.items():
            result[f"calls/{role}"] = n
        for role, n in tokens_in.items():
            result[f"tokens/{role}_in"] = n
        for role, n in tokens_out.items():
            result[f"tokens/{role}_out"] = n

        result["calls/solver_side_total"] = sum(calls.get(role, 0) for role in SOLVER_ROLES)
        result["calls/mgmt_side_total"] = sum(calls.get(role, 0) for role in MGMT_ROLES)
        return result


def _looks_like_a_seed_rejection(exc: Exception) -> bool:
    """Heuristic for "this request failed because the provider doesn't understand `seed`".

    Different OpenAI-compatible providers reject an unknown parameter with different
    exception types (a real `openai` client raises `BadRequestError` on an HTTP 400; the test
    double raises a plain `TypeError`) -- there is no single exception class to catch. What is
    stable across providers is that the error message names the offending parameter, so this
    only treats the failure as seed-related if "seed" appears in the message; anything else is
    a real error and must propagate unchanged.
    """
    return "seed" in str(exc).lower()


def install_accounting(accountant: CallAccountant, *, seed: int | None = None) -> Callable[[], None]:
    """Monkey-patch ``Agent.get_action_from_gpt`` to attribute every call to the active role
    (see ``role_scope``) and, if ``seed`` is given, inject it into every request.

    Returns a zero-argument callable that restores the original method.

    Raises ``RuntimeError`` if accounting is already installed (i.e. a previous
    ``install_accounting()`` call's ``uninstall`` has not been invoked yet). Two independent
    patches stacked on the class cannot be unwound safely in arbitrary order -- an
    out-of-order uninstall would silently leave the class monkey-patched with no error --
    so this is refused up front instead of failing invisibly later.
    """
    if getattr(Agent.get_action_from_gpt, "_is_harness_accounting_patch", False):
        raise RuntimeError(
            "accounting is already installed on Agent.get_action_from_gpt; call the "
            "uninstall() returned by the previous install_accounting() before installing again"
        )

    original = Agent.get_action_from_gpt

    # A mutable holder (not a plain bool) so the closure below can flip it from inside the
    # patched method and have every subsequent call -- across every `Agent` instance, since the
    # patch is class-level -- see the update. Once a provider has rejected `seed` once, it is
    # never retried: the assignment/test explicitly require exactly one probing retry, not one
    # retry per call.
    seed_state = {"enabled": seed is not None, "value": seed}

    @functools.wraps(original)
    def patched_get_action_from_gpt(self: Agent, obs):
        role = _current_role.get()

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": obs})

        base_kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
        )

        def _create(with_seed: bool):
            call_kwargs = dict(base_kwargs)
            if with_seed:
                call_kwargs["seed"] = seed_state["value"]
            return self.client.chat.completions.create(**call_kwargs)

        if seed_state["enabled"]:
            try:
                response = _create(True)
            except Exception as exc:
                if not _looks_like_a_seed_rejection(exc):
                    raise
                logger.warning("provider rejected `seed`, disabling seed injection for the rest of this run: %s", exc)
                seed_state["enabled"] = False
                response = _create(False)
        else:
            response = _create(False)

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        accountant.record(role, prompt_tokens, completion_tokens)

        message = response.choices[0].message
        reasoning_payload = None
        if hasattr(message, "reasoning_content"):
            reasoning_payload = message.reasoning_content
        elif hasattr(message, "reasoning"):
            reasoning_payload = message.reasoning

        reasoning_text = ""
        if isinstance(reasoning_payload, str) and reasoning_payload.strip():
            reasoning_text = "<think>\n" + reasoning_payload.strip() + "\n</think>\n"

        action = message.content.strip()
        return reasoning_text + action

    patched_get_action_from_gpt._is_harness_accounting_patch = True
    Agent.get_action_from_gpt = patched_get_action_from_gpt

    def uninstall() -> None:
        # Idempotent: calling this more than once just reassigns the same original object.
        Agent.get_action_from_gpt = original

    return uninstall
