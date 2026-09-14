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
"""The Task B driver: closes the online-learning loop around the existing, unmodified
`informal_math_evolving` solver (``evolving_main.run_problem`` / ``create_runtime_for_problem``)
by wrapping it in a **batch-serial, within-batch-parallel** cross-problem protocol (design doc
S5.1/S6.1).

Per batch, in this fixed order:

1. ``arm.begin_batch(i)`` freezes whatever cross-problem knowledge (harness snapshot / raw
   experience pool) this batch is allowed to see.
2. **Serially**, before any parallel work starts, every problem in the batch gets its injected
   text via ``arm.system_prompt_for(problem)``.
3. **In parallel** (the only parallel step), each problem is solved via ``run_problem_fn``.
4. **Serially, in the batch's original order**, ``record_selection`` / ``observe`` are called and
   per-problem metrics are logged.
5. ``arm.end_batch(i)`` is only called once every problem in the batch has finished; harness-scale
   and per-role call-accounting metrics are logged right after.

Steps 2 and 4 must stay outside the parallel region because ``SkillStore``'s write path
(``_next_id()`` in particular) is not safe under concurrent access -- two problems selecting or
recording against the *live* store at the same time could silently corrupt the id sequence and
the audit trail. This is a structural invariant of the harness, not an incidental choice; see
``arms.py``'s module docstring for the matching guarantee on the arm side (every arm freezes its
state at ``begin_batch`` and only ``end_batch`` may write to it).

This module never imports ``alphaapollo.core.environments.memory`` (that boundary belongs to the
in-problem solution memory this driver deliberately does not touch), and it never hardcodes the
upstream ``run_problem`` / runtime-construction call inside ``run_stream`` itself -- both are
injected as callables so this module's core logic is testable without constructing a real
environment, agent, or network client. The real upstream is only ever imported --lazily, inside
``run()`` -- for the CLI entry point actually used to launch a run.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from alphaapollo.core.harness.accounting import role_scope
from alphaapollo.core.harness.loader import batches
from alphaapollo.core.harness.reflect import _GT_CHANNEL, sanitize_feedback

logger = logging.getLogger(__name__)

# The one confirmed ground-truth leakage channel (design doc S5.4 gate 2): AlphaApollo's own
# Python verification tool (`core/tools/informalmath_verify.py:107,176`) writes a line like
# "Matches ground truth: True/False" straight into its return text, and `env.py`'s `_parse_action`
# parses this tag *unconditionally* -- it only fails to fire today because no prompt template
# advertises it, which is not a real defense. Matched as a bare substring/tag rather than a full
# opening+closing pair, deliberately: even a truncated or malformed occurrence of the tag in an
# action is enough to prove the model attempted to invoke it.
_VERIFY_TAG = re.compile(r"<informalmath_verify>")


class StreamAbort(RuntimeError):
    """Raised when the first batch produced no usable trajectory at all.

    The per-problem log-and-continue path exists so a flaky provider cannot destroy a multi-hour
    adaptation run, and it works -- but it degrades a dead problem into the same all-zero result a
    genuinely-wrong answer would produce, which means a *wholly* broken configuration finishes
    quietly and looks like a completed run. That is not theoretical: the first real end-to-end run
    lost every problem to ``KeyError: 'data_source'`` (a key the stream loader did not carry but
    that upstream's ``run_problem`` indexes unconditionally) and still reported success.

    The breaker is scoped to the first batch on purpose. Until one batch has completed, nothing
    has demonstrated that the config, the stream schema, the credentials and the upstream call
    path fit together; after that, a wholly-failed batch is a provider outage, which is exactly
    what the surrounding resilience is for.
    """


class FrozenArmWithoutState(ValueError):
    """Raised when a run is configured as frozen but the arm loaded no cross-problem state.

    Held-out evaluation points a frozen arm at whatever the adaptation run produced -- a skill
    store, a raw-experience pool. Point it somewhere empty (a typo, a stale path, an adaptation
    run that never finished) and the arm injects nothing and behaves exactly like Baseline. The
    result is three identical curves, a wasted held-out budget, and nothing in the output
    explaining why: the failure is indistinguishable from a genuine null result, which is the
    worst possible way for it to fail in an experiment whose whole purpose is to interpret null
    results.

    Refused up front, before a single problem runs, because by the time the numbers are in there
    is no way to tell this apart from the finding it imitates.
    """


def assert_frozen_arm_has_state(arm, *, arm_name: str, frozen: bool, state_path=None) -> None:
    """Refuse a frozen run whose arm has cross-problem state and loaded none of it.

    Only fires when all three hold: the run is frozen, the arm *has* a notion of cross-problem
    state (``cross_problem_state_size()`` is not ``None``, so Baseline is exempt), and that state
    is empty. An unfrozen arm starting empty is the normal cold start -- the adaptation run's
    first batch -- and must not be blocked.
    """
    if not frozen:
        return
    size = arm.cross_problem_state_size()
    if size is None or size > 0:
        return
    raise FrozenArmWithoutState(
        f"arm {arm_name!r} is configured frozen but loaded no cross-problem state from "
        f"{state_path!r}. A frozen arm with an empty store/pool injects nothing and behaves "
        f"exactly like the baseline, which is indistinguishable from a null result. Point this "
        f"run at the directory the adaptation run wrote, or unset `frozen`."
    )


class LeakageError(RuntimeError):
    """Raised when a policy action contains the ground-truth verification tool call.

    This is a data-validity failure, not a transient one: once a rollout has attempted to call
    ``<informalmath_verify>``, the ground-truth-matching text it got back may already have reached
    the model's context, so nothing downstream of that action (this round's outcome, any skill
    later compiled from it) can be trusted. Callers must let this propagate and stop, never catch
    it alongside ordinary per-problem failures.
    """


def assert_no_gt_tool_call(action_text: str) -> None:
    """Fail fast if ``action_text`` invokes the ground-truth verification tool.

    Called once per round's action text. Deliberately does not try to recover or sanitize the
    surrounding text -- by the time this tag appears in an action, the leak has already happened
    (the tool's ``Matches ground truth: ...`` reply is what everything downstream would need
    sanitizing *from*), so the only correct response is to stop using this rollout's data.
    """
    if _VERIFY_TAG.search(action_text or ""):
        raise LeakageError("ground-truth leakage: action text invokes <informalmath_verify>, whose tool reply echoes 'Matches ground truth: True/False' back to the policy")


# The all-zero shape `extract_result` returns for an empty `step_outputs`, and that `run_stream`
# reuses to stand in for a problem whose execution raised an ordinary (non-leakage) exception --
# both cases mean "no usable signal from this problem", and must look identical to every
# downstream consumer (arms, tracker) rather than requiring them to special-case two shapes.
_ZERO_RESULT: dict[str, Any] = {
    "pass1_round0": 0,
    "pass_final": 0,
    "final_answer_given": "",
    "verifier_feedback": "",
    "tool_errors": "",
    "reasoning_excerpt": "",
    "round_count": 0,
}

# Fields this module ever reads out of a `run_problem`-shaped payload. This is a whitelist, not a
# blacklist, deliberately mirroring `reflect.build_reflect_context`'s own contract: `problem_payload`
# and every `step_outputs` entry carry `ground_truth` / `gt_traj` (evolving_main.py:562-564,
# 714-716, 828-833 -- at BOTH the payload's top level and on every single step_outputs entry), plus
# `observation` / `next_observation` / `infos` / `policy_memory` / `verifier_memory` / `data_source`,
# none of which anything below ever names. A `dict(entry)` copy or a `**entry` expansion anywhere in
# this module would silently reintroduce the leak this whitelist exists to prevent -- the fields
# below are read one at a time, by name, and nothing else.
_MAX_FIELD_CHARS = 2000


def _truncate(text: str, limit: int = _MAX_FIELD_CHARS) -> str:
    """Cap a field's length before it can ever reach a prompt. Applied *after* `sanitize_feedback`
    (never before): sanitizing first guarantees a GT-channel line is scrubbed in full even when it
    would otherwise straddle the truncation boundary."""
    text = text or ""
    return text if len(text) <= limit else text[:limit] + " ...(truncated)"


def _action_list(actions: Any) -> list[str]:
    """Normalize `policy_actions` to a list of strings. It is a list in the real schema (one raw
    action string per env; real payloads only ever run a single env for the informal_math_evolving
    path, but this stays list-shaped regardless -- see `evolving_main.py:556-568`). A bare string
    is accepted too, defensively, in case a caller hands this an already-flattened value -- and is
    wrapped in a single-element list rather than iterated character-by-character."""
    if isinstance(actions, str):
        return [actions] if actions else []
    if not actions:
        return []
    return [str(a) for a in actions if a]


def _as_text(actions: Any) -> str:
    return "\n".join(_action_list(actions))


def _policy_rounds(step_outputs: list[dict]) -> dict[int, list[dict]]:
    """Group ``role == "policy"`` entries by ``evolving_round`` (never ``role == "verifier"``
    entries -- the two are interleaved in the same flat list, `evolving_main.py:556-568` for
    policy vs. `:704-721` for verifier, and mixing them up is exactly the "policy 与 verifier 的行
    混淆" failure mode this function exists to rule out structurally). Only ever reads `role`,
    `evolving_round`, and `step` -- see the whitelist note above. Within one round there is
    normally one `step_outputs` entry per policy step (a round can retry across several steps
    before an environment is done); entries are sorted by `step` so the *last* one in each round's
    list is always that round's own concluding attempt, never an earlier in-progress one."""
    grouped: dict[int, list[dict]] = {}
    for entry in step_outputs:
        if entry.get("role") != "policy":
            continue
        grouped.setdefault(entry.get("evolving_round", 0), []).append(entry)
    for entries in grouped.values():
        entries.sort(key=lambda e: e.get("step", 0))
    return grouped


def _last_verifier_text(step_outputs: list[dict]) -> str:
    """The most recent ``role == "verifier"`` entry's own action text (its `<report>...</report>`
    reply) -- never a policy entry's. Returns `""` if the verifier path was never entered (e.g.
    `verifier_configs.get("enabled")` was false for this run).

    This is the FALLBACK path for ``verifier_feedback``, used only when no
    ``role == "verifier_aggregation"`` entry exists (see ``_last_verifier_aggregation_report``,
    which is tried first). A real ``role == "verifier"`` entry's ``policy_actions`` is the
    verifier LLM's *own* raw reply stream -- its `<think>` reasoning, any `<python_code>` block it
    ran to double-check the policy's arithmetic, AND its concluding `<report>...</report>` verdict
    all concatenated together (see a real recorded example in
    ``tests/harness/fixtures/real_problem_payload.json``'s ``role == "verifier"`` entry) -- not a
    clean verdict string. It is kept as a fallback rather than removed because a run with
    ``verifier_env.enable`` on but no aggregation step configured still needs *something* for
    ``verifier_feedback``, and this is the only verifier-authored text available in that shape."""
    verifier_entries = [e for e in step_outputs if e.get("role") == "verifier"]
    if not verifier_entries:
        return ""
    return _as_text(verifier_entries[-1].get("policy_actions"))


def _last_verifier_aggregation_report(step_outputs: list[dict]) -> str | None:
    """The most recent ``role == "verifier_aggregation"`` entry's ``representative_report`` --
    the PREFERRED source for ``verifier_feedback`` (task-14 Finding 1).

    ``role == "verifier_aggregation"`` is a third row shape this module did not previously know
    about: it is emitted once per round, after every individual ``role == "verifier"`` pass for
    that round has run, and carries the majority-vote outcome (``majority_judgment``,
    ``judgment_counts``) plus ``representative_report`` -- the single verifier reply that produced
    the majority judgment, already isolated from that verifier's own `<think>`/`<python_code>`
    scratch work (see ``final_verifier_actions`` for the raw stream this was extracted from, which
    this module never reads). This is exactly the clean natural-language verdict text Reflect
    needs (e.g. "The policy agent's solution appears correct. ... All calculations are verified
    and consistent." -- see the fixture above), where ``_last_verifier_text``'s raw
    ``role == "verifier"`` path instead hands back that verifier's entire `<think>` + tool-call +
    `<report>` stream.

    Returns ``None`` -- never ``""`` -- both when no ``verifier_aggregation`` entry exists at all
    and when the most recent one's ``representative_report`` is itself empty/missing, so
    ``extract_result`` can treat both cases identically: fall back to ``_last_verifier_text``."""
    agg_entries = [e for e in step_outputs if e.get("role") == "verifier_aggregation"]
    if not agg_entries:
        return None
    return agg_entries[-1].get("representative_report") or None


def _sanitize_reasoning_excerpt(text: str) -> str:
    """Scrub the GT-matching-line leak channel from a policy reasoning excerpt WITHOUT stripping
    ``<think>...</think>`` blocks (task-14 Finding 2) -- the one respect in which this deliberately
    does NOT behave like ``reflect.sanitize_feedback``.

    ``reflect.sanitize_feedback`` (via its private ``_strip_reasoning`` helper) strips think blocks
    because it feeds text that gets *parsed* for structured fields (Reflect's ``ACTION:``/`SCOPE:``
    lines, the curator's ``ADD:``/``MERGE:``/etc.) -- an unstripped think block sitting before the
    real formatted answer would win first-match-by-position over the real answer, and that
    stripping behavior is correct there and is NOT touched by this module (see ``reflect.py`` /
    ``evolver.py``, both untouched here).

    ``reasoning_excerpt`` is different: it is *captured*, never parsed. AlphaApollo's own policy
    prompt (``core/environments/prompts/informal_math_evolving.py``) forces every solving attempt
    into ``<think>...</think>`` -- "This process MUST be enclosed within `<think> </think>` tags"
    -- so the think block IS the model's actual reasoning, not scratch noise around it. A real
    recorded rollout's think block ran to 2241 characters of genuine step-by-step derivation
    (``tests/harness/fixtures/real_problem_payload.json``); running that text through
    ``sanitize_feedback`` collapses it to just the trailing ``<answer>...</answer>`` tag, leaving
    Reflect nothing to distill a lesson from on a failed problem. This function keeps the think
    content and only removes the GT-channel line (``core/tools/informalmath_verify.py``'s
    ``Matches ground truth: .../Matches GT: ...`` text, which a reasoning trace can narrate/quote
    even when it did not come from a `<think>` block itself) -- reusing ``reflect.py``'s own
    ``_GT_CHANNEL`` regex (imported, not duplicated) as the single source of truth for what that
    line looks like, the same private-import pattern ``arms.py``/``evolver.py`` already use for
    ``reflect._strip_reasoning``."""
    text = text or ""
    return "\n".join(line for line in text.split("\n") if not _GT_CHANNEL.match(line))


def _tool_error_text(tool_events: list) -> str:
    """Pull genuine tool-execution error text out of a round's `tool_events`
    (`utils/utils.py:collect_tool_events`, `{"tool_name", "tool_input", "raw_observation",
    "tool_payload"}` per event) -- and ONLY `tool_payload["stderr"]` / a non-"Finished"
    `tool_payload["run_status"]` (the python_code tool's real error channel,
    `core/tools/python_code.py:193-273`). This never reads a tool's `stdout` or
    `raw_observation`: the `informalmath_verify` tool's own `stdout` is exactly where the
    'Matches ground truth: .../Matches GT: ...' leak text lives (`core/tools/
    informalmath_verify.py`'s `call_informalmath_verify`), and that tool can also legitimately
    print a ground-truth-derived execution result in the same field. Restricting to
    stderr/run_status excludes that whole field *structurally*, rather than relying solely on
    `sanitize_feedback()` to catch a specific known phrase inside it after the fact."""
    lines: list[str] = []
    for event in tool_events or []:
        if not isinstance(event, dict):
            continue
        payload = event.get("tool_payload")
        if not isinstance(payload, dict):
            continue
        stderr = (payload.get("stderr") or "").strip()
        if stderr:
            lines.append(stderr)
        elif payload.get("run_status") not in (None, "Finished"):
            lines.append(f"{event.get('tool_name', 'tool')}: {payload.get('run_status')}")
    return "\n".join(lines)


def extract_result(problem_payload: dict) -> dict:
    """Extract the seven fields every arm/metric consumer needs from one problem's
    ``run_problem``-shaped payload: ``pass1_round0``, ``pass_final``, ``final_answer_given``,
    ``verifier_feedback``, ``tool_errors``, ``reasoning_excerpt``, ``round_count``.

    Reads ``problem_payload["step_outputs"]`` -- a flat list mixing ``role == "policy"`` entries
    (`evolving_main.py:556-568`) and ``role == "verifier"`` entries (`:704-721`); a single
    ``evolving_round`` can span several policy entries (one per step) before the round concludes.
    Two things this deliberately does NOT use, both explained in the design doc (S6.2④): the
    payload's own ``success_rate`` (an average across evolving rounds, not Pass@1 -- with
    ``evolving_round=3`` it can only take the values {0, 1/3, 2/3, 1}), and any round before the
    last one for anything other than ``pass1_round0``. The clean signal is each policy round's own
    ``policy_answer_correct``, read off that round's *last* step (its concluding attempt):

    - ``pass1_round0`` is round 0's correctness -- the harness-injection effect *before* any
      in-problem self-correction had a chance to run, which is why it is the cleanest measure of
      what the cross-problem mechanism itself contributed.
    - ``pass_final`` is the last round's correctness -- the joint in-problem + cross-problem
      effect.
    - ``round_count`` is the number of *distinct* ``evolving_round`` values seen among policy
      entries (not ``max(evolving_round) + 1``): the two agree whenever rounds run contiguously
      from 0, which is the only way `evolving_main.py`'s own `for evolving_round in
      range(runtime["evolving_round"])` loop ever produces them, but counting distinct values
      degrades safely instead of silently overcounting if that ever stops being true.
    - ``final_answer_given`` / ``reasoning_excerpt`` / ``tool_errors`` all describe the *last*
      round specifically, since that is the attempt a failure-triggered Reflect call would
      actually be reflecting on; ``verifier_feedback`` prefers the most recent
      ``role == "verifier_aggregation"`` entry's ``representative_report`` (a clean verdict
      string), falling back to the most recent raw ``role == "verifier"`` entry's own reply only
      when no aggregation entry is present (see ``_last_verifier_aggregation_report`` /
      ``_last_verifier_text``) -- normally that same last round's own verifier pass either way.

    Every action text this payload carries -- policy or verifier -- is checked for the
    ground-truth tool-call leak (``assert_no_gt_tool_call``) before anything else happens; a hit
    raises ``LeakageError``, which is not caught here and must propagate to the caller.
    ``verifier_feedback`` and ``tool_errors`` are passed through ``sanitize_feedback()`` (which
    strips ``<think>`` blocks -- correct there, since a verifier's report text is a candidate
    Reflect-prompt fragment, not the substantive content itself); ``reasoning_excerpt`` instead
    goes through ``_sanitize_reasoning_excerpt()``, which scrubs the same GT-channel line WITHOUT
    stripping ``<think>`` blocks, because for the *policy's own* reasoning the think block -- the
    genuine, required-by-prompt step-by-step derivation -- is the one piece of substantive content
    a failure-triggered Reflect call needs (see that function's docstring for the full rationale;
    task-14 Finding 2). A reasoning trace can itself narrate/quote the tool's GT-matching line even
    when the verifier report proper does not (see ``reflect.py``'s module docstring), which is why
    the GT-channel scrub still applies to ``reasoning_excerpt`` even though the think-stripping
    half of ``sanitize_feedback`` does not. This function never reads ``ground_truth`` /
    ``gt_traj`` / ``data_source`` / ``observation`` / ``next_observation`` / ``infos`` /
    ``policy_memory`` / ``verifier_memory`` from ``problem_payload`` or any ``step_outputs`` entry
    -- see the whitelist note above.

    Note on Finding 3 (not a defect): on a problem the policy solved correctly, both
    ``final_answer_given`` and ``reasoning_excerpt`` will legitimately contain the same string as
    ``ground_truth`` -- the model's own correct answer necessarily equals the correct answer. This
    is NOT the leakage this module guards against (that is ``assert_no_gt_tool_call`` /
    ``sanitize_feedback``'s GT-channel scrub, both about the verification *tool*'s echoed
    ground-truth text, not the model's own arrived-at answer). It is also harmless in practice:
    Reflect (``arms.py``) only ever calls this compilation path on a FAILED problem, at which point
    the model's own answer is, by definition, different from ``ground_truth``. Do not
    ``!= ground_truth``-scrub ``final_answer_given`` / ``reasoning_excerpt`` to "fix" this
    coincidence -- see ``tests/harness/test_driver.py``'s
    ``test_extract_result_never_leaks_the_ground_truth_sentinel``, which already proves the real
    leak channels are closed using a sentinel that could never legitimately appear in a model's own
    answer, and must keep passing unmodified.

    An empty (or missing) ``step_outputs``, or one with no ``role == "policy"`` entries at all,
    returns an all-zero structure -- this is a normal, expected shape (e.g. a problem that
    produced no rounds at all), not an error condition, so it never raises.
    """
    step_outputs = (problem_payload or {}).get("step_outputs") or []

    for entry in step_outputs:
        for action in _action_list(entry.get("policy_actions")):
            assert_no_gt_tool_call(action)

    rounds = _policy_rounds(step_outputs)
    if not rounds:
        return dict(_ZERO_RESULT)

    round_indices = sorted(rounds)
    round0_entry = rounds[round_indices[0]][-1]
    final_round_steps = rounds[round_indices[-1]]
    final_entry = final_round_steps[-1]

    final_actions_text = _as_text(final_entry.get("policy_actions"))
    final_answer_given = final_entry.get("policy_answer") or final_actions_text

    final_round_tool_events: list = []
    for step_entry in final_round_steps:
        final_round_tool_events.extend(step_entry.get("tool_events") or [])

    # Prefer the clean role=="verifier_aggregation" verdict (representative_report); only fall
    # back to the raw role=="verifier" reply stream when no aggregation entry is present. See
    # _last_verifier_aggregation_report / _last_verifier_text docstrings (task-14 Finding 1).
    verifier_feedback_text = _last_verifier_aggregation_report(step_outputs)
    if verifier_feedback_text is None:
        verifier_feedback_text = _last_verifier_text(step_outputs)

    return {
        "pass1_round0": int(bool(round0_entry.get("policy_answer_correct"))),
        "pass_final": int(bool(final_entry.get("policy_answer_correct"))),
        "final_answer_given": _truncate(sanitize_feedback(str(final_answer_given))),
        "verifier_feedback": _truncate(sanitize_feedback(verifier_feedback_text)),
        "tool_errors": _truncate(sanitize_feedback(_tool_error_text(final_round_tool_events))),
        # NOT sanitize_feedback() -- that strips <think> blocks, which here would strip the
        # policy's actual reasoning (task-14 Finding 2). See _sanitize_reasoning_excerpt.
        "reasoning_excerpt": _truncate(_sanitize_reasoning_excerpt(final_actions_text)),
        "round_count": len(round_indices),
    }


def run_stream(
    *,
    problems: list[dict],
    arm: Any,
    runtime_factory: Callable[[str], dict],
    run_problem_fn: Callable[[int, dict, dict], dict],
    tracker: Any,
    accountant: Any,
    batch_size: int = 8,
    max_workers: int = 8,
    start_batch: int = 0,
    on_batch_complete: Callable[[int], None] | None = None,
    trajectory_sink: Callable[[int, dict], None] | None = None,
) -> dict:
    """Run ``problems`` to completion against ``arm``'s cross-problem mechanism, batch-serial /
    within-batch-parallel (see module docstring for the five-step protocol).

    A single problem raising an ordinary exception is caught, counted in the returned
    ``n_errors``, logged, and degrades to :data:`_ZERO_RESULT` so the rest of the batch (and the
    rest of the stream) keeps running -- one flaky call must never abort an entire adaptation run.
    ``LeakageError`` is the one exception that is never caught this way: it always propagates out
    of this function immediately, because a leaked ground-truth signal invalidates this rollout's
    data outright, and continuing would just spend more budget on unusable data.

    Trailing problems that do not fill a whole batch are dropped (``loader.batches``), so
    ``n_problems`` in the returned summary can be smaller than ``len(problems)``.

    ``start_batch`` skips that many leading batches outright -- the resume path (see
    ``harness.resume``, which owns the decision about *which* batch that is and the cleanup of
    whatever partial rows the aborted batch left behind). Skipped batches consume no budget and,
    critically, do not call ``arm.begin_batch``/``end_batch``: the arm's state already reflects
    them, having been reloaded from disk.

    ``on_batch_complete(batch_idx)`` fires once a batch is durably finished -- after
    ``end_batch`` has committed any harness change and after that batch's metrics are logged --
    and is how the resume marker advances. It is deliberately *not* called for a batch whose
    ``end_batch`` raised: that batch's harness update did not happen, so resuming past it would
    skip the update permanently.

    ``trajectory_sink(problem_idx, payload)`` receives each successful rollout for archival (see
    ``harness.trajectory``). It is called inside the serial step-4 loop, never from a worker
    thread, so a sink writing to one directory needs no locking.
    """
    n_problems = 0
    n_errors = 0

    def _run_one(problem_idx: int, problem: dict, runtime: dict) -> dict:
        # Entered from *inside* the worker thread that actually executes this call -- role_scope
        # sets a contextvars.ContextVar, which is per-thread; setting it from the submitting
        # thread would never be visible to the pool thread that runs `run_problem_fn` (see
        # accounting.py's module docstring). Every LLM call `run_problem_fn` makes on this thread
        # (policy, verifier, the in-problem summarizer/aggregator it constructs internally) is
        # therefore attributed to the "solver" role.
        with role_scope("solver"):
            return run_problem_fn(problem_idx, problem, runtime)

    for batch_idx, batch in enumerate(batches(problems, batch_size)):
        if batch_idx < start_batch:
            continue
        is_first_executed_batch = batch_idx == start_batch

        arm.begin_batch(batch_idx)

        # Step 2 -- serial, arm-mutating (system_prompt_for may update selection bookkeeping):
        # freeze every problem's injected text *before* any parallel work starts.
        system_prompts = [arm.system_prompt_for(problem) for problem in batch]

        # Step 3 -- parallel, arm-read-only: only run_problem_fn itself executes concurrently.
        # Futures are submitted, then awaited, in the batch's own order (not `as_completed`), so
        # `outcomes` always lines up with `batch` positionally regardless of which one finishes
        # first -- the actual work still runs concurrently in the pool's worker threads either way.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_one, problem["problem_idx"], problem, runtime_factory(system_prompt)) for problem, system_prompt in zip(batch, system_prompts)]
            outcomes: list[tuple[str, Any]] = []
            for future in futures:
                try:
                    outcomes.append(("ok", future.result()))
                except LeakageError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- one problem's failure must not abort the batch
                    outcomes.append(("error", exc))

        # Circuit breaker: a first batch with nothing but failures is a broken configuration, and
        # every downstream stage is built to survive exactly this, which is what makes it
        # invisible. Tripped before step 4 so no management budget is spent reflecting on it.
        # Keyed to the first batch *this process* executes, not literally batch 0: a resumed run
        # is a new process with its own credentials, proxy and stream file, so it needs the same
        # proof-of-life before it is allowed to spend hours degrading every problem to zeros.
        if is_first_executed_batch and all(kind == "error" for kind, _ in outcomes):
            first_error = outcomes[0][1]
            raise StreamAbort(f"every problem in the first batch failed; the run is misconfigured, not flaky. First error: {first_error!r}") from first_error

        # Step 4 -- serial, batch order, arm-mutating: record_selection / observe / metrics.
        # Fixed order keeps jsonl/wandb steps reproducible run over run, regardless of the
        # parallel step's actual completion order.
        for problem, (kind, payload) in zip(batch, outcomes):
            step = problem["problem_idx"]
            if kind == "error":
                n_errors += 1
                logger.warning("run_problem failed for problem %s; recording an all-zero result", step, exc_info=payload)
                result = dict(_ZERO_RESULT)
            else:
                # LeakageError from a poisoned action inside the payload propagates from here,
                # uncaught, aborting the whole stream -- see module/function docstring.
                result = extract_result(payload.get("problem_payload", {}))
                # Archived before anything downstream can fail, so a rollout that triggers a bug
                # in observe/Reflect is still on disk to debug against.
                if trajectory_sink is not None:
                    trajectory_sink(step, payload.get("problem_payload", {}))
            n_problems += 1

            arm.record_selection(problem, result)
            arm.observe(problem, result)
            # `adapt/error` is what keeps a dead problem distinguishable from a wrong answer.
            # Both degrade to pass_final == 0, so without this flag the two are identical in the
            # record -- and the three arms run concurrently against one provider, so they lose
            # *different* problems to the same rate-limit storm. A few silently-lost problems
            # bias one arm's rate by several points, against a measured noise floor of ~3.6
            # points at n=149: large enough to invent or erase the effect being measured.
            # With the flag, the arms can be compared on the intersection of problems all three
            # actually completed, which is the only fair comparison available.
            # `topic` rides along so the per-topic breakdown -- required for *all three* arms --
            # is a group-by rather than a join against the stream file. The Baseline arm writes
            # no selection log, so without this its per-topic numbers would be the only ones
            # needing a different code path to compute, which is how breakdowns end up
            # inconsistent between arms.
            tracker.log(step, {"adapt/pass1_round0": result["pass1_round0"],
                               "adapt/pass_final": result["pass_final"],
                               "adapt/error": int(kind == "error"),
                               "topic": problem.get("topic") or ""})

        # Step 5 -- only now may the harness/pool actually change. A misbehaving arm's own
        # end_batch() failing must not take down the rest of the adaptation stream (the
        # assignment's "skill-update failures ... must never break the underlying baseline"), so
        # it gets the same log-and-continue treatment as an individual solver failure; every arm
        # shipped in `arms.py` already degrades to a no-op edit list internally, so this is a
        # second line of defense against a future/foreign arm implementation, not the primary one.
        batch_committed = True
        try:
            arm.end_batch(batch_idx)
        except Exception:
            batch_committed = False
            logger.exception("arm.end_batch failed for batch %s; harness left unchanged for this batch", batch_idx)

        last_step = batch[-1]["problem_idx"]
        if hasattr(arm, "store"):
            tracker.log_harness_state(last_step, arm.store)
        tracker.log_accounting(last_step, accountant)

        # Last thing in the iteration, and only for a batch that actually committed. Advancing
        # past a batch whose end_batch raised would drop that batch's harness update for good --
        # a resumed run would never revisit it, and the harness would silently be missing
        # everything those problems should have taught it.
        if on_batch_complete is not None and batch_committed:
            on_batch_complete(batch_idx)

    return {"n_problems": n_problems, "n_errors": n_errors}


def run(config: str | None = None) -> None:
    """Fire entry point: ``python -m alphaapollo.core.generation.evolving.evolving_harness_main
    --config examples/configs/harness_evo.yaml``.

    Wires this module's ``run_stream`` around AlphaApollo's existing, unmodified
    ``informal_math_evolving`` solver loop. Every upstream symbol this needs
    (``load_run_configuration`` / ``create_runtime_for_problem`` / ``run_problem`` / ``Agent``) is
    imported lazily, inside this function, specifically so importing this module -- e.g. to unit
    test ``run_stream`` / ``extract_result`` / ``assert_no_gt_tool_call`` -- never pulls in the
    upstream generation stack (env managers, the real OpenAI client construction, `verl`) at
    collection time.

    The ``harness:`` config section is read as sketched by the companion config task (batch_size,
    max_workers, seed, frozen, feedback_level, store_root, caps, budget, wandb, stream_path); a
    fresh ``Agent`` is constructed per problem inside ``runtime_factory`` (rather than reusing one
    shared instance) so that concurrently-running problems in the same batch, which the harness
    protocol deliberately gives *different* injected text, never race on a single mutable
    ``Agent.system_prompt`` attribute.

    ``run_stream``/``extract_result`` are exercised by ``tests/harness/test_driver.py`` against a
    minimal payload, a fully-realistic one, and a recorded real rollout. This function itself has
    no unit test -- it constructs the live upstream stack -- but it *has* been run end to end
    against the real API several times, most substantially a 12-problem Evo-Harness run over the
    prepared adaptation stream (1354.7s; 93 solver + 20 management calls; cross-problem reuse
    confirmed in ``selection_log.jsonl``). Every defect that run exposed is fixed and pinned by a
    test; see the commits referenced from ``README_HARNESS.md``'s traceability section.

    What this function still lacks is the committed run configs -- runs so far have used
    hand-assembled YAML outside the repo.
    """
    if not config:
        raise ValueError("--config is required, e.g. --config examples/configs/harness_evo.yaml")

    from omegaconf import OmegaConf

    from alphaapollo.core.generation.evolving.evolving_main import create_runtime_for_problem
    from alphaapollo.core.generation.evolving.evolving_main import run_problem as upstream_run_problem
    from alphaapollo.core.generation.evolving.utils.agent import Agent
    from alphaapollo.core.generation.evolving.utils.utils import load_run_configuration
    from alphaapollo.core.harness.accounting import CallAccountant, install_accounting
    from alphaapollo.core.harness.arms import build_arm
    from alphaapollo.core.harness.loader import load_stream
    from alphaapollo.core.harness.resume import fingerprint, prepare_resume, write_progress
    from alphaapollo.core.harness.store import Budget, Caps
    from alphaapollo.core.harness.tracker import HarnessTracker
    from alphaapollo.core.harness.trajectory import save_trajectory

    cfg_bundle = load_run_configuration(config)
    full_cfg = OmegaConf.to_container(cfg_bundle["cfg"], resolve=True)
    harness_cfg = full_cfg.get("harness") or {}
    if not harness_cfg:
        raise ValueError(f"{config} has no `harness:` section")

    # Same construction evolving_main.run() uses for its own `full_config` (evolving_main.py's
    # own run(), not this module's) -- kept identical so create_runtime_for_problem sees the same
    # shape regardless of which driver built it.
    full_config = {k: v for k, v in full_cfg.items() if not k.endswith("_config")}
    verifier_max_workers = int(OmegaConf.select(cfg_bundle["env_config"], "informal_math_evolving.concurrency.verifier_max_workers") or 0)

    problems = load_stream(harness_cfg["stream_path"])

    # Management-side agents (Reflect, the curators, the selector). Defaults to the same config
    # as the policy so a single-model setup needs no extra block; `harness.selector_model_cfg`
    # overrides it for selection alone, which is what the paper does (Appendix F: Claude Sonnet
    # 4.5 "for harness selection ... across all experiments", a stronger model than the solver).
    # Three management roles, three sampling settings, one model -- matching the reference
    # implementation (propose 0.3, curate 0.0, select 0.0). They are distinct `Agent` objects
    # only because `Agent` fixes its temperature at construction; `harness.mgmt_model_overrides`
    # carries just the deltas so the model, base_url and key are stated once.
    def _mgmt_agent(role: str) -> Any:
        cfg = dict(cfg_bundle["policy_model_cfg"])
        cfg.update((harness_cfg.get("mgmt_model_overrides") or {}).get(role) or {})
        return Agent(cfg)

    mgmt_agent = _mgmt_agent("reflect")
    curator_agent = _mgmt_agent("curator")
    selector_cfg = harness_cfg.get("selector_model_cfg")
    selector_agent = Agent(selector_cfg) if selector_cfg else _mgmt_agent("curator")
    arm_name = harness_cfg["arm"]
    if arm_name == "baseline":
        arm = build_arm(arm_name)
    elif arm_name == "raw":
        arm = build_arm(arm_name, agent=mgmt_agent,
                        budget=Budget(**(harness_cfg.get("budget") or {})),
                        pool_root=harness_cfg.get("store_root"),
                        frozen=bool(harness_cfg.get("frozen", False)))
    else:
        arm = build_arm(
            arm_name,
            store_root=harness_cfg["store_root"],
            agent=mgmt_agent,
            selector_agent=selector_agent,
            curator_agent=curator_agent,
            caps=Caps(**(harness_cfg.get("caps") or {})),
            budget=Budget(**(harness_cfg.get("budget") or {})),
            feedback_level=harness_cfg.get("feedback_level", "standard"),
            frozen=bool(harness_cfg.get("frozen", False)),
        )

    # Fail before spending a single call: a frozen arm that loaded nothing behaves like Baseline
    # and its result is indistinguishable from a genuine null finding.
    assert_frozen_arm_has_state(arm, arm_name=arm_name, frozen=bool(harness_cfg.get("frozen", False)),
                                state_path=harness_cfg.get("store_root"))

    wandb_cfg = harness_cfg.get("wandb") or {}
    # Six runs share one project (3 arms x adaptation/held-out). Defaulting the name from the
    # arm and the phase means the dashboard is readable even when a config forgets to set it --
    # an unnamed run gets a random wandb nickname, which makes the three arms indistinguishable
    # and defeats the only reason to watch this live.
    phase = "heldout" if harness_cfg.get("frozen") else "adapt"
    tracker = HarnessTracker(
        harness_cfg.get("run_dir", "./outputs/harness"),
        project=wandb_cfg.get("project"),
        run_name=wandb_cfg.get("run_name") or f"{phase}-{arm_name}",
        group=wandb_cfg.get("group") or phase,
        enabled=bool(wandb_cfg.get("enabled", False)),
        config=harness_cfg,
    )
    accountant = CallAccountant()
    uninstall = install_accounting(accountant, seed=harness_cfg.get("seed"), extra_body=harness_cfg.get("extra_body"))

    # Resume planning happens after the tracker exists (so its metrics.jsonl can be truncated)
    # but before a single call is billed. A fingerprint mismatch raises out of here, which is the
    # intended behaviour -- see `resume.plan_resume`.
    run_dir = harness_cfg.get("run_dir", "./outputs/harness")
    batch_size = int(harness_cfg.get("batch_size", 8))
    # Every log an aborted batch could have written per-problem rows into. Both arms record a
    # selection per problem in step 4; their batch-committed artifacts (skills/, harness_log,
    # pool.jsonl) are written only by end_batch and so need no truncation.
    partial_logs = {str(tracker.metrics_path): "step"}
    if getattr(arm, "store", None) is not None:
        partial_logs[str(arm.store.selection_log)] = "problem_idx"
    if getattr(arm, "pool_root", None) is not None:
        partial_logs[str(arm.pool_root / "selection_log.jsonl")] = "problem_idx"
    plan = prepare_resume(run_dir, fp=fingerprint(harness_cfg), batch_size=batch_size,
                          partial_logs=partial_logs)

    def on_batch_complete(batch_idx: int) -> None:
        write_progress(run_dir, completed_batches=batch_idx + 1, fp=fingerprint(harness_cfg))

    save_trajectories = bool(harness_cfg.get("save_trajectories", True))

    def trajectory_sink(problem_idx: int, payload: dict) -> None:
        save_trajectory(run_dir, problem_idx, payload)

    def runtime_factory(system_prompt: str) -> dict:
        # A fresh Agent per problem, not a shared one -- see the docstring above: problems within
        # the same batch run concurrently and may carry different injected text.
        policy_agent = Agent(cfg_bundle["policy_model_cfg"])
        policy_agent.system_prompt = system_prompt
        verifier_agent = Agent(cfg_bundle["verifier_cfg"]) if cfg_bundle["enable_verify"] else None
        return create_runtime_for_problem(cfg_bundle, cfg_bundle["env_config"], policy_agent, verifier_agent, full_config, verifier_max_workers)

    try:
        summary = run_stream(
            problems=problems,
            arm=arm,
            runtime_factory=runtime_factory,
            run_problem_fn=upstream_run_problem,
            tracker=tracker,
            accountant=accountant,
            batch_size=batch_size,
            max_workers=int(harness_cfg.get("max_workers", 8)),
            start_batch=plan.start_batch,
            on_batch_complete=on_batch_complete,
            trajectory_sink=trajectory_sink if save_trajectories else None,
        )
        logger.info("evolving_harness_main.run finished: %s", summary)
    finally:
        uninstall()
        tracker.finish()


if __name__ == "__main__":
    import fire

    fire.Fire(run)
