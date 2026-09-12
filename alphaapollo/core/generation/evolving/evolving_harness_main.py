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
from alphaapollo.core.harness.reflect import sanitize_feedback

logger = logging.getLogger(__name__)

# The one confirmed ground-truth leakage channel (design doc S5.4 gate 2): AlphaApollo's own
# Python verification tool (`core/tools/informalmath_verify.py:107,176`) writes a line like
# "Matches ground truth: True/False" straight into its return text, and `env.py`'s `_parse_action`
# parses this tag *unconditionally* -- it only fails to fire today because no prompt template
# advertises it, which is not a real defense. Matched as a bare substring/tag rather than a full
# opening+closing pair, deliberately: even a truncated or malformed occurrence of the tag in an
# action is enough to prove the model attempted to invoke it.
_VERIFY_TAG = re.compile(r"<informalmath_verify>")


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


def extract_result(problem_payload: dict) -> dict:
    """Extract the seven fields every arm/metric consumer needs from one problem's
    ``run_problem``-shaped payload: ``pass1_round0``, ``pass_final``, ``final_answer_given``,
    ``verifier_feedback``, ``tool_errors``, ``reasoning_excerpt``, ``round_count``.

    Two things this deliberately does NOT use, both explained in the design doc (S6.2④): the
    payload's own ``success_rate`` (an average across evolving rounds, not Pass@1 -- with
    ``evolving_round=3`` it can only take the values {0, 1/3, 2/3, 1}), and any round before the
    last one for anything other than ``pass1_round0``. The clean signal is each round's own
    ``policy_answer_correct``:

    - ``pass1_round0`` is round 0's correctness -- the harness-injection effect *before* any
      in-problem self-correction had a chance to run, which is why it is the cleanest measure of
      what the cross-problem mechanism itself contributed.
    - ``pass_final`` is the last round's correctness -- the joint in-problem + cross-problem
      effect.
    - the remaining fields describe the *last* round specifically, since that is the attempt a
      failure-triggered Reflect call would actually be reflecting on.

    Every round's action text is checked for the ground-truth tool-call leak
    (``assert_no_gt_tool_call``) before anything else happens; a hit raises ``LeakageError``,
    which is not caught here and must propagate to the caller. ``verifier_feedback``,
    ``tool_errors``, and ``reasoning_excerpt`` are all passed through ``sanitize_feedback()`` --
    every one of them can end up in a Reflect prompt (``arms.py``'s ``EvoHarnessArm.end_batch``),
    and a reasoning trace can itself narrate/quote the tool's GT-matching line even when the
    verifier report proper does not (see ``reflect.py``'s module docstring).

    An empty (or missing) ``step_outputs`` returns an all-zero structure -- this is a normal,
    expected shape (e.g. a problem that produced no rounds at all), not an error condition, so it
    never raises.
    """
    step_outputs = (problem_payload or {}).get("step_outputs") or []
    if not step_outputs:
        return dict(_ZERO_RESULT)

    for step in step_outputs:
        assert_no_gt_tool_call(step.get("policy_action", ""))

    round0 = step_outputs[0]
    final = step_outputs[-1]

    return {
        "pass1_round0": int(bool(round0.get("policy_answer_correct"))),
        "pass_final": int(bool(final.get("policy_answer_correct"))),
        "final_answer_given": final.get("policy_action", ""),
        "verifier_feedback": sanitize_feedback(final.get("verifier_report", "") or ""),
        "tool_errors": sanitize_feedback(final.get("tool_errors", "") or ""),
        "reasoning_excerpt": sanitize_feedback(final.get("reasoning_excerpt", final.get("policy_action", "")) or ""),
        "round_count": len(step_outputs),
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
            n_problems += 1

            arm.record_selection(problem, result)
            arm.observe(problem, result)
            tracker.log(step, {"adapt/pass1_round0": result["pass1_round0"], "adapt/pass_final": result["pass_final"]})

        # Step 5 -- only now may the harness/pool actually change. A misbehaving arm's own
        # end_batch() failing must not take down the rest of the adaptation stream (the
        # assignment's "skill-update failures ... must never break the underlying baseline"), so
        # it gets the same log-and-continue treatment as an individual solver failure; every arm
        # shipped in `arms.py` already degrades to a no-op edit list internally, so this is a
        # second line of defense against a future/foreign arm implementation, not the primary one.
        try:
            arm.end_batch(batch_idx)
        except Exception:
            logger.exception("arm.end_batch failed for batch %s; harness left unchanged for this batch", batch_idx)

        last_step = batch[-1]["problem_idx"]
        if hasattr(arm, "store"):
            tracker.log_harness_state(last_step, arm.store)
        tracker.log_accounting(last_step, accountant)

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
    from alphaapollo.core.harness.store import Budget, Caps
    from alphaapollo.core.harness.tracker import HarnessTracker

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

    mgmt_agent = Agent(cfg_bundle["policy_model_cfg"])
    arm_name = harness_cfg["arm"]
    if arm_name == "baseline":
        arm = build_arm(arm_name)
    elif arm_name == "raw":
        arm = build_arm(arm_name, agent=mgmt_agent, budget=Budget(**(harness_cfg.get("budget") or {})))
    else:
        arm = build_arm(
            arm_name,
            store_root=harness_cfg["store_root"],
            agent=mgmt_agent,
            caps=Caps(**(harness_cfg.get("caps") or {})),
            budget=Budget(**(harness_cfg.get("budget") or {})),
            feedback_level=harness_cfg.get("feedback_level", "standard"),
            frozen=bool(harness_cfg.get("frozen", False)),
        )

    wandb_cfg = harness_cfg.get("wandb") or {}
    tracker = HarnessTracker(
        harness_cfg.get("run_dir", "./outputs/harness"),
        project=wandb_cfg.get("project"),
        enabled=bool(wandb_cfg.get("enabled", False)),
        config=harness_cfg,
    )
    accountant = CallAccountant()
    uninstall = install_accounting(accountant, seed=harness_cfg.get("seed"))

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
            batch_size=int(harness_cfg.get("batch_size", 8)),
            max_workers=int(harness_cfg.get("max_workers", 8)),
        )
        logger.info("evolving_harness_main.run finished: %s", summary)
    finally:
        uninstall()
        tracker.finish()


if __name__ == "__main__":
    import fire

    fire.Fire(run)
