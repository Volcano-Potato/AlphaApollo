# --- fixture note -----------------------------------------------------------------------------
#
# `SIMPLIFIED_PAYLOAD` below is intentionally MINIMAL, not the fictional schema an earlier draft
# of this file used (that draft invented `round` / `policy_action` (singular) / `verifier_report`
# / `tool_errors` keys that do not exist anywhere in the real upstream). Every key it DOES use is
# real: `role`, `evolving_round`, `step`, `policy_actions` (a list), `policy_answer`,
# `policy_answer_correct`, `tool_events` -- it is just a payload with the surrounding noise fields
# (`observation`, `infos`, `rewards`, `dones`, `policy_memory`, ...) omitted, since `extract_result`
# never reads them anyway.
#
# `REALISTIC_PAYLOAD` is the fuller shape: `role` interleaves `"policy"`/`"verifier"` exactly as
# `evolving_main.py:556-568` (policy) and `:704-721` (verifier) actually produce it, every noise
# field is present (and, at every ground-truth-carrying position, filled with a unique sentinel),
# and one round's `tool_events` includes both a genuine tool error (`python_code`, non-"Finished"
# `run_status`) and an `informalmath_verify` call whose `tool_payload["stdout"]` carries the real
# GT-matching leak text this tool is known to emit (`core/tools/informalmath_verify.py`).
#
# Do not treat `SIMPLIFIED_PAYLOAD` as documentation of the real schema -- it only exists to keep
# the round0-vs-final / GT-sanitization assertions readable. `REALISTIC_PAYLOAD` is the one that
# actually exercises the whitelist.

import time

import pytest

from alphaapollo.core.generation.evolving.evolving_harness_main import LeakageError, assert_no_gt_tool_call, extract_result, run_stream
from alphaapollo.core.harness.accounting import CallAccountant
from alphaapollo.core.harness.arms import BaselineArm, CrossProblemArm
from alphaapollo.core.harness.tracker import HarnessTracker

SENTINEL = "SENTINEL_GT_9999"


def problems(n=16):
    return [{"problem_idx": i, "question": f"Q{i}", "ground_truth": str(i), "gt_traj": "", "topic": "number_theory", "problem_shape": "shape"} for i in range(n)]


SIMPLIFIED_PAYLOAD = {
    "step_outputs": [
        {"role": "policy", "evolving_round": 0, "step": 0, "policy_actions": ["<answer>1</answer>"], "policy_answer": "1", "policy_answer_correct": 0, "tool_events": [{"tool_name": "python_code", "tool_payload": {"stdout": "", "stderr": "NameError: x", "returncode": 1, "run_status": "Error"}}]},
        {"role": "verifier", "evolving_round": 0, "step": 0, "policy_actions": ["<report>Wrong modulus.</report>"]},
        {"role": "policy", "evolving_round": 1, "step": 0, "policy_actions": ["<answer>7</answer>"], "policy_answer": "7", "policy_answer_correct": 1, "tool_events": []},
        {"role": "verifier", "evolving_round": 1, "step": 0, "policy_actions": ["<report>Looks right.\nMatches GT: True</report>"]},
    ]
}


def _noise_step(role: str, evolving_round: int, step: int, policy_actions, **extra) -> dict:
    """Build one `step_outputs` entry with every real noise field present and stuffed with
    `SENTINEL`, plus whatever policy/verifier-specific fields the caller passes as `extra`. Mirrors
    the real shape produced at `evolving_main.py:556-568`/`:704-721` closely enough to prove
    `extract_result` never reads the noise."""
    entry = {
        "role": role,
        "step": step,
        "evolving_round": evolving_round,
        "observation": {"text": [f"prompt text mentioning {SENTINEL}"]},
        "policy_actions": policy_actions,
        "next_observation": {"text": [f"next obs mentioning {SENTINEL}"]},
        "rewards": [0.0],
        "dones": [False],
        "infos": [{"tool_infos": [], "note": SENTINEL}],
        "verifier": None,
        "ground_truth": SENTINEL,
        "gt_traj": SENTINEL,
        "data_source": SENTINEL,
        "policy_memory": [SENTINEL],
        "verifier_memory": [SENTINEL],
        "tool_events": [],
    }
    entry.update(extra)
    return entry


REALISTIC_PAYLOAD = {
    "problem_index": 42,
    "question": f"What is the answer to {SENTINEL}?",
    "ground_truth": SENTINEL,
    "gt_traj": SENTINEL,
    "data_source": SENTINEL,
    "step_outputs": [
        # Round 0: a single, wrong, done-in-one-step attempt. Only used for pass1_round0 (an int),
        # so it is safe for its own text fields to carry the sentinel -- nothing about round 0
        # should ever surface in extract_result's output.
        _noise_step("policy", 0, 0, [f"<answer>{SENTINEL}_WRONG</answer>"], policy_answer=f"{SENTINEL}_WRONG", policy_answer_correct=0),
        # Round 0's verifier reply is model-authored free text -- it carries the real "Matches
        # ground truth: ..." leak line (which sanitize_feedback scrubs whole-line), but is NOT
        # made to gratuitously restate the sentinel itself: a real model doesn't literally type a
        # synthetic test marker, and this field is never read into any returned value anyway
        # (round 0 only ever contributes an int, pass1_round0).
        _noise_step("verifier", 0, 0, ["<report>Wrong.\nMatches ground truth: False</report>"]),
        # Round 1 (the final round): two policy steps -- a tool-call step that hits a real error
        # AND calls informalmath_verify (whose own stdout legitimately carries a GT-derived leak
        # line, since that tool is allowed to see ground_truth to compute the match), then a
        # concluding answer step. tool_errors must pick up the python_code stderr and must NOT
        # pick up anything from informalmath_verify's stdout.
        _noise_step(
            "policy",
            1,
            0,
            ["<python_code>raise NameError('x')</python_code>"],
            policy_answer=None,
            policy_answer_correct=0,
            tool_events=[
                {"env_index": 0, "tool_name": "python_code", "tool_payload": {"stdout": "", "stderr": "NameError: name 'x' is not defined", "returncode": 1, "run_status": "Error"}, "raw_observation": f"contains {SENTINEL}"},
                {"env_index": 0, "tool_name": "informalmath_verify", "tool_payload": {"score": 0.0, "stdout": f"informalmath_verify: score=0.0\nMatches GT: False\n{SENTINEL}", "stderr": ""}, "raw_observation": f"contains {SENTINEL}"},
            ],
        ),
        _noise_step("policy", 1, 1, ["<answer>7</answer>"], policy_answer="7", policy_answer_correct=1),
        # This is the verifier reply extract_result actually surfaces (verifier_feedback comes
        # from the LAST verifier entry in the whole payload) -- same reasoning as round 0's:
        # real 'Matches GT: ...' leak line included (and must be scrubbed), no gratuitous sentinel.
        _noise_step("verifier", 1, 0, ["<report>Looks right.\nMatches GT: True</report>"]),
    ],
}


class RecordingArm(CrossProblemArm):
    def __init__(self):
        self.events = []
        self.batch = -1

    def begin_batch(self, batch_idx):
        self.batch = batch_idx
        self.events.append(("begin", batch_idx))

    def system_prompt_for(self, problem):
        self.events.append(("prompt", problem["problem_idx"], self.batch))
        return f"harness-v{self.batch}"

    def observe(self, problem, result):
        self.events.append(("observe", problem["problem_idx"]))

    def end_batch(self, batch_idx):
        self.events.append(("end", batch_idx))
        return []


def fake_run_problem(problem_idx, problem, runtime):
    return {"problem_idx": problem_idx, "problem_payload": SIMPLIFIED_PAYLOAD, "system_prompt_seen": runtime["policy_agent"].system_prompt}


def runtime_factory(system_prompt):
    agent = type("A", (), {"system_prompt": system_prompt})()
    return {"policy_agent": agent}


def test_extract_result_uses_round0_and_final_not_success_rate():
    result = extract_result(SIMPLIFIED_PAYLOAD)
    assert result["pass1_round0"] == 0
    assert result["pass_final"] == 1
    assert result["round_count"] == 2


def test_extract_result_sanitizes_the_gt_channel():
    assert "Matches GT" not in extract_result(SIMPLIFIED_PAYLOAD)["verifier_feedback"]


def test_leakage_detector_rejects_the_verify_tool_call():
    with pytest.raises(LeakageError):
        assert_no_gt_tool_call("let me check <informalmath_verify>x</informalmath_verify>")


def test_leakage_detector_allows_normal_actions():
    assert_no_gt_tool_call("<python_code>print(1)</python_code><answer>7</answer>")


def test_end_batch_runs_only_after_every_problem_in_the_batch(tmp_path):
    arm = RecordingArm()
    run_stream(problems=problems(16), arm=arm, runtime_factory=runtime_factory, run_problem_fn=fake_run_problem, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=4)

    order = [e for e in arm.events if e[0] in ("observe", "end")]
    first_end = next(i for i, e in enumerate(order) if e == ("end", 0))
    observed_before = {e[1] for e in order[:first_end] if e[0] == "observe"}
    assert observed_before == set(range(8))


def test_all_problems_in_a_batch_see_the_same_frozen_harness(tmp_path):
    arm = RecordingArm()
    run_stream(problems=problems(16), arm=arm, runtime_factory=runtime_factory, run_problem_fn=fake_run_problem, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=4)

    prompts = [e for e in arm.events if e[0] == "prompt"]
    assert {e[2] for e in prompts[:8]} == {0}
    assert {e[2] for e in prompts[8:]} == {1}


def test_incomplete_trailing_batch_is_dropped(tmp_path):
    arm = RecordingArm()
    summary = run_stream(problems=problems(20), arm=arm, runtime_factory=runtime_factory, run_problem_fn=fake_run_problem, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=4)
    assert summary["n_problems"] == 16


def test_a_failing_problem_does_not_abort_the_stream(tmp_path):
    def flaky(problem_idx, problem, runtime):
        if problem_idx == 3:
            raise RuntimeError("transient API error")
        return fake_run_problem(problem_idx, problem, runtime)

    summary = run_stream(problems=problems(8), arm=BaselineArm(), runtime_factory=runtime_factory, run_problem_fn=flaky, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=4)
    assert summary["n_problems"] == 8 and summary["n_errors"] == 1


# --- adversarial cases (own design, not from the brief) -------------------------------------
#
# The brief calls this out explicitly: this is where the closed loop actually gets assembled, so
# protocol correctness matters more than any single module in isolation. Each case below attacks
# one of the "decisions written in stone" or a specific race the batch-serial/within-batch-
# parallel design exists to close.


def test_live_store_mutation_during_parallel_execution_does_not_change_frozen_prompts(tmp_path):
    """Attack the protocol's core guarantee (module docstring step 2/4): if something mutates the
    arm's live state *while* the parallel step is still running, problems already given their
    prompt text in step 2 must not see a different prompt materialize underneath them, and later
    problems in the *same* batch (whose prompts were already frozen up front, before step 3 ever
    started) must not see it either -- only the *next* batch's begin_batch() is allowed to observe
    a mutation.

    Modeled with an arm whose system_prompt_for reads a live, mutable list, and a run_problem_fn
    that appends to that same list mid-execution (simulating a concurrent bad actor). Because
    step 2 (compute every system_prompt_for) runs entirely before step 3 (parallel execution)
    starts, the mutation performed *during* step 3 can only be observed by the *next* batch's
    prompts, never this one's.
    """
    live_state = ["skill-v0"]

    class LiveReadingArm(CrossProblemArm):
        def __init__(self):
            self.frozen = None

        def begin_batch(self, batch_idx):
            self.frozen = list(live_state)

        def system_prompt_for(self, problem):
            return ",".join(self.frozen)

    def poisoning_run_problem(problem_idx, problem, runtime):
        # Simulate a concurrent writer: every problem in the batch tries to mutate the live
        # state while the parallel step is in flight.
        live_state.append(f"poison-from-{problem_idx}")
        return {"problem_idx": problem_idx, "problem_payload": {"step_outputs": []}, "system_prompt_seen": runtime["policy_agent"].system_prompt}

    arm = LiveReadingArm()
    seen_prompts = {}

    def capturing_runtime_factory(system_prompt):
        return runtime_factory(system_prompt)

    def wrapped_run_problem(problem_idx, problem, runtime):
        seen_prompts[problem_idx] = runtime["policy_agent"].system_prompt
        return poisoning_run_problem(problem_idx, problem, runtime)

    summary = run_stream(problems=problems(8), arm=arm, runtime_factory=capturing_runtime_factory, run_problem_fn=wrapped_run_problem, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=8)

    assert summary["n_problems"] == 8
    # Every problem in this single batch must have seen the SAME pre-parallel snapshot,
    # regardless of how many problems had already "poisoned" the live list by the time a later
    # one's future actually ran.
    assert set(seen_prompts.values()) == {"skill-v0"}


def test_shuffled_completion_order_does_not_reorder_the_observe_log(tmp_path):
    """Attack the "as_completed order leaks into the log" failure mode explicitly called out in
    the brief: make problems finish in the REVERSE of submission order (highest problem_idx
    returns fastest) and verify record_selection/observe still fire in the batch's original
    order, not completion order.
    """
    arm = RecordingArm()

    def slow_for_low_idx(problem_idx, problem, runtime):
        # Problem 0 sleeps the longest, problem 7 returns first -- if run_stream naively used
        # as_completed(), the observe log would come back exactly reversed.
        time.sleep((7 - (problem_idx % 8)) * 0.01)
        return fake_run_problem(problem_idx, problem, runtime)

    run_stream(problems=problems(8), arm=arm, runtime_factory=runtime_factory, run_problem_fn=slow_for_low_idx, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=8)

    observed_order = [e[1] for e in arm.events if e[0] == "observe"]
    assert observed_order == list(range(8))


def test_leakage_error_aborts_before_later_problems_in_the_batch_are_observed(tmp_path):
    """A poisoned action must abort the whole run_stream call immediately -- not just skip that
    one problem, and not get folded into n_errors. Problem 5 (mid-batch) leaks; problems 6 and 7
    (later in the fixed batch order) must never reach observe()."""
    arm = RecordingArm()

    def leaking_run_problem(problem_idx, problem, runtime):
        if problem_idx == 5:
            payload = {
                "step_outputs": [
                    {"role": "policy", "evolving_round": 0, "step": 0, "policy_answer_correct": 0, "policy_actions": ["<informalmath_verify>x</informalmath_verify>"]},
                ]
            }
        else:
            payload = SIMPLIFIED_PAYLOAD
        return {"problem_idx": problem_idx, "problem_payload": payload, "system_prompt_seen": runtime["policy_agent"].system_prompt}

    with pytest.raises(LeakageError):
        run_stream(problems=problems(8), arm=arm, runtime_factory=runtime_factory, run_problem_fn=leaking_run_problem, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=1)

    observed = {e[1] for e in arm.events if e[0] == "observe"}
    assert observed == set(range(5))
    assert not any(e[0] == "end" for e in arm.events)


def test_plain_exception_mid_batch_does_not_stop_the_rest_of_that_batch(tmp_path):
    """A distinct case from the brief's own (single-batch, error at the edge): put the failure
    in the middle of a batch and confirm every OTHER problem in that same batch still completes
    (observe fires for all of them), across two full batches."""
    arm = RecordingArm()

    def flaky_in_the_middle(problem_idx, problem, runtime):
        if problem_idx in (2, 11):
            raise RuntimeError(f"boom at {problem_idx}")
        return fake_run_problem(problem_idx, problem, runtime)

    summary = run_stream(problems=problems(16), arm=arm, runtime_factory=runtime_factory, run_problem_fn=flaky_in_the_middle, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=4)

    assert summary == {"n_problems": 16, "n_errors": 2}
    observed = {e[1] for e in arm.events if e[0] == "observe"}
    assert observed == set(range(16))


def test_end_batch_raising_does_not_abort_the_stream(tmp_path):
    """A misbehaving arm whose end_batch() itself raises must not take down the rest of the
    adaptation stream -- the assignment requires skill-update failures to degrade gracefully,
    never break the underlying baseline."""

    class ExplodingEndBatchArm(RecordingArm):
        def end_batch(self, batch_idx):
            super().end_batch(batch_idx)
            raise RuntimeError("harness apply exploded")

    arm = ExplodingEndBatchArm()
    summary = run_stream(problems=problems(16), arm=arm, runtime_factory=runtime_factory, run_problem_fn=fake_run_problem, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=4)

    assert summary == {"n_problems": 16, "n_errors": 0}
    assert [e for e in arm.events if e[0] == "end"] == [("end", 0), ("end", 1)]


def test_zero_problems_below_one_batch_size_yields_an_empty_summary(tmp_path):
    """batch_size=8 with only 5 problems -- nothing meets the minimum, n_problems must be 0, and
    the arm must never even see begin_batch()."""
    arm = RecordingArm()
    summary = run_stream(problems=problems(5), arm=arm, runtime_factory=runtime_factory, run_problem_fn=fake_run_problem, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=4)
    assert summary == {"n_problems": 0, "n_errors": 0}
    assert arm.events == []


def test_extract_result_with_a_single_round_uses_it_for_both_metrics():
    single_round_payload = {
        "step_outputs": [
            {"role": "policy", "evolving_round": 0, "step": 0, "policy_answer_correct": 1, "policy_actions": ["<answer>3</answer>"], "policy_answer": "3", "tool_events": []},
            {"role": "verifier", "evolving_round": 0, "step": 0, "policy_actions": ["<report>Correct.</report>"]},
        ]
    }
    result = extract_result(single_round_payload)
    assert result["pass1_round0"] == 1
    assert result["pass_final"] == 1
    assert result["round_count"] == 1


def test_extract_result_on_empty_step_outputs_is_all_zero_not_an_exception():
    result = extract_result({"step_outputs": []})
    assert result == {
        "pass1_round0": 0,
        "pass_final": 0,
        "final_answer_given": "",
        "verifier_feedback": "",
        "tool_errors": "",
        "reasoning_excerpt": "",
        "round_count": 0,
    }


# --- real-schema adversarial cases (fix round 1) ---------------------------------------------
#
# These attack the actual bug the coordinator found: an earlier version of `extract_result` was
# written against a schema (`round`/`policy_action`/`verifier_report`/`tool_errors` keys) that
# does not exist in the real `run_problem` output. Wired against a real payload, that version
# silently returned the all-zero result for every problem, forever, with no error raised.


def test_extract_result_on_a_realistic_payload_gets_every_field_right():
    result = extract_result(REALISTIC_PAYLOAD)
    assert result["pass1_round0"] == 0  # round 0's own (wrong) attempt
    assert result["pass_final"] == 1  # round 1's concluding attempt
    assert result["round_count"] == 2  # two distinct evolving_round values: 0 and 1
    assert result["final_answer_given"] == "7"  # round 1's policy_answer, not round 0's
    assert "NameError: name 'x' is not defined" in result["tool_errors"]
    assert "Looks right." in result["verifier_feedback"]


def test_extract_result_never_leaks_the_ground_truth_sentinel():
    """Every ground-truth-carrying position in REALISTIC_PAYLOAD (payload-level and per-step
    `ground_truth`/`gt_traj`/`data_source`, plus every noise field: `observation`,
    `next_observation`, `infos`, `policy_memory`, `verifier_memory`, and the `informalmath_verify`
    tool's own `stdout`) is filled with the same unique sentinel. If any of the seven returned
    fields ever contained it, that would prove a whitelist violation -- a `dict`/`**` passthrough
    somewhere, or `tool_errors` reading a tool's `stdout` instead of just `stderr`/`run_status`."""
    result = extract_result(REALISTIC_PAYLOAD)
    for field_name, value in result.items():
        assert SENTINEL not in str(value), f"{field_name} leaked the ground-truth sentinel: {value!r}"


def test_extract_result_does_not_confuse_policy_and_verifier_rows():
    """role=="verifier" text must never end up in a policy-only field, and vice versa."""
    result = extract_result(REALISTIC_PAYLOAD)
    assert "<report>" not in result["final_answer_given"]
    assert "<report>" not in result["reasoning_excerpt"]
    assert "<answer>" not in result["verifier_feedback"]


def test_extract_result_tool_errors_excludes_the_verify_tools_stdout():
    """The `informalmath_verify` tool_event in REALISTIC_PAYLOAD's round 1 carries a real
    GT-matching leak line ('Matches GT: False') AND the sentinel inside its `tool_payload["stdout"]`
    -- neither may reach `tool_errors`, which must contain only the sibling `python_code` tool's
    genuine stderr."""
    result = extract_result(REALISTIC_PAYLOAD)
    assert "Matches GT" not in result["tool_errors"]
    assert "informalmath_verify" not in result["tool_errors"]
    assert result["tool_errors"].strip() == "NameError: name 'x' is not defined"
