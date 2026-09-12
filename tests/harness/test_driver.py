import time

import pytest

from alphaapollo.core.generation.evolving.evolving_harness_main import LeakageError, assert_no_gt_tool_call, extract_result, run_stream
from alphaapollo.core.harness.accounting import CallAccountant
from alphaapollo.core.harness.arms import BaselineArm, CrossProblemArm
from alphaapollo.core.harness.tracker import HarnessTracker


def problems(n=16):
    return [{"problem_idx": i, "question": f"Q{i}", "ground_truth": str(i), "gt_traj": "", "topic": "number_theory", "problem_shape": "shape"} for i in range(n)]


PAYLOAD = {
    "step_outputs": [
        {"round": 0, "policy_answer_correct": 0, "policy_action": "<answer>1</answer>", "verifier_report": "Wrong modulus.\nMatches GT: False", "tool_errors": "NameError: x"},
        {"round": 1, "policy_answer_correct": 1, "policy_action": "<answer>7</answer>", "verifier_report": "Looks right.", "tool_errors": ""},
    ]
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
    return {"problem_idx": problem_idx, "problem_payload": PAYLOAD, "system_prompt_seen": runtime["policy_agent"].system_prompt}


def runtime_factory(system_prompt):
    agent = type("A", (), {"system_prompt": system_prompt})()
    return {"policy_agent": agent}


def test_extract_result_uses_round0_and_final_not_success_rate():
    result = extract_result(PAYLOAD)
    assert result["pass1_round0"] == 0
    assert result["pass_final"] == 1
    assert result["round_count"] == 2


def test_extract_result_sanitizes_the_gt_channel():
    assert "Matches GT" not in extract_result(PAYLOAD)["verifier_feedback"]


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
                    {"policy_answer_correct": 0, "policy_action": "<informalmath_verify>x</informalmath_verify>"},
                ]
            }
        else:
            payload = PAYLOAD
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
            {"policy_answer_correct": 1, "policy_action": "<answer>3</answer>", "verifier_report": "Correct.", "tool_errors": ""},
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
