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
#
# --- task-14 addendum: the first fixture from a REAL execution ------------------------------
#
# Neither `SIMPLIFIED_PAYLOAD` nor `REALISTIC_PAYLOAD` above was ever run against a real model --
# both are hand-written against what the schema was *believed* to look like. That is exactly how
# three bugs slipped past every test in this file until a real qwen3-8b/AIME24 rollout was
# recorded and diffed against `extract_result`'s output:
#
#   1. `role` has a THIRD value neither fixture included: `"verifier_aggregation"`, emitted once
#      per round with a clean `representative_report` verdict string. `verifier_feedback` must
#      prefer it over the raw `role == "verifier"` reply stream, falling back to the latter only
#      when no aggregation entry is present.
#   2. `reasoning_excerpt` was being run through `sanitize_feedback()`, which strips `<think>`
#      blocks -- but AlphaApollo's own policy prompt forces the model's entire solving process
#      into `<think>...</think>` (see `core/environments/prompts/informal_math_evolving.py`), so
#      that stripped every real reasoning trace down to a bare `<answer>...</answer>` tag. Fixed
#      by `_sanitize_reasoning_excerpt()`, which scrubs only the GT-channel line.
#   3. On a problem the policy solved correctly, `final_answer_given` / `reasoning_excerpt`
#      legitimately contain the same string as `ground_truth` (the model's own correct answer
#      equals the correct answer) -- not leakage, since Reflect only ever runs on failures, where
#      the model's answer is by definition different from `ground_truth`. See
#      `test_real_fixture_answer_matching_ground_truth_is_not_leakage` below.
#
# `REAL_PROBLEM_PAYLOAD` (loaded from `fixtures/real_problem_payload.json`) is a trimmed recording
# of qwen3-8b actually solving AIME24 problem 0 end to end, `evolving_round=1` (one round only,
# and it succeeded first try, so `pass1_round0 == pass_final == 1` here -- this fixture does not
# exercise the multi-round-disagreement path, which `REALISTIC_PAYLOAD` above already covers with
# synthetic data). It keeps only the keys `extract_result` actually reads off each `step_outputs`
# entry (plus `role`/`step`/`evolving_round` on every entry), and drops the run's `full_config` /
# `env_config` blocks entirely (configuration noise, unrelated to this module). `ground_truth` /
# `gt_traj` are deliberately still present at the payload's top level -- their presence is exactly
# what makes `test_real_fixture_answer_matching_ground_truth_is_not_leakage` meaningful.

import json
import threading
import time
from pathlib import Path

import pytest

from alphaapollo.core.generation.evolving.evolving_harness_main import (
    FrozenArmWithoutState,
    LeakageError,
    StreamAbort,
    assert_frozen_arm_has_state,
    assert_no_gt_tool_call,
    extract_result,
    run_stream,
)
from alphaapollo.core.harness.accounting import CallAccountant
from alphaapollo.core.harness.arms import BaselineArm, CrossProblemArm, EvoHarnessArm
from alphaapollo.core.harness.schema import CandidateMemory, SkillEdit
from alphaapollo.core.harness.tracker import HarnessTracker

REAL_PROBLEM_PAYLOAD = json.loads((Path(__file__).parent / "fixtures" / "real_problem_payload.json").read_text())

SENTINEL = "SENTINEL_GT_9999"


def problems(n=16):
    return [{"problem_idx": i, "question": f"Q{i}", "ground_truth": str(i), "gt_traj": "", "topic": "number_theory"} for i in range(n)]


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


# --- task-14 fixes: verifier_aggregation, <think>-preserving reasoning_excerpt, real fixture ----
#
# See the addendum near the top of this file for the three findings a real recorded rollout
# exposed. The tests below cover each one directly, plus the first fixture built from a real
# execution rather than hand-written data.


def test_verifier_feedback_prefers_the_aggregation_role_representative_report():
    """Finding 1: when a `role == "verifier_aggregation"` entry is present, `verifier_feedback`
    must come from its `representative_report` -- a clean verdict string -- not from the raw
    `role == "verifier"` entry's own `<think>`/`<python_code>`/`<report>` reply stream, even
    though the latter is textually present in the same payload and comes later in `step_outputs`
    order among non-aggregation entries."""
    payload = {
        "step_outputs": [
            {"role": "policy", "evolving_round": 0, "step": 0, "policy_actions": ["<answer>7</answer>"], "policy_answer": "7", "policy_answer_correct": 1, "tool_events": []},
            {"role": "verifier", "evolving_round": 0, "step": 0, "policy_actions": ["<think>scratch work, python code, etc.</think>\n```python\nprint(1)\n```\n<report>messy raw reply</report>"]},
            {
                "role": "verifier_aggregation",
                "evolving_round": 0,
                "step": 4,
                "majority_judgment": 1,
                "representative_report": "Clean verdict: the solution is correct.",
                "final_verifier_actions": ["<think>this must never leak into verifier_feedback</think><report>Clean verdict: the solution is correct.</report>"],
            },
        ]
    }
    result = extract_result(payload)
    assert result["verifier_feedback"] == "Clean verdict: the solution is correct."
    assert "scratch work" not in result["verifier_feedback"]
    assert "python" not in result["verifier_feedback"]
    assert "must never leak" not in result["verifier_feedback"]


def test_verifier_feedback_falls_back_to_the_raw_verifier_role_when_no_aggregation_entry_exists():
    """Finding 1's other half: a payload with only `role == "verifier"` entries (no
    `"verifier_aggregation"` at all -- the shape both `SIMPLIFIED_PAYLOAD` and `REALISTIC_PAYLOAD`
    already use) must keep using the pre-existing `_last_verifier_text` path. This is the
    behavior every pre-task-14 test above already exercises implicitly; this test names it
    explicitly so the fallback is not accidentally lost in a future refactor."""
    result = extract_result(SIMPLIFIED_PAYLOAD)
    # `_last_verifier_text` returns the verifier's raw action text as-is (tags included) -- only
    # the GT-channel line is scrubbed by `sanitize_feedback`; this is the pre-existing fallback
    # behavior, unchanged by task-14.
    assert "Looks right." in result["verifier_feedback"]
    assert "Matches GT" not in result["verifier_feedback"]


def test_reasoning_excerpt_keeps_the_think_block_but_still_scrubs_the_gt_channel_line():
    """Finding 2: `reasoning_excerpt` must NOT be run through `sanitize_feedback()` (which strips
    `<think>...</think>`, discarding the policy's actual derivation -- AlphaApollo's own prompt
    forces the model's whole reasoning process into that tag). It must still scrub a GT-channel
    line if one appears in the text, even inside the think block itself."""
    payload = {
        "step_outputs": [
            {
                "role": "policy",
                "evolving_round": 0,
                "step": 0,
                "policy_actions": ["<think>Step 1: set up equations.\nMatches ground truth: True\nStep 2: solve for s.</think>\n<answer>42</answer>"],
                "policy_answer": "42",
                "policy_answer_correct": 1,
                "tool_events": [],
            }
        ]
    }
    result = extract_result(payload)
    assert "<think>" in result["reasoning_excerpt"]
    assert "Step 1: set up equations." in result["reasoning_excerpt"]
    assert "Step 2: solve for s." in result["reasoning_excerpt"]
    assert "Matches ground truth" not in result["reasoning_excerpt"]


def test_real_fixture_extracts_every_field_correctly():
    """The first fixture built from an actual qwen3-8b/AIME24 execution (see the addendum near the
    top of this file), not hand-written data. Exercises the `verifier_aggregation`-preferred path
    (Finding 1) and the think-preserving `reasoning_excerpt` (Finding 2) simultaneously, against
    real model output rather than a synthetic stand-in for it."""
    result = extract_result(REAL_PROBLEM_PAYLOAD)
    assert result["pass1_round0"] == 1  # the model got it right on round 0
    assert result["pass_final"] == 1  # only one round ran, so pass_final agrees
    assert result["round_count"] == 1
    assert result["final_answer_given"] == "204"
    # verifier_feedback must come from the clean verifier_aggregation.representative_report, not
    # the raw role=="verifier" entry's <think>+python_code+<report> stream.
    assert result["verifier_feedback"].startswith("The policy agent's solution appears correct.")
    assert "<think>" not in result["verifier_feedback"]
    assert "```python" not in result["verifier_feedback"]
    # reasoning_excerpt must be the policy's real <think> derivation, not collapsed to the bare
    # trailing <answer> tag (the pre-fix bug: sanitize_feedback() stripped the whole think block).
    assert result["reasoning_excerpt"].startswith("<think>")
    assert "quadratic formula" in result["reasoning_excerpt"]
    assert len(result["reasoning_excerpt"]) > 500  # the pre-fix bug left ~35 chars here


def test_real_fixture_answer_matching_ground_truth_is_not_leakage():
    """Finding 3 (not a defect, but worth guarding against a future misreading): on this real
    payload the policy solved the problem correctly, so `final_answer_given` legitimately equals
    `ground_truth` ("204") -- the model's own correct answer necessarily equals the correct
    answer. This is NOT the leakage channel `test_extract_result_never_leaks_the_ground_truth_
    sentinel` (above, against `REALISTIC_PAYLOAD`) guards against -- that test uses a sentinel
    string that could never legitimately be a model's own answer, specifically so it stays
    meaningful. Reflect is only ever invoked on FAILED problems (see `arms.py`), at which point
    the model's answer is by definition different from `ground_truth`, so this coincidence never
    actually reaches a Reflect prompt in practice. Do not "fix" this by scrubbing
    `final_answer_given`/`reasoning_excerpt` against `ground_truth` -- doing so would break the
    sentinel test's premise that a real leak channel, not an answer coincidence, is what is being
    detected."""
    assert REAL_PROBLEM_PAYLOAD["ground_truth"] == "204"
    result = extract_result(REAL_PROBLEM_PAYLOAD)
    assert result["final_answer_given"] == "204"
    # The raw policy reasoning also legitimately contains "204" (the model's own derivation, not
    # a leak) -- checked against the fixture's raw text rather than `reasoning_excerpt` itself,
    # since `_MAX_FIELD_CHARS` truncation happens to clip this particular 2000+-character trace
    # before its concluding "\boxed{204}" line.
    raw_final_round_text = "\n".join(REAL_PROBLEM_PAYLOAD["step_outputs"][0]["policy_actions"])
    assert "204" in raw_final_round_text


# --- first-batch circuit breaker --------------------------------------------------------------


def test_a_first_batch_where_every_problem_dies_aborts_the_run(tmp_path):
    """A stream whose very first batch yields zero trajectories is misconfigured, not flaky, and
    the per-problem degradation that makes the stream survive real flakiness is exactly what
    hides it: the first real end-to-end run lost all four problems to `KeyError: 'data_source'`
    and still finished "successfully", having spent four policy calls plus a full round of
    management calls compiling skills out of four empty trajectories.

    Only the FIRST batch trips the breaker. By the second batch the run has proven the wiring
    works end to end, so a wholly-failed batch there is a provider outage -- precisely the case
    the run is supposed to ride out rather than abandon.
    """
    def always_dies(problem_idx, problem, runtime):
        raise KeyError("data_source")

    with pytest.raises(StreamAbort) as excinfo:
        run_stream(problems=problems(16), arm=BaselineArm(), runtime_factory=runtime_factory, run_problem_fn=always_dies, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=4)
    assert "data_source" in str(excinfo.value)


def test_a_partly_failing_first_batch_does_not_trip_the_breaker(tmp_path):
    """7 of 8 surviving is flakiness; the stream must continue."""
    def mostly_ok(problem_idx, problem, runtime):
        if problem_idx == 0:
            raise RuntimeError("transient")
        return fake_run_problem(problem_idx, problem, runtime)

    summary = run_stream(problems=problems(16), arm=BaselineArm(), runtime_factory=runtime_factory, run_problem_fn=mostly_ok, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=4)
    assert summary == {"n_problems": 16, "n_errors": 1}


def test_a_wholly_failed_later_batch_is_ridden_out(tmp_path):
    """Batch 0 proved the wiring; a total outage in batch 1 is the transient case the
    log-and-continue path exists for."""
    def dies_in_second_batch(problem_idx, problem, runtime):
        if problem_idx >= 8:
            raise RuntimeError("provider outage")
        return fake_run_problem(problem_idx, problem, runtime)

    summary = run_stream(problems=problems(16), arm=BaselineArm(), runtime_factory=runtime_factory, run_problem_fn=dies_in_second_batch, tracker=HarnessTracker(tmp_path, enabled=False), accountant=CallAccountant(), batch_size=8, max_workers=4)
    assert summary == {"n_problems": 16, "n_errors": 8}


class _NoopAgent:
    """Never proposes anything, so end_batch is a no-op and these tests isolate scheduling."""

    def get_action_from_gpt(self, obs):
        return "NO_PROPOSALS"


def test_batch_size_and_max_workers_are_independent_knobs(tmp_path):
    """batch_size is the protocol knob (how often the harness may change); max_workers is the
    throughput knob (how many problems run at once). Conflating them was a live design error:
    "batch 8 x 3 arms = 24 concurrent, over the provider's ceiling of 12" framed a tradeoff
    between paper fidelity and wall clock that does not exist -- batch_size=8 with
    max_workers=4 keeps the paper's update granularity at half the concurrency.

    batch_size=1 is NOT a safe fallback: the general-skill layer exists to find patterns across
    multiple problems, so a one-problem batch can only ever yield topic skills, collapsing the
    two-layer design Task A requires. Hence this property is load-bearing.
    """
    peak = {"n": 0, "now": 0}
    lock = threading.Lock()

    def tracking_run_problem(problem_idx, problem, runtime):
        with lock:
            peak["now"] += 1
            peak["n"] = max(peak["n"], peak["now"])
        time.sleep(0.02)
        with lock:
            peak["now"] -= 1
        return fake_run_problem(problem_idx, problem, runtime)

    arm = EvoHarnessArm(store_root=tmp_path / "store", agent=_NoopAgent())
    summary = run_stream(problems=problems(16), arm=arm, runtime_factory=runtime_factory,
                         run_problem_fn=tracking_run_problem,
                         tracker=HarnessTracker(tmp_path, enabled=False),
                         accountant=CallAccountant(), batch_size=8, max_workers=4)

    assert summary["n_problems"] == 16
    assert peak["n"] <= 4, "max_workers must cap concurrency regardless of batch_size"


def test_a_batch_larger_than_max_workers_still_shares_one_frozen_harness(tmp_path):
    """The protocol guarantee must not quietly depend on batch_size == max_workers: every
    problem in the batch sees the same injected text even when they run in separate waves."""
    arm = EvoHarnessArm(store_root=tmp_path / "store", agent=_NoopAgent())
    arm.store.apply(
        [SkillEdit(op="ADD", actor="general_curator", reason="seed",
                   payload=CandidateMemory(trigger="t", lesson="l", failure_mode="a",
                                           scope_hint="general", topic=None, evidence=["p_0"]))],
        problem_idx=0, batch=0, question_texts=[], ground_truths=[])

    seen = []

    def recording_run_problem(problem_idx, problem, runtime):
        seen.append(runtime["policy_agent"].system_prompt)
        return fake_run_problem(problem_idx, problem, runtime)

    run_stream(problems=problems(8), arm=arm, runtime_factory=runtime_factory,
               run_problem_fn=recording_run_problem,
               tracker=HarnessTracker(tmp_path, enabled=False),
               accountant=CallAccountant(), batch_size=8, max_workers=3)

    assert len(set(seen)) == 1, "one batch must mean one frozen harness, however it is scheduled"


# --- held-out misconfiguration guard -----------------------------------------------------------


class _StatefulArm(BaselineArm):
    def __init__(self, size):
        self._size = size

    def cross_problem_state_size(self):
        return self._size


def test_a_frozen_arm_with_an_empty_state_is_refused_before_any_problem_runs():
    """Held-out evaluation points a frozen arm at the store or pool the adaptation run built.
    Point it somewhere empty -- a typo, a wrong path, a run that never completed -- and the arm
    injects nothing, silently becoming Baseline. Three identical curves, one wasted held-out
    budget, and nothing in the output says why. This is the same defect the Raw arm had in code,
    reappearing at the configuration layer.
    """
    with pytest.raises(FrozenArmWithoutState) as excinfo:
        assert_frozen_arm_has_state(_StatefulArm(0), arm_name="evo", frozen=True, state_path="/tmp/nope")
    assert "/tmp/nope" in str(excinfo.value), "the message must name the path to be actionable"


def test_a_frozen_arm_carrying_state_is_accepted():
    assert_frozen_arm_has_state(_StatefulArm(7), arm_name="evo", frozen=True, state_path="/tmp/ok")


def test_an_unfrozen_arm_with_an_empty_state_is_the_normal_cold_start():
    """Adaptation starts with nothing. Refusing that would refuse the experiment's first run."""
    assert_frozen_arm_has_state(_StatefulArm(0), arm_name="evo", frozen=False, state_path="/tmp/cold")


def test_a_frozen_baseline_is_not_a_misconfiguration():
    """Baseline has no cross-problem state to load; `frozen` is meaningless for it, and tripping
    the guard would block the held-out baseline run the comparison needs."""
    assert_frozen_arm_has_state(BaselineArm(), arm_name="baseline", frozen=True, state_path=None)


# --- resume, trajectory archival (crash recovery for a multi-hour run) ---------------------


def test_start_batch_skips_the_leading_batches_entirely(tmp_path):
    """A skipped batch must consume no budget and must not touch the arm: the arm's state
    already reflects it, having been reloaded from disk."""
    arm = RecordingArm()
    ran = []

    def counting_run_problem(problem_idx, problem, runtime):
        ran.append(problem_idx)
        return fake_run_problem(problem_idx, problem, runtime)

    summary = run_stream(problems=problems(24), arm=arm, runtime_factory=runtime_factory,
                         run_problem_fn=counting_run_problem,
                         tracker=HarnessTracker(tmp_path, enabled=False),
                         accountant=CallAccountant(), batch_size=8, max_workers=4, start_batch=2)

    assert ran == list(range(16, 24)), "only the third batch's problems may run"
    assert summary["n_problems"] == 8
    assert [e for e in arm.events if e[0] == "begin"] == [("begin", 2)]


def test_the_circuit_breaker_arms_on_the_first_executed_batch_not_literally_batch_zero(tmp_path):
    """A resumed run is a new process with its own credentials and proxy, so it needs the same
    proof-of-life before it is allowed to spend hours writing zeros."""
    def always_dies(problem_idx, problem, runtime):
        raise RuntimeError("provider down")

    with pytest.raises(StreamAbort):
        run_stream(problems=problems(24), arm=BaselineArm(), runtime_factory=runtime_factory,
                   run_problem_fn=always_dies, tracker=HarnessTracker(tmp_path, enabled=False),
                   accountant=CallAccountant(), batch_size=8, max_workers=4, start_batch=2)


def test_on_batch_complete_fires_once_per_batch_in_order(tmp_path):
    done = []
    run_stream(problems=problems(24), arm=RecordingArm(), runtime_factory=runtime_factory,
               run_problem_fn=fake_run_problem, tracker=HarnessTracker(tmp_path, enabled=False),
               accountant=CallAccountant(), batch_size=8, max_workers=4,
               on_batch_complete=done.append)
    assert done == [0, 1, 2]


def test_on_batch_complete_does_not_fire_for_a_batch_whose_end_batch_raised(tmp_path):
    """Advancing past an uncommitted batch would drop its harness update permanently -- a
    resumed run never revisits it."""
    class ExplodingEndBatch(RecordingArm):
        def end_batch(self, batch_idx):
            if batch_idx == 0:
                raise RuntimeError("curator exploded")
            return []

    done = []
    summary = run_stream(problems=problems(16), arm=ExplodingEndBatch(),
                         runtime_factory=runtime_factory, run_problem_fn=fake_run_problem,
                         tracker=HarnessTracker(tmp_path, enabled=False),
                         accountant=CallAccountant(), batch_size=8, max_workers=4,
                         on_batch_complete=done.append)
    assert done == [1], "batch 0 never committed, so the marker may not advance past it"
    assert summary["n_problems"] == 16, "the stream still continues -- only the marker holds back"


def test_trajectory_sink_receives_every_successful_rollout(tmp_path):
    seen = {}
    run_stream(problems=problems(8), arm=BaselineArm(), runtime_factory=runtime_factory,
               run_problem_fn=fake_run_problem, tracker=HarnessTracker(tmp_path, enabled=False),
               accountant=CallAccountant(), batch_size=8, max_workers=4,
               trajectory_sink=lambda idx, payload: seen.__setitem__(idx, payload))
    assert sorted(seen) == list(range(8))
    assert seen[0] is SIMPLIFIED_PAYLOAD


def test_trajectory_sink_is_not_called_for_a_failed_problem(tmp_path):
    """A failed problem has no payload; a sink called with the all-zero stand-in would write a
    file that looks like a real rollout."""
    def dies_on_three(problem_idx, problem, runtime):
        if problem_idx == 3:
            raise RuntimeError("boom")
        return fake_run_problem(problem_idx, problem, runtime)

    seen = []
    run_stream(problems=problems(8), arm=BaselineArm(), runtime_factory=runtime_factory,
               run_problem_fn=dies_on_three, tracker=HarnessTracker(tmp_path, enabled=False),
               accountant=CallAccountant(), batch_size=8, max_workers=4,
               trajectory_sink=lambda idx, payload: seen.append(idx))
    assert 3 not in seen and len(seen) == 7


def test_a_failing_trajectory_sink_never_aborts_the_run(tmp_path):
    """The run's real product is already on disk; an analysis artifact may not take it down."""
    summary = run_stream(problems=problems(8), arm=BaselineArm(), runtime_factory=runtime_factory,
                         run_problem_fn=fake_run_problem,
                         tracker=HarnessTracker(tmp_path, enabled=False),
                         accountant=CallAccountant(), batch_size=8, max_workers=4,
                         trajectory_sink=save_trajectory_that_explodes)
    assert summary["n_problems"] == 8


def save_trajectory_that_explodes(problem_idx, payload):
    from alphaapollo.core.harness.trajectory import save_trajectory
    return save_trajectory("/proc/nonexistent-and-unwritable", problem_idx, payload)


def test_saved_trajectory_round_trips_and_sorts_in_stream_order(tmp_path):
    from alphaapollo.core.harness.trajectory import save_trajectory, trajectory_path
    for idx in (0, 7, 12, 149):
        save_trajectory(tmp_path, idx, SIMPLIFIED_PAYLOAD)
    names = sorted(p.name for p in (tmp_path / "trajectories").iterdir())
    assert names == ["problem_0000.json", "problem_0007.json", "problem_0012.json",
                     "problem_0149.json"], "zero-padded so a plain listing is stream order"
    restored = json.loads(trajectory_path(tmp_path, 7).read_text())
    assert restored["step_outputs"][0]["policy_answer"] == "1"


def test_a_frozen_run_evaluates_every_problem_including_a_short_final_batch(tmp_path):
    """The held-out set is 30 problems at batch_size 8. Dropping the tail loses 6 of them from
    the only set the headline number is computed on."""
    seen = []

    def counting(problem_idx, problem, runtime):
        seen.append(problem_idx)
        return fake_run_problem(problem_idx, problem, runtime)

    summary = run_stream(problems=problems(30), arm=BaselineArm(),
                         runtime_factory=runtime_factory, run_problem_fn=counting,
                         tracker=HarnessTracker(tmp_path, enabled=False),
                         accountant=CallAccountant(), batch_size=8, max_workers=4,
                         drop_last_batch=False)
    assert seen == list(range(30))
    assert summary["n_problems"] == 30


def test_an_adaptation_run_still_drops_its_short_final_batch(tmp_path):
    summary = run_stream(problems=problems(30), arm=BaselineArm(),
                         runtime_factory=runtime_factory, run_problem_fn=fake_run_problem,
                         tracker=HarnessTracker(tmp_path, enabled=False),
                         accountant=CallAccountant(), batch_size=8, max_workers=4)
    assert summary["n_problems"] == 24
