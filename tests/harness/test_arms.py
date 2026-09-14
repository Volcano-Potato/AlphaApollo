import json
import re

import pytest

import alphaapollo.core.generation.evolving.utils.agent as agent_module
from alphaapollo.core.generation.evolving.utils.agent import Agent
from alphaapollo.core.harness.accounting import CallAccountant, install_accounting
from alphaapollo.core.harness.arms import BaselineArm, EvoHarnessArm, RawExperienceArm, build_arm
from alphaapollo.core.harness.render import NEUTRAL_SYSTEM_PROMPT
from alphaapollo.core.harness.schema import Skill

PROBLEM = {"problem_idx": 0, "question": "Count integers divisible by seven.",
           "topic": "number_theory"}

FAILED = {"pass_final": 0, "pass1_round0": 0, "final_answer_given": "412",
          "verifier_feedback": "The modular step is wrong.\nMatches GT: False",
          "tool_errors": "", "reasoning_excerpt": "I assumed uniform residues.",
          "round_count": 3}

PASSED = {**FAILED, "pass_final": 1, "pass1_round0": 1}

REFLECTION = """TOPIC: number theory
SCOPE: topic
TRIGGER: Counting integers under congruence constraints.
LESSON:
- Enumerate a small range in python before generalizing.
AVOID: Extrapolating without numeric verification."""


class ScriptedAgent:
    """Replays a fixed script, except for harness-selection prompts.

    Selection is a model call now (paper Appendix F), so a purely positional script would have
    its Reflect/curator replies eaten by whichever selection call happened to come first. A
    selection prompt is answered by naming every skill id it lists -- "select everything offered"
    -- which keeps these tests about the arm's protocol rather than about selection judgement,
    and leaves the script aligned with the reflect/curate calls it was written for.
    """

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def get_action_from_gpt(self, obs):
        self.prompts.append(obs)
        if "Choose at most" in obs:
            ids = re.findall(r"\bsk_\d+\b", obs)
            return ", ".join(ids) if ids else "NONE"
        return self.replies.pop(0) if self.replies else "NO_PROPOSALS"


@pytest.fixture
def arms(tmp_path):
    return {
        "baseline": BaselineArm(),
        "raw": RawExperienceArm(agent=ScriptedAgent(["A prior attempt failed on modular arithmetic."] * 20)),
        "evo": EvoHarnessArm(store_root=tmp_path / "evo",
                             agent=ScriptedAgent([REFLECTION, "ADD: 1\nREASON: useful", "NO_PATTERNS"])),
    }


def test_every_arm_emits_a_non_empty_system_prompt(arms):
    """agent.py:37 is `if self.system_prompt:` — an empty string deletes the
    system message entirely and makes the arms structurally different."""
    for name, arm in arms.items():
        arm.begin_batch(0)
        prompt = arm.system_prompt_for(PROBLEM)
        assert prompt, f"{name} produced an empty system prompt"


def test_cold_start_arms_all_emit_the_same_neutral_prompt(arms):
    for arm in arms.values():
        arm.begin_batch(0)
        assert arm.system_prompt_for(PROBLEM) == NEUTRAL_SYSTEM_PROMPT


def test_baseline_never_changes_its_prompt(arms):
    arm = arms["baseline"]
    arm.begin_batch(0)
    before = arm.system_prompt_for(PROBLEM)
    arm.observe(PROBLEM, FAILED)
    assert arm.end_batch(0) == []
    arm.begin_batch(1)
    assert arm.system_prompt_for(PROBLEM) == before


def test_evo_arm_prompt_changes_only_after_a_batch_boundary(arms):
    arm = arms["evo"]
    arm.begin_batch(0)
    cold = arm.system_prompt_for(PROBLEM)

    arm.observe(PROBLEM, FAILED)
    assert arm.system_prompt_for(PROBLEM) == cold, "a problem must not affect itself"

    arm.end_batch(0)
    arm.begin_batch(1)
    assert arm.system_prompt_for(PROBLEM) != cold


def test_evo_arm_reflects_only_on_failures(arms):
    arm = arms["evo"]
    arm.begin_batch(0)
    arm.observe(PROBLEM, PASSED)
    assert arm.end_batch(0) == []


def test_evo_arm_strips_the_gt_channel_before_reflecting(arms):
    arm = arms["evo"]
    arm.begin_batch(0)
    arm.observe(PROBLEM, FAILED)
    arm.end_batch(0)
    assert all("Matches GT" not in p for p in arm.agent.prompts)


def test_frozen_arm_does_not_update(tmp_path):
    arm = EvoHarnessArm(store_root=tmp_path / "f",
                        agent=ScriptedAgent([REFLECTION, "ADD: 1\nREASON: x", "NO_PATTERNS"]),
                        frozen=True)
    arm.begin_batch(0)
    arm.observe(PROBLEM, FAILED)
    assert arm.end_batch(0) == []
    assert arm.store.all() == []


def test_raw_experience_stores_successes_too(arms):
    arm = arms["raw"]
    arm.begin_batch(0)
    arm.observe(PROBLEM, PASSED)
    arm.end_batch(0)
    arm.begin_batch(1)
    assert arm.system_prompt_for(PROBLEM) != NEUTRAL_SYSTEM_PROMPT


def test_raw_experience_respects_the_same_token_budget_as_evo(arms):
    from alphaapollo.core.harness.render import count_tokens
    arm = arms["raw"]
    for batch in range(3):
        arm.begin_batch(batch)
        for _ in range(8):
            arm.observe(PROBLEM, FAILED)
        arm.end_batch(batch)
    arm.begin_batch(3)
    assert count_tokens(arm.system_prompt_for(PROBLEM)) <= 800 + count_tokens(NEUTRAL_SYSTEM_PROMPT)


def test_build_arm_dispatches_by_name(tmp_path):
    assert isinstance(build_arm("baseline"), BaselineArm)
    assert isinstance(build_arm("evo", store_root=tmp_path, agent=ScriptedAgent([])), EvoHarnessArm)
    with pytest.raises(ValueError):
        build_arm("nonexistent")


# ---------------------------------------------------------------------------
# Self-authored adversarial cases (brief requires >=3; six shipped, one per
# suggested attack direction).
# ---------------------------------------------------------------------------


def test_mixed_batch_evo_only_reflects_on_the_failure(arms):
    """A batch with both a passed and a failed problem must yield exactly one Reflect call
    plus the two curator calls (3 agent calls total) -- never a fourth call for the pass."""
    arm = arms["evo"]
    other = {**PROBLEM, "problem_idx": 1}
    arm.begin_batch(0)
    arm.observe(PROBLEM, PASSED)
    arm.observe(other, FAILED)
    results = arm.end_batch(0)

    assert len(arm.agent.prompts) == 3, "expected Reflect + TopicCurator + GeneralCurator, not more"
    accepted = [r for r in results if r["accepted"]]
    assert len(accepted) == 1
    assert len(arm.store.all()) == 1
    # The GeneralCurator answered NO_PATTERNS, so the candidate it was shown is recorded as a
    # rejection rather than silently dropped -- the provenance guarantee in _bind_payloads.
    assert [r["reject_reason"] for r in results if not r["accepted"]] == ["skipped"]


def test_mixed_batch_raw_experience_stores_both_outcomes(arms):
    """RawExperienceArm's whole point of contrast with Evo is that it learns from
    successes too -- a mixed batch must summarise both problems, not just the failure."""
    arm = arms["raw"]
    other = {**PROBLEM, "problem_idx": 1}
    arm.begin_batch(0)
    arm.observe(PROBLEM, PASSED)
    arm.observe(other, FAILED)
    added = arm.end_batch(0)

    assert len(added) == 2
    assert {e["problem_idx"] for e in added} == {0, 1}


def test_store_apply_failure_mid_end_batch_leaves_the_harness_unchanged(tmp_path):
    """If store.apply() itself blows up after Reflect/curate already ran, the harness must be
    left exactly as it was -- not half-updated, and end_batch must degrade to []."""
    arm = EvoHarnessArm(store_root=tmp_path / "boom",
                        agent=ScriptedAgent([REFLECTION, "ADD: 1\nREASON: useful", "NO_PATTERNS"]))
    arm.begin_batch(0)
    arm.observe(PROBLEM, FAILED)

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    arm.store.apply = boom

    assert arm.end_batch(0) == []
    assert arm.store.all() == []


def test_mutating_the_live_store_after_begin_batch_does_not_change_this_batchs_prompt(tmp_path):
    """Protocol core: begin_batch() must freeze a real snapshot, not a lazy view onto the live
    store. Writing directly to the live store mid-batch (bypassing the arm's own apply path
    entirely) must never leak into an already-open batch's injected text -- only the *next*
    begin_batch() may observe it."""
    arm = EvoHarnessArm(store_root=tmp_path / "evo2", agent=ScriptedAgent([]))
    arm.begin_batch(0)
    cold = arm.system_prompt_for(PROBLEM)

    injected = Skill(id="sk_0099", name="injected", level="general", topic=None,
                     trigger="Count integers divisible by seven.", lesson="- cheat.",
                     failure_mode="n/a")
    arm.store._write_skill(injected)

    assert arm.system_prompt_for(PROBLEM) == cold, "frozen snapshot must ignore a live mutation mid-batch"

    arm.begin_batch(1)
    assert arm.system_prompt_for(PROBLEM) != cold, "the NEXT batch's snapshot must see it"


def test_raw_experience_selection_count_never_exceeds_b():
    """A large raw-experience pool must still be capped at budget.b entries per prompt, not just
    at the token limit -- growth of the pool must never silently raise the injection count."""
    from alphaapollo.core.harness.arms import _select_raw

    arm = RawExperienceArm(agent=ScriptedAgent([f"Note about attempt {i}." for i in range(30)]))
    arm.begin_batch(0)
    for i in range(30):
        arm.observe({**PROBLEM, "problem_idx": i}, FAILED)
    arm.end_batch(0)

    arm.begin_batch(1)
    picked = _select_raw(arm._frozen_pool, PROBLEM["question"], arm.budget)
    assert 0 < len(picked) <= arm.budget.b


def test_consecutive_begin_batch_without_end_batch_discards_pending_observations(arms):
    """Calling begin_batch() again before ending the previous one must not crash, and the
    abandoned batch's observations must not silently leak into the next end_batch()."""
    evo = arms["evo"]
    evo.begin_batch(0)
    evo.observe(PROBLEM, FAILED)
    evo.begin_batch(1)  # no end_batch(0) in between
    assert evo.end_batch(1) == []
    assert evo.store.all() == []

    raw = arms["raw"]
    raw.begin_batch(0)
    raw.observe(PROBLEM, FAILED)
    raw.begin_batch(1)
    assert raw.end_batch(1) == []


def test_record_selection_without_a_prior_system_prompt_for_call_does_not_crash(tmp_path):
    """A problem that was solved without ever calling system_prompt_for() (e.g. a caller bug,
    or a problem skipped by an upstream stage) must still be safely record_selection()-able,
    logging an empty selection rather than raising."""
    arm = EvoHarnessArm(store_root=tmp_path / "sel", agent=ScriptedAgent([]))
    arm.begin_batch(0)
    arm.record_selection(PROBLEM, FAILED)

    log_path = tmp_path / "sel" / "selection_log.jsonl"
    line = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert line["skill_ids"] == []


def test_end_batch_tags_harness_edits_with_the_batch_index_not_a_problem_index(tmp_path):
    """store.apply() takes one `problem_idx` per call, but a batch's accepted edits can be
    drawn from multiple failed problems (see the comment at the `store.apply()` call site in
    EvoHarnessArm.end_batch) -- pins down that the harness log and the resulting skill's
    `created_at` are tagged with the *batch*'s own index (3 here), not the one problem's own
    `problem_idx` (7, deliberately chosen to differ from the batch index so a bug that swaps
    them would be caught)."""
    arm = EvoHarnessArm(store_root=tmp_path / "tag",
                        agent=ScriptedAgent([REFLECTION, "ADD: 1\nREASON: useful", "NO_PATTERNS"]))
    weird_problem = {**PROBLEM, "problem_idx": 7}
    arm.begin_batch(3)
    arm.observe(weird_problem, FAILED)
    results = arm.end_batch(3)

    assert results[0]["problem_idx"] == 3
    assert results[0]["batch"] == 3
    assert arm.store.all()[0].created_at == 3


def test_baseline_and_raw_record_selection_is_a_safe_no_op(arms):
    """Baseline and RawExperience have no skills to record usage for, but record_selection()
    must still be unconditionally callable without raising (brief requirement)."""
    for name in ("baseline", "raw"):
        arm = arms[name]
        arm.begin_batch(0)
        arm.record_selection(PROBLEM, FAILED)


# ---------------------------------------------------------------------------
# Accounting: RawExperience's per-problem summariser call must be counted as
# management-side cost (Task C's "solver calls vs. skill-management calls,
# reported separately" requirement). A ScriptedAgent test double CANNOT catch a
# missing role_scope() here: install_accounting() patches Agent.get_action_from_gpt
# at the *class* level, so a plain stub object that isn't an `Agent` instance never
# goes through the patch at all, and any missing role_scope() would silently pass.
# These tests therefore construct a real `Agent` (with only `OpenAI` itself
# monkeypatched out, exactly as tests/harness/test_accounting.py's own `agent`
# fixture does, for the same proxy/socksio-avoidance reason documented there).
# ---------------------------------------------------------------------------


class FakeOpenAIClient:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs


class SequencedFakeCompletions:
    """Returns each of ``replies`` in order on successive ``.create()`` calls, then repeats
    "NO_PROPOSALS" once exhausted -- mirrors ``ScriptedAgent``'s own fallback (this file's
    stub-agent test double), so a curator call that outruns a short scripted reply list
    degrades to a harmless no-op instead of an ``IndexError``."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.kwargs_seen = []

    def create(self, **kwargs):
        self.kwargs_seen.append(kwargs)
        content = self.replies.pop(0) if self.replies else "NO_PROPOSALS"
        message = type("M", (), {"content": content})()
        usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 5})()
        return type("R", (), {"usage": usage, "choices": [type("C", (), {"message": message})()]})()


@pytest.fixture
def make_real_agent(monkeypatch):
    """Factory for a *real* ``Agent`` instance (only ``OpenAI`` itself monkeypatched out, exactly
    as ``tests/harness/test_accounting.py``'s own ``agent`` fixture does) that returns ``replies``
    in order. Needed because ``install_accounting()`` patches ``Agent.get_action_from_gpt`` at the
    class level -- a plain stub object such as this file's ``ScriptedAgent`` is never an ``Agent``
    instance, so it never goes through the patch at all, and a missing ``role_scope()`` around a
    real call site would silently pass every ``ScriptedAgent``-based test in this file."""
    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAIClient)

    def _make(replies):
        a = Agent({"model_name": "m", "base_url": "http://x/v1", "api_key": "EMPTY"})
        completions = SequencedFakeCompletions(replies)
        a.client = type("C", (), {"chat": type("Ch", (), {"completions": completions})()})()
        return a

    return _make


def test_raw_experience_summariser_calls_are_accounted_as_management_side(make_real_agent, tmp_path):
    """RawExperienceArm's per-problem summariser call is its ENTIRE cross-problem management
    overhead. If it isn't tagged with role_scope("raw_summarizer"), Task C's cost report would
    show RawExperience as having near-zero management overhead while EvoHarness shows several
    calls per batch -- exactly backwards, since Raw's summariser fires once per problem
    (denser) while Evo's reflect/curate fire once per batch."""
    agent = make_real_agent(["A prior attempt failed on modular arithmetic."] * 3)
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        arm = RawExperienceArm(agent=agent)
        arm.begin_batch(0)
        for i in range(3):
            arm.observe({**PROBLEM, "problem_idx": i}, FAILED)
        arm.end_batch(0)
    finally:
        uninstall()

    snap = acc.snapshot()
    assert snap["calls/raw_summarizer"] == 3
    assert snap["calls/mgmt_side_total"] == 3
    assert snap.get("calls/solver_side_total", 0) == 0


def test_evo_harness_arm_introduces_no_bare_unaccounted_model_call(make_real_agent, tmp_path):
    """Verifies, rather than assumes, that EvoHarnessArm's own agent calls (reflect() and both
    curators) are all correctly role-scoped internally, and that arms.py itself adds no
    additional bare `get_action_from_gpt` call on this path: every call this batch makes must
    land under a named mgmt role, none under "unscoped", and none under any solver role."""
    agent = make_real_agent([REFLECTION, "ADD: 1\nREASON: useful", "NO_PATTERNS"])
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        arm = EvoHarnessArm(store_root=tmp_path / "acct", agent=agent)
        arm.begin_batch(0)
        arm.observe(PROBLEM, FAILED)
        arm.end_batch(0)
    finally:
        uninstall()

    snap = acc.snapshot()
    assert snap.get("calls/unscoped", 0) == 0
    assert snap["calls/reflect"] == 1
    assert snap["calls/topic_curator"] == 1
    assert snap["calls/general_curator"] == 1
    assert snap["calls/mgmt_side_total"] == 3
    assert snap.get("calls/solver_side_total", 0) == 0


# `_ZERO_RESULT` from evolving_harness_main, reproduced here rather than imported so this test
# keeps failing if the driver's shape and the arms' expectations ever drift apart.
NO_TRAJECTORY = {"pass1_round0": 0, "pass_final": 0, "final_answer_given": "",
                 "verifier_feedback": "", "tool_errors": "", "reasoning_excerpt": "",
                 "round_count": 0}


def test_evo_arm_does_not_compile_a_skill_from_a_problem_that_never_ran(arms):
    """A problem whose `run_problem` raised is recorded as the driver's all-zero result so the
    rest of the stream survives. But "no trajectory" is missing data, not a failure to learn
    from: `pass_final == 0` alone would send an empty context to Reflect, which dutifully invents
    a plausible-sounding skill out of nothing and merges it into the harness.

    This is not hypothetical -- it is what the first real end-to-end run actually did. Every one
    of four problems died on `KeyError: 'data_source'`, and the harness still came back with four
    confidently-worded skills compiled from four empty trajectories.
    """
    arm = arms["evo"]
    arm.begin_batch(0)
    arm.system_prompt_for(PROBLEM)
    arm.observe(PROBLEM, NO_TRAJECTORY)
    assert arm.end_batch(0) == []
    assert arm.store.all() == []
    assert arm.agent.prompts == [], "no LLM call may be spent on an empty trajectory"


def test_raw_arm_does_not_summarize_a_problem_that_never_ran(arms):
    """Same failure, same cost: the raw-experience arm would spend one management call per dead
    problem summarizing an empty string, and pollute its pool with the result."""
    arm = arms["raw"]
    arm.begin_batch(0)
    arm.observe(PROBLEM, NO_TRAJECTORY)
    assert arm.end_batch(0) == []
    assert arm.agent.prompts == []


def test_a_genuine_failure_is_still_reflected_on(arms):
    """The guard must key on "no trajectory", not on "did not pass" -- a real wrong answer is
    exactly the signal this whole mechanism exists to learn from."""
    arm = arms["evo"]
    arm.begin_batch(0)
    arm.system_prompt_for(PROBLEM)
    arm.observe(PROBLEM, FAILED)
    assert arm.end_batch(0) != []
    assert arm.store.all() != []


def test_adversarial_a_round_that_ran_but_produced_no_text_is_still_a_failure(arms):
    """A rollout that completed a round and returned a wrong answer with no salvageable
    reasoning excerpt is degenerate but real; `round_count` is the discriminator, and this must
    not be swept up by the no-trajectory guard."""
    arm = arms["evo"]
    arm.begin_batch(0)
    arm.system_prompt_for(PROBLEM)
    arm.observe(PROBLEM, {**NO_TRAJECTORY, "round_count": 1, "final_answer_given": "9"})
    assert arm.end_batch(0) != []


def test_adversarial_mixed_batch_reflects_only_the_problem_that_ran(arms):
    """One dead problem must not suppress its batch-mates, and must not contribute evidence."""
    arm = arms["evo"]
    arm.begin_batch(0)
    arm.system_prompt_for(PROBLEM)
    arm.observe({**PROBLEM, "problem_idx": 0}, NO_TRAJECTORY)
    arm.observe({**PROBLEM, "problem_idx": 1}, FAILED)
    assert arm.end_batch(0) != []
    evidence = [e for skill in arm.store.all() for e in skill.evidence]
    assert "p_1" in evidence
    assert "p_0" not in evidence


# --- RawExperienceArm parity with EvoHarnessArm -----------------------------------------------
#
# Three asymmetries, all of the same kind: something thought through for the Evo arm and never
# mirrored onto the arm it is compared against. Raw Experience is the assignment's second
# control -- "is simply adding history already enough?" -- so an arm that cannot be frozen,
# cannot persist, or cannot report its own cost does not answer that question.


def test_the_raw_arm_can_be_frozen_for_held_out_evaluation(tmp_path):
    """Held-out evaluation requires ALL THREE arms frozen. EvoHarnessArm had `frozen`; this one
    did not, so a held-out run would keep summarising and growing its pool -- learning on the
    test set."""
    arm = RawExperienceArm(agent=ScriptedAgent(["a summary"] * 5), frozen=True)
    arm.begin_batch(0)
    arm.observe(PROBLEM, FAILED)

    assert arm.end_batch(0) == []
    assert arm.pool == []
    assert arm.agent.prompts == [], "a frozen arm must not spend a management call either"


def test_the_raw_pool_survives_a_fresh_arm_instance(tmp_path):
    """EvoHarnessArm's SkillStore is on disk, so a held-out run in a separate process loads the
    harness adaptation built. The raw pool was memory-only, so the same held-out run would start
    empty and the Raw arm would silently degenerate into Baseline."""
    first = RawExperienceArm(agent=ScriptedAgent(["learned something"]), pool_root=tmp_path / "raw")
    first.begin_batch(0)
    first.observe(PROBLEM, FAILED)
    first.end_batch(0)

    second = RawExperienceArm(agent=ScriptedAgent([]), pool_root=tmp_path / "raw")
    assert [e["summary"] for e in second.pool] == ["learned something"]

    second.begin_batch(1)
    assert "learned something" in second.system_prompt_for(PROBLEM)


def test_a_frozen_raw_arm_still_injects_the_pool_it_loaded(tmp_path):
    """Frozen means "stop learning", not "stop using what was learned" -- otherwise held-out
    evaluation would measure Baseline three times."""
    first = RawExperienceArm(agent=ScriptedAgent(["learned something"]), pool_root=tmp_path / "raw")
    first.begin_batch(0)
    first.observe(PROBLEM, FAILED)
    first.end_batch(0)

    frozen = RawExperienceArm(agent=ScriptedAgent([]), pool_root=tmp_path / "raw", frozen=True)
    frozen.begin_batch(0)
    assert "learned something" in frozen.system_prompt_for(PROBLEM)
    frozen.observe(PROBLEM, FAILED)
    assert frozen.end_batch(0) == []
    assert len(frozen.pool) == 1, "a frozen arm must not append to the pool it loaded"


def test_the_raw_arm_records_what_it_injected_into_each_problem(tmp_path):
    """"Injected-context token cost" is a required result for all three arms -- it is how the
    comparison accounts for overhead. Raw inherited a no-op record_selection, so its injected
    tokens were simply absent from the report."""
    arm = RawExperienceArm(agent=ScriptedAgent(["learned something"]), pool_root=tmp_path / "raw")
    arm.begin_batch(0)
    arm.observe(PROBLEM, FAILED)
    arm.end_batch(0)

    arm.begin_batch(1)
    arm.system_prompt_for(PROBLEM)
    arm.record_selection(PROBLEM, FAILED)

    lines = [json.loads(x) for x in (tmp_path / "raw" / "selection_log.jsonl").read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["problem_idx"] == PROBLEM["problem_idx"]
    assert lines[0]["n_tokens"] > 0
    assert lines[0]["n_selected"] == 1
    assert lines[0]["success"] is False


def test_adversarial_record_selection_without_a_prior_prompt_call_is_safe(tmp_path):
    """The base class documents this: a problem that errored before selection still reaches
    record_selection."""
    arm = RawExperienceArm(agent=ScriptedAgent([]), pool_root=tmp_path / "raw")
    arm.begin_batch(0)
    arm.record_selection(PROBLEM, FAILED)

    lines = [json.loads(x) for x in (tmp_path / "raw" / "selection_log.jsonl").read_text().splitlines()]
    assert lines[0]["n_selected"] == 0 and lines[0]["n_tokens"] == 0


def test_adversarial_an_arm_with_no_pool_root_still_works_in_memory(tmp_path):
    """Persistence is opt-in; the unit tests above and any throwaway run must not need a path."""
    arm = RawExperienceArm(agent=ScriptedAgent(["s"]))
    arm.begin_batch(0)
    arm.observe(PROBLEM, FAILED)
    assert arm.end_batch(0) != []
    arm.record_selection(PROBLEM, FAILED)


def test_adversarial_a_corrupt_pool_line_does_not_destroy_the_rest(tmp_path):
    """A run killed mid-write leaves a truncated final line; losing the whole adaptation pool
    over one partial record would be far worse than skipping it."""
    root = tmp_path / "raw"
    root.mkdir(parents=True)
    (root / "pool.jsonl").write_text(
        json.dumps({"problem_idx": 0, "topic": "t", "batch": 0, "summary": "good", "n_tokens": 2})
        + "\n{\"problem_idx\": 1, \"summ")

    arm = RawExperienceArm(agent=ScriptedAgent([]), pool_root=root)
    assert [e["summary"] for e in arm.pool] == ["good"]


def test_adversarial_frozen_is_available_on_every_arm_that_can_learn(tmp_path):
    """A held-out config sets frozen on each arm uniformly; an arm silently lacking the flag
    would keep learning while the others stopped."""
    import inspect

    for cls in (RawExperienceArm, EvoHarnessArm):
        assert "frozen" in inspect.signature(cls.__init__).parameters, cls.__name__


def test_only_arms_that_can_learn_report_a_cross_problem_state_size(tmp_path):
    """`None` means "this arm has no cross-problem state at all", which is different from "it has
    state and the state is empty". The held-out guard needs to tell those apart: an empty Evo
    store is a misconfiguration, an absent Baseline store is the whole point of Baseline."""
    assert BaselineArm().cross_problem_state_size() is None
    assert RawExperienceArm(agent=ScriptedAgent([])).cross_problem_state_size() == 0
    assert EvoHarnessArm(store_root=tmp_path / "e", agent=ScriptedAgent([])).cross_problem_state_size() == 0


def test_cross_problem_state_size_counts_what_was_loaded(tmp_path):
    first = RawExperienceArm(agent=ScriptedAgent(["learned"]), pool_root=tmp_path / "raw")
    first.begin_batch(0)
    first.observe(PROBLEM, FAILED)
    first.end_batch(0)

    assert RawExperienceArm(agent=ScriptedAgent([]), pool_root=tmp_path / "raw").cross_problem_state_size() == 1


# --- proposal and curation are sampled differently, so they are separate agents --------------


def test_reflect_and_the_curators_use_different_agents_when_one_is_supplied(tmp_path):
    """`Agent` fixes its temperature at construction, so "propose at 0.3, curate at 0.0" (what
    the reference implementation does) can only be expressed as two instances. Curating at the
    solver's 0.7 would mean the same candidate draws a different verdict each time it is seen.
    """
    proposer = ScriptedAgent([REFLECTION])
    curator = ScriptedAgent(["ADD: 1\nREASON: generalises", "NO_PATTERNS"])
    arm = EvoHarnessArm(store_root=tmp_path / "split", agent=proposer, curator_agent=curator)

    arm.begin_batch(0)
    arm.observe(PROBLEM, FAILED)
    arm.end_batch(0)

    assert len(proposer.prompts) == 1, "only Reflect may reach the proposing agent"
    assert len(curator.prompts) == 2, "both curators must reach the curating agent"
    assert arm.store.all(), "the split must not break the pipeline it splits"


def test_curator_agent_defaults_to_the_proposing_agent(tmp_path):
    """A single-model setup, and every test written before the split, must keep working."""
    agent = ScriptedAgent([REFLECTION, "ADD: 1\nREASON: x", "NO_PATTERNS"])
    arm = EvoHarnessArm(store_root=tmp_path / "one", agent=agent)
    assert arm.curator_agent is agent
