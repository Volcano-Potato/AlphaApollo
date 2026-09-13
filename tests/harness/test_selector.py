import pytest

from alphaapollo.core.harness.accounting import MGMT_ROLES, CallAccountant, install_accounting
from alphaapollo.core.harness.schema import Skill
from alphaapollo.core.harness.selector import SELECT_PROMPT, parse_selection, select_skills
from alphaapollo.core.harness.store import Budget

QUESTION = "Count the positive integers below 600 that are uniquely determined by three floors."


class ScriptedAgent:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def get_action_from_gpt(self, obs):
        self.prompts.append(obs)
        return self.replies.pop(0) if self.replies else "NONE"


def skill(sid, level="topic", topic="number_theory", n_tokens=50):
    return Skill(id=sid, name=f"{level}-{sid}", level=level,
                 topic=None if level == "general" else topic,
                 trigger=f"When {sid} applies.", lesson=f"- do {sid}",
                 failure_mode=f"avoid {sid}", n_tokens=n_tokens)


def test_the_model_picks_the_skills_and_they_come_back_in_its_order():
    """Paper Appendix F: "For harness selection, we use Claude Sonnet 4.5 across all experiments
    to retrieve relevant skills from the current harness before task execution." Selection is a
    model call, not a lexical filter."""
    skills = [skill("sk_0001"), skill("sk_0002"), skill("sk_0003")]
    picked = select_skills(ScriptedAgent("sk_0003, sk_0001"), QUESTION, skills, Budget())
    assert [s.id for s in picked] == ["sk_0003", "sk_0001"]


def test_the_prompt_shows_every_skill_with_its_id_and_trigger():
    skills = [skill("sk_0001"), skill("sk_0002", level="general", topic=None)]
    agent = ScriptedAgent("sk_0001")
    select_skills(agent, QUESTION, skills, Budget())
    prompt = agent.prompts[0]
    assert QUESTION in prompt
    for s in skills:
        assert s.id in prompt and s.trigger in prompt


def test_an_empty_harness_costs_no_model_call():
    agent = ScriptedAgent("sk_0001")
    assert select_skills(agent, QUESTION, [], Budget()) == []
    assert agent.prompts == []


def test_the_budget_is_enforced_on_the_model_s_answer_not_trusted_to_it():
    """Task A requires the injected content to respect a configurable count/token budget. A model
    told about a budget will sometimes ignore it, so the cap is applied deterministically after
    the fact -- that is what makes the requirement testable rather than hoped-for."""
    skills = [skill(f"sk_000{i}") for i in range(1, 6)]
    picked = select_skills(ScriptedAgent("sk_0001, sk_0002, sk_0003, sk_0004, sk_0005"),
                           QUESTION, skills, Budget(b=2, general_max=5, topic_max=5, tokens=10_000))
    assert [s.id for s in picked] == ["sk_0001", "sk_0002"]


def test_the_token_cap_skips_an_oversized_skill_without_stopping_selection():
    """A big skill must not act as a hard stop: a smaller lower-ranked one still fits the
    remaining headroom. Same rule the deterministic selector used."""
    skills = [skill("sk_0001", n_tokens=90), skill("sk_0002", n_tokens=400), skill("sk_0003", n_tokens=5)]
    picked = select_skills(ScriptedAgent("sk_0001, sk_0002, sk_0003"), QUESTION, skills,
                           Budget(b=6, general_max=6, topic_max=6, tokens=100))
    assert [s.id for s in picked] == ["sk_0001", "sk_0003"]


def test_per_level_quotas_still_apply():
    skills = [skill(f"sk_000{i}", level="general", topic=None) for i in range(1, 5)]
    picked = select_skills(ScriptedAgent("sk_0001, sk_0002, sk_0003, sk_0004"), QUESTION, skills,
                           Budget(b=6, general_max=2, topic_max=4, tokens=10_000))
    assert [s.id for s in picked] == ["sk_0001", "sk_0002"]


def test_selection_calls_are_accounted_as_management_not_solver(monkeypatch):
    assert "selector" in MGMT_ROLES

    import alphaapollo.core.generation.evolving.utils.agent as agent_module
    from alphaapollo.core.generation.evolving.utils.agent import Agent

    class FakeCompletions:
        def create(self, **kwargs):
            usage = type("U", (), {"prompt_tokens": 30, "completion_tokens": 4})()
            message = type("M", (), {"content": "sk_0001"})()
            return type("R", (), {"usage": usage, "choices": [type("C", (), {"message": message})()]})()

    monkeypatch.setattr(agent_module, "OpenAI", lambda **kw: object())
    agent = Agent({"model_name": "m", "api_key": "EMPTY"})
    agent.client = type("C", (), {"chat": type("Ch", (), {"completions": FakeCompletions()})()})()

    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        select_skills(agent, QUESTION, [skill("sk_0001")], Budget())
    finally:
        uninstall()

    snap = acc.snapshot()
    assert snap["calls/selector"] == 1
    assert snap["calls/mgmt_side_total"] == 1
    assert snap["calls/solver_side_total"] == 0


def test_a_failing_selector_degrades_to_no_skills_rather_than_breaking_the_problem():
    """Assignment Task B: a skill-mechanism failure must never break the underlying baseline. An
    unselectable harness means the problem runs with the neutral prompt, exactly like Baseline."""
    class ExplodingAgent:
        def get_action_from_gpt(self, obs):
            raise RuntimeError("API is down")

    assert select_skills(ExplodingAgent(), QUESTION, [skill("sk_0001")], Budget()) == []


@pytest.mark.parametrize("reply,expected", [
    ("sk_0001, sk_0002", ["sk_0001", "sk_0002"]),
    ("sk_0001\nsk_0002", ["sk_0001", "sk_0002"]),
    ("SELECTED: sk_0002", ["sk_0002"]),
    ("- sk_0001\n- sk_0002", ["sk_0001", "sk_0002"]),
    ("I would use sk_0002 here.", ["sk_0002"]),
    ("NONE", []),
    ("", []),
])
def test_replies_are_parsed_into_ids(reply, expected):
    assert parse_selection(reply) == expected


def test_adversarial_a_hallucinated_id_is_dropped_not_fabricated():
    """The model can name a skill that does not exist; the result must contain only real skills."""
    picked = select_skills(ScriptedAgent("sk_9999, sk_0001"), QUESTION, [skill("sk_0001")], Budget())
    assert [s.id for s in picked] == ["sk_0001"]


def test_adversarial_a_repeated_id_is_injected_once():
    """A duplicate would spend the count budget twice on the same text."""
    picked = select_skills(ScriptedAgent("sk_0001, sk_0001, sk_0002"), QUESTION,
                           [skill("sk_0001"), skill("sk_0002")], Budget())
    assert [s.id for s in picked] == ["sk_0001", "sk_0002"]


def test_adversarial_a_think_block_naming_decoy_ids_does_not_hijack_the_choice():
    """Same hijack shape already proven for Reflect and the curators: a reasoning model rehearses
    ids while deliberating, and a bare id scan would take the rehearsal over the answer."""
    reply = "<think>\nMaybe sk_0002? No, that is about geometry.\n</think>\nsk_0001"
    picked = select_skills(ScriptedAgent(reply), QUESTION, [skill("sk_0001"), skill("sk_0002")], Budget())
    assert [s.id for s in picked] == ["sk_0001"]


def test_adversarial_selecting_nothing_is_a_legitimate_answer():
    """An empty selection is not a failure: with an irrelevant harness, injecting nothing is the
    correct choice and must not be silently back-filled with "best available"."""
    picked = select_skills(ScriptedAgent("NONE"), QUESTION, [skill("sk_0001")], Budget())
    assert picked == []


def test_adversarial_the_prompt_never_carries_the_answer_only_the_question():
    import inspect

    params = list(inspect.signature(select_skills).parameters)
    assert params[:3] == ["agent", "question", "skills"], params
    assert "ground_truth" not in SELECT_PROMPT
