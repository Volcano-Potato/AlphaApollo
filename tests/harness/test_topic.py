import pytest

from alphaapollo.core.harness.accounting import MGMT_ROLES, CallAccountant, install_accounting
from alphaapollo.core.harness.topic import TOPICS, classify_topic, label_stream

QUESTION = "Find the number of positive integers $n \\le 600$ whose value can be uniquely determined."


class ScriptedAgent:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def get_action_from_gpt(self, obs):
        self.prompts.append(obs)
        return self.replies.pop(0) if self.replies else "number_theory"


def test_classify_returns_a_topic_from_the_fixed_vocabulary():
    assert classify_topic(ScriptedAgent("number_theory"), QUESTION) == "number_theory"


def test_the_labeller_only_ever_sees_the_question():
    """The whole defence of model-side labelling is that a topic derived from the question leaks
    nothing -- the solver reads that same question in full. That only holds if the answer cannot
    reach this call, which is enforced by the signature taking a single `question` string rather
    than a problem dict that happens to carry `ground_truth` alongside it.
    """
    import inspect

    params = list(inspect.signature(classify_topic).parameters)
    assert params == ["agent", "question"], params

    agent = ScriptedAgent("algebra")
    classify_topic(agent, QUESTION)
    assert QUESTION in agent.prompts[0]
    for forbidden in ("ground_truth", "gt_traj", "answer is", "Answer:"):
        assert forbidden not in agent.prompts[0]


def test_the_prompt_pins_the_vocabulary_so_labels_are_comparable_across_problems():
    """An unconstrained "what is this about?" yields a new topic string per problem, which makes
    every topic bucket hold exactly one skill and destroys the layer."""
    agent = ScriptedAgent("geometry")
    classify_topic(agent, QUESTION)
    for topic in TOPICS:
        assert topic in agent.prompts[0]


@pytest.mark.parametrize("reply,expected", [
    ("number_theory", "number_theory"),
    ("  Number Theory  ", "number_theory"),
    ("TOPIC: combinatorics", "combinatorics"),
    ("This is clearly geometry.", "geometry"),
    ("<think>hmm, counting</think>\ncombinatorics", "combinatorics"),
])
def test_replies_are_normalised_onto_the_vocabulary(reply, expected):
    assert classify_topic(ScriptedAgent(reply), QUESTION) == expected


@pytest.mark.parametrize("reply", ["", "calculus", "I am not sure", "NONE"])
def test_an_unusable_reply_yields_no_topic_rather_than_a_guess(reply):
    """A wrong topic is worse than no topic: it files the skill in a bucket no related problem
    will ever retrieve from. `None` degrades to general-skills-only retrieval, which is safe."""
    assert classify_topic(ScriptedAgent(reply), QUESTION) is None


def test_a_failing_agent_degrades_to_no_topic():
    class ExplodingAgent:
        def get_action_from_gpt(self, obs):
            raise RuntimeError("API is down")

    assert classify_topic(ExplodingAgent(), QUESTION) is None


def test_labelling_calls_are_accounted_as_management_not_solver(monkeypatch):
    """The assignment requires solver calls and cross-problem-management calls reported
    separately. Labelling exists only because Evo-Harness retrieves by topic, so it is
    management overhead and must not hide inside the solver bucket."""
    assert "offline_labeling" in MGMT_ROLES

    import alphaapollo.core.generation.evolving.utils.agent as agent_module
    from alphaapollo.core.generation.evolving.utils.agent import Agent

    class FakeCompletions:
        def create(self, **kwargs):
            usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 2})()
            message = type("M", (), {"content": "algebra"})()
            return type("R", (), {"usage": usage, "choices": [type("C", (), {"message": message})()]})()

    monkeypatch.setattr(agent_module, "OpenAI", lambda **kw: object())
    agent = Agent({"model_name": "m", "api_key": "EMPTY"})
    agent.client = type("C", (), {"chat": type("Ch", (), {"completions": FakeCompletions()})()})()

    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        classify_topic(agent, QUESTION)
    finally:
        uninstall()

    snap = acc.snapshot()
    assert snap["calls/offline_labeling"] == 1
    assert snap["calls/mgmt_side_total"] == 1
    assert snap["calls/solver_side_total"] == 0


def test_label_stream_fills_every_problem_and_is_deterministic_per_question():
    """Labels are frozen into the stream once and shared by all three arms -- the per-topic
    breakdown is required for Baseline and Raw too, and it is only comparable if all three see
    identical topics. Repeated questions must therefore not be re-labelled."""
    agent = ScriptedAgent("algebra", "geometry")
    problems = [
        {"problem_idx": 0, "question": "Q-A"},
        {"problem_idx": 1, "question": "Q-B"},
        {"problem_idx": 2, "question": "Q-A"},
    ]
    labelled = label_stream(agent, problems)

    assert [p["topic"] for p in labelled] == ["algebra", "geometry", "algebra"]
    assert len(agent.prompts) == 2, "an identical question must be labelled once, not twice"


def test_label_stream_does_not_mutate_its_input():
    agent = ScriptedAgent("algebra")
    problems = [{"problem_idx": 0, "question": "Q-A"}]
    label_stream(agent, problems)
    assert "topic" not in problems[0]


def test_label_stream_keeps_a_topic_that_is_already_present():
    """MathArena's 2025 split ships human `problem_type` labels; those are better than anything
    this labeller produces and must not be overwritten."""
    agent = ScriptedAgent("algebra")
    labelled = label_stream(agent, [{"problem_idx": 0, "question": "Q", "topic": "number_theory"}])

    assert labelled[0]["topic"] == "number_theory"
    assert agent.prompts == [], "no call may be spent re-labelling an already-labelled problem"


def test_adversarial_an_unlabellable_problem_is_kept_with_an_empty_topic():
    """Dropping it would silently shrink the stream and break the "identical problem order across
    all three arms" requirement."""
    agent = ScriptedAgent("calculus")
    labelled = label_stream(agent, [{"problem_idx": 0, "question": "Q"}])

    assert len(labelled) == 1
    assert labelled[0]["topic"] == ""


def test_adversarial_agreement_against_human_labels_is_measurable():
    """Reporting the labeller's agreement with MathArena's human `problem_type` on the held-out
    split is what turns "label quality" from an assumption into a number in the README."""
    from alphaapollo.core.harness.topic import agreement

    predicted = [{"problem_idx": i, "topic": t} for i, t in enumerate(["algebra", "geometry", "algebra", ""])]
    human = [{"problem_idx": i, "topic": t} for i, t in enumerate(["algebra", "geometry", "number_theory", "algebra"])]

    score = agreement(predicted, human)
    assert score["n_compared"] == 4
    assert score["n_agree"] == 2
    assert score["rate"] == pytest.approx(0.5)
