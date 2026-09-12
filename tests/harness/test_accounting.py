import threading

import pytest

import alphaapollo.core.generation.evolving.utils.agent as agent_module
from alphaapollo.core.generation.evolving.utils.agent import Agent
from alphaapollo.core.harness.accounting import CallAccountant, install_accounting, role_scope


class FakeOpenAIClient:
    """Stand-in for `openai.OpenAI` used only so `Agent.__init__` has something to call.

    `Agent.__init__` (utils/agent.py:27) unconditionally constructs a real `openai.OpenAI(...)`,
    which eagerly builds an httpx transport and resolves this machine's proxy environment
    variables (HTTP_PROXY/ALL_PROXY/...) at construction time -- on a machine with
    `ALL_PROXY=socks5://...` set and the optional `socksio` extra not installed, that raises
    `ImportError` before a single test assertion runs, and does so regardless of anything this
    test file does afterward. Monkeypatching the `OpenAI` name inside the `agent` module (rather
    than reaching for `Agent.__new__` and hand-setting attributes) keeps every test routed
    through the real `Agent.__init__` -- so its config-reading logic (api_key resolution from
    `vllm_config`/`OPENAI_API_KEY`, model_name/temperature/max_tokens/system_prompt defaults) is
    still exercised -- while never touching a socket, a proxy setting, or any other piece of the
    network stack. The instance's `.chat.completions` is immediately replaced by the `agent`
    fixture below with `FakeCompletions`; this class only needs to survive construction.
    """

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs


class FakeUsage:
    prompt_tokens, completion_tokens = 11, 5


class FakeMessage:
    content = "  an answer  "


class FakeResponse:
    usage = FakeUsage()
    choices = [type("C", (), {"message": FakeMessage()})()]


class FakeCompletions:
    def __init__(self):
        self.kwargs_seen = []

    def create(self, **kwargs):
        self.kwargs_seen.append(kwargs)
        return FakeResponse()


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAIClient)

    a = Agent({"model_name": "m", "base_url": "http://x/v1", "api_key": "EMPTY"})
    completions = FakeCompletions()
    a.client = type("C", (), {"chat": type("Ch", (), {"completions": completions})()})()
    a._completions = completions
    return a


def test_calls_are_attributed_to_the_active_role(agent):
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        with role_scope("solver"):
            agent.get_action_from_gpt("q")
        with role_scope("reflect"):
            agent.get_action_from_gpt("q")
            agent.get_action_from_gpt("q")
    finally:
        uninstall()

    assert acc.calls == {"solver": 1, "reflect": 2}
    assert acc.tokens_in["reflect"] == 22 and acc.tokens_out["solver"] == 5


def test_seed_is_injected_into_every_request(agent):
    acc = CallAccountant()
    uninstall = install_accounting(acc, seed=1234)
    try:
        with role_scope("solver"):
            agent.get_action_from_gpt("q")
    finally:
        uninstall()
    assert agent._completions.kwargs_seen[0]["seed"] == 1234


def test_a_provider_that_rejects_seed_falls_back_instead_of_failing(agent):
    """Not every OpenAI-compatible provider accepts `seed`; a 400 on an unknown
    parameter must not take down the whole run."""
    calls = {"n": 0}
    original_create = agent._completions.create

    def picky_create(**kwargs):
        calls["n"] += 1
        if "seed" in kwargs:
            raise TypeError("Unrecognized request argument supplied: seed")
        return original_create(**kwargs)

    agent._completions.create = picky_create

    acc = CallAccountant()
    uninstall = install_accounting(acc, seed=1234)
    try:
        with role_scope("solver"):
            assert agent.get_action_from_gpt("q") == "an answer"
        with role_scope("solver"):
            agent.get_action_from_gpt("q")
    finally:
        uninstall()

    # first call retries without seed; the second must not retry again
    assert calls["n"] == 3
    assert acc.calls["solver"] == 2


def test_patch_preserves_the_original_return_value(agent):
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        with role_scope("solver"):
            assert agent.get_action_from_gpt("q") == "an answer"
    finally:
        uninstall()


def test_uninstall_restores_the_original_method(agent):
    original = Agent.get_action_from_gpt
    install_accounting(CallAccountant())()
    assert Agent.get_action_from_gpt is original


def test_snapshot_separates_solver_side_from_management_side(agent):
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        for role in ("solver", "summarizer", "aggregator", "reflect", "topic_curator"):
            with role_scope(role):
                agent.get_action_from_gpt("q")
    finally:
        uninstall()

    snap = acc.snapshot()
    assert snap["calls/solver_side_total"] == 3
    assert snap["calls/mgmt_side_total"] == 2
    assert snap["calls/aggregator"] == 1


def test_calls_outside_any_role_scope_are_attributed_to_unscoped(agent):
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        agent.get_action_from_gpt("q")
    finally:
        uninstall()
    assert acc.calls["unscoped"] == 1


# --- Adversarial cases devised beyond the brief -----------------------------------------


def test_adversarial_exception_during_request_does_not_pollute_accounting(agent):
    """A request that fails for a reason unrelated to `seed` must propagate untouched, and
    must leave no trace in the accountant -- a half-recorded call would silently corrupt the
    cost report the three experiment arms are compared on."""

    def boom(**kwargs):
        raise ValueError("upstream 500")

    agent._completions.create = boom

    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        with role_scope("solver"):
            with pytest.raises(ValueError):
                agent.get_action_from_gpt("q")
    finally:
        uninstall()

    assert dict(acc.calls) == {}
    assert dict(acc.tokens_in) == {}
    assert dict(acc.tokens_out) == {}


def test_adversarial_missing_or_none_usage_defaults_to_zero_tokens(agent):
    """Not every OpenAI-compatible backend populates `usage` (or all of its fields); a
    provider quirk here must degrade to a zero count, never crash the call."""

    class NoUsageResponse:
        usage = None
        choices = [type("C", (), {"message": FakeMessage()})()]

    class PartialUsage:
        prompt_tokens = 7
        # completion_tokens is intentionally absent

    class PartialUsageResponse:
        usage = PartialUsage()
        choices = [type("C", (), {"message": FakeMessage()})()]

    responses = [NoUsageResponse(), PartialUsageResponse()]
    agent._completions.create = lambda **kwargs: responses.pop(0)

    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        with role_scope("solver"):
            agent.get_action_from_gpt("q")
            agent.get_action_from_gpt("q")
    finally:
        uninstall()

    assert acc.calls["solver"] == 2
    assert acc.tokens_in["solver"] == 7
    assert acc.tokens_out["solver"] == 0


def test_adversarial_role_labels_do_not_leak_across_threads(agent):
    """role_scope must use a ContextVar, not a module-level global: a batch of problems may
    be solved concurrently, and a shared mutable "current role" would let one thread's tag
    bleed into another's accounting."""
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    roles = ["role_a", "role_b", "role_c", "role_d"]
    iterations = 25

    def worker(role):
        with role_scope(role):
            for _ in range(iterations):
                agent.get_action_from_gpt("q")

    try:
        threads = [threading.Thread(target=worker, args=(r,)) for r in roles]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        uninstall()

    for r in roles:
        assert acc.calls[r] == iterations
    assert sum(dict(acc.calls).values()) == len(roles) * iterations


def test_adversarial_empty_system_prompt_matches_upstream_message_shape(agent):
    """`if self.system_prompt:` in the upstream method treats "" the same as None (no system
    message at all) -- the patch must reproduce that exact truthiness check, not e.g. `is not
    None`, or an agent configured with an empty-string system prompt would silently start
    getting a spurious system message it never got before accounting was installed."""
    acc = CallAccountant()
    uninstall = install_accounting(acc)
    try:
        agent.system_prompt = ""
        with role_scope("solver"):
            agent.get_action_from_gpt("q")
        messages_empty = agent._completions.kwargs_seen[-1]["messages"]

        agent.system_prompt = "You are a careful solver."
        with role_scope("solver"):
            agent.get_action_from_gpt("q")
        messages_nonempty = agent._completions.kwargs_seen[-1]["messages"]
    finally:
        uninstall()

    assert messages_empty == [{"role": "user", "content": "q"}]
    assert messages_nonempty == [
        {"role": "system", "content": "You are a careful solver."},
        {"role": "user", "content": "q"},
    ]


def test_adversarial_double_install_raises_instead_of_silently_corrupting(agent):
    """Stacking two independent patches on the same class method is only safe to unwind in
    LIFO order; an out-of-order uninstall would otherwise leave Agent permanently patched
    with no error at all. Refuse the second install up front instead."""
    acc1 = CallAccountant()
    u1 = install_accounting(acc1)
    try:
        with pytest.raises(RuntimeError):
            install_accounting(CallAccountant())
    finally:
        u1()


def test_adversarial_double_uninstall_is_a_harmless_noop(agent):
    original = Agent.get_action_from_gpt
    uninstall = install_accounting(CallAccountant())
    uninstall()
    uninstall()
    assert Agent.get_action_from_gpt is original
