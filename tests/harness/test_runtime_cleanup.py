"""The leak that killed the first full adaptation run.

A fresh `Agent` per problem means a fresh `OpenAI` client -- an httpx pool plus, on first use, an
asyncio event loop with its own socketpair -- and `utils/agent.py` has no `close()` anywhere. At
two agents per problem over a 144-problem stream, all three arms died between problem 104 and 112
with `OSError: [Errno 24] Too many open files`, three hours in, before the held-out phase ever
started.
"""

import pytest

from alphaapollo.core.harness.runtime_cleanup import close_runtime


class FakeClient:
    def __init__(self, explode=False):
        self.closed = 0
        self.explode = explode

    def close(self):
        if self.explode:
            raise RuntimeError("transport already torn down")
        self.closed += 1


class FakeAgent:
    def __init__(self, explode=False):
        self.client = FakeClient(explode)


def runtime(policy=True, verifier=True, **kw):
    out = {}
    if policy:
        out["policy_agent"] = FakeAgent(**kw)
    if verifier:
        out["verifier_configs"] = {"enabled": True, "verifier_agent": FakeAgent(**kw)}
    return out


def test_both_agents_clients_are_closed():
    rt = runtime()
    assert close_runtime(rt) == 2
    assert rt["policy_agent"].client.closed == 1
    assert rt["verifier_configs"]["verifier_agent"].client.closed == 1


def test_a_runtime_without_a_verifier_is_fine():
    """A config may disable verification; the unit-test runtimes carry only a policy stub."""
    assert close_runtime(runtime(verifier=False)) == 1


def test_a_runtime_with_no_agents_at_all_is_fine():
    assert close_runtime({}) == 0


def test_verifier_configs_present_but_carrying_no_agent():
    assert close_runtime({"verifier_configs": {"enabled": False, "verifier_agent": None}}) == 0


@pytest.mark.parametrize("value", [None, "not-a-dict", 42, []])
def test_a_non_dict_runtime_is_ignored_rather_than_raising(value):
    assert close_runtime(value) == 0


def test_an_agent_without_a_client_is_skipped():
    assert close_runtime({"policy_agent": object()}) == 0


def test_a_client_whose_close_raises_does_not_propagate():
    """Cleanup runs in a `finally` after a problem has already succeeded. Raising here would
    turn a completed problem into a failed one -- and a failed problem is indistinguishable from
    a wrong answer in the results."""
    rt = runtime(explode=True)
    assert close_runtime(rt) == 0, "nothing closed, but nothing raised either"


def test_one_agent_failing_to_close_does_not_stop_the_other():
    rt = {"policy_agent": FakeAgent(explode=True),
          "verifier_configs": {"verifier_agent": FakeAgent()}}
    assert close_runtime(rt) == 1
    assert rt["verifier_configs"]["verifier_agent"].client.closed == 1


def test_a_client_with_a_non_callable_close_attribute_is_skipped():
    agent = FakeAgent()
    agent.client.close = "not callable"
    assert close_runtime({"policy_agent": agent}) == 0


# --- env managers: insurance against __del__ timing, not a demonstrated leak ------------------


class FakeEnvManager:
    """Upstream's manager exposes close() at its own level (base.py:114), which delegates to the
    env. This mirrors that shape -- reaching past it into `.envs` would couple us to internals."""

    def __init__(self, explode=False):
        self.closed = 0
        self.explode = explode

    def close(self):
        if self.explode:
            raise RuntimeError("loop already closed")
        self.closed += 1


def full_runtime(**kw):
    return {
        "policy_agent": FakeAgent(**kw),
        "policy_env_manager": FakeEnvManager(**kw),
        "verifier_configs": {
            "enabled": True,
            "verifier_agent": FakeAgent(**kw),
            "verifier_env_manager": FakeEnvManager(**kw),
        },
    }


def test_all_four_resources_are_released():
    rt = full_runtime()
    assert close_runtime(rt) == 4
    assert rt["policy_env_manager"].closed == 1
    assert rt["verifier_configs"]["verifier_env_manager"].closed == 1


def test_env_managers_are_released_even_without_agents():
    rt = {"policy_env_manager": FakeEnvManager(),
          "verifier_configs": {"verifier_env_manager": FakeEnvManager()}}
    assert close_runtime(rt) == 2


def test_agents_are_still_released_when_no_env_manager_is_present():
    """The unit-test runtimes carry only a stub policy agent."""
    assert close_runtime(runtime()) == 2


def test_an_env_manager_without_close_is_skipped():
    assert close_runtime({"policy_env_manager": object()}) == 0


def test_an_env_manager_whose_close_raises_does_not_propagate():
    """envs.py's close() shuts down a ThreadPoolExecutor and an event loop; either can already be
    gone. Raising here would turn a completed problem into a failed one."""
    rt = {"policy_env_manager": FakeEnvManager(explode=True)}
    assert close_runtime(rt) == 0


def test_one_resource_failing_does_not_stop_the_others():
    rt = full_runtime()
    rt["policy_env_manager"].explode = True
    assert close_runtime(rt) == 3
    assert rt["verifier_configs"]["verifier_env_manager"].closed == 1
    assert rt["policy_agent"].client.closed == 1


def test_closing_twice_is_safe():
    """close() is called here and again from envs.py's __del__; upstream guards it with a
    _closed flag (envs.py:185), so this must not be a problem."""
    rt = full_runtime()
    assert close_runtime(rt) == 4
    assert close_runtime(rt) == 4, "idempotent from this module's side too"
    assert rt["policy_env_manager"].closed == 2
