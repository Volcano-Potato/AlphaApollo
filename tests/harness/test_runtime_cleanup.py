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
