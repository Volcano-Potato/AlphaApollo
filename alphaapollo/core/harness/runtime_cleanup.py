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
"""Release the per-problem runtime's network resources once its problem is done.

The driver builds a **fresh** ``Agent`` for every problem rather than sharing one, and it has to:
problems inside a batch run concurrently and are deliberately given *different* injected text, so
a single mutable ``Agent.system_prompt`` would race. ``create_runtime_for_problem`` then builds
env managers around those agents, also per problem.

What neither side does is close anything. ``Agent.__init__`` constructs an ``OpenAI`` client --
an httpx connection pool holding keep-alive sockets, and on first use an asyncio event loop with
its own socketpair -- and ``utils/agent.py`` contains no ``close()`` at all. Two agents per
problem across a 144-problem stream is ~288 client objects whose descriptors are never returned.

That is not a slow leak; it killed the first full adaptation run. All three arms died between
problem 104 and 112 with::

    File ".../asyncio/selector_events.py", line 120, in _make_self_pipe
        self._ssock, self._csock = socket.socketpair()
    OSError: [Errno 24] Too many open files

Three hours in, with no held-out phase ever reached. Raising ``ulimit`` only moves the wall.

This module is deliberately defensive rather than precise about what it closes: the runtime dict
is upstream's shape, the objects inside it are third-party, and a cleanup step that raises would
turn a completed problem into a failed one. Anything unclosable is skipped and logged.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _call_close(target: Any, what: str) -> int:
    """Call ``target.close()`` if it exists. Returns 1 if it ran without raising."""
    close = getattr(target, "close", None)
    if not callable(close):
        return 0
    try:
        close()
        return 1
    except Exception:  # noqa: BLE001 -- cleanup must never fail a finished problem
        logger.debug("could not close %s", what, exc_info=True)
        return 0


def _close_agent(agent: Any, what: str) -> int:
    """Close the model client an agent holds, if any."""
    if agent is None:
        return 0
    return _call_close(getattr(agent, "client", None), f"the client on {what}")


def close_runtime(runtime: dict) -> int:
    """Release everything one problem's runtime holds open; return how many things closed.

    Two kinds of resource, for two different reasons:

    **Model clients** (``policy_agent``, ``verifier_agent``) -- the actual leak. Nothing upstream
    closes these, because upstream builds two agents for a whole run while this driver builds two
    per problem. Without this, descriptors grow without bound; see the module docstring.

    **Env managers** (``policy_env_manager``, ``verifier_env_manager``) -- insurance, not a fix.
    Each wraps an ``InformalMathEvolvingMultiProcessEnv`` holding a ThreadPoolExecutor and an
    asyncio event loop, and upstream *does* release those: ``envs.py``'s ``__del__`` calls
    ``close()``. Measured over an 8-minute run, live kqueue descriptors fell 8 -> 2 while
    problems kept completing, so the collector demonstrably keeps up. But `__del__` fires on
    refcount, which is a timing guarantee rather than a structural one, and a five-hour run is a
    poor place to depend on one. ``close()`` guards itself with a ``_closed`` flag
    (``envs.py:185``), so calling it here and again from ``__del__`` is harmless.

    The manager's own ``close()`` is the right level to call: ``EnvironmentManagerBase.close``
    (``base.py:114``) delegates down to the env. Reaching past it into ``.envs`` would couple this
    module to upstream's internals for no gain.

    Every step tolerates absence: a config may disable verification, and the unit-test runtimes
    carry only a stub policy agent.
    """
    if not isinstance(runtime, dict):
        return 0

    closed = _close_agent(runtime.get("policy_agent"), "policy_agent")
    closed += _call_close(runtime.get("policy_env_manager"), "policy_env_manager")

    verifier_configs = runtime.get("verifier_configs")
    if isinstance(verifier_configs, dict):
        closed += _close_agent(verifier_configs.get("verifier_agent"), "verifier_agent")
        closed += _call_close(verifier_configs.get("verifier_env_manager"), "verifier_env_manager")
    return closed
