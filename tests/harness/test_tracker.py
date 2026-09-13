import json
import sys
import types

import numpy as np

from alphaapollo.core.harness.accounting import CallAccountant
from alphaapollo.core.harness.schema import Skill
from alphaapollo.core.harness.store import SkillStore
from alphaapollo.core.harness.tracker import HarnessTracker


def test_metrics_are_written_to_jsonl_without_wandb(tmp_path):
    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log(step=3, metrics={"adapt/pass_final": 1})
    tracker.finish()

    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["step"] == 3 and row["adapt/pass_final"] == 1


def test_harness_state_reports_counts_and_tokens(tmp_path):
    store = SkillStore(tmp_path / "store")
    store._write_skill(Skill(id="sk_0001", name="g-1", level="general", topic=None,
                             trigger="T.", lesson="- L.", failure_mode="A."))
    store._write_skill(Skill(id="sk_0002", name="t-2", level="topic", topic="number_theory",
                             trigger="T.", lesson="- L.", failure_mode="A."))

    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log_harness_state(step=1, store=store)
    tracker.finish()

    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["harness/n_general"] == 1
    assert row["harness/n_topic_total"] == 1
    assert row["harness/total_tokens"] > 0


def test_accounting_separates_solver_side_from_management_side(tmp_path):
    acc = CallAccountant()
    acc.record("solver", 10, 5)
    acc.record("reflect", 7, 3)

    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log_accounting(step=1, accountant=acc)
    tracker.finish()

    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["calls/solver_side_total"] == 1 and row["calls/mgmt_side_total"] == 1


def test_tracker_degrades_silently_when_wandb_is_unavailable(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no wandb here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_wandb)

    tracker = HarnessTracker(tmp_path, enabled=True, project="p", run_name="r")
    tracker.log(step=1, metrics={"x": 1})
    tracker.finish()
    assert tracker.wandb_run is None
    assert (tmp_path / "metrics.jsonl").exists()


def test_every_log_call_appends_one_line(tmp_path):
    tracker = HarnessTracker(tmp_path, enabled=False)
    for step in range(4):
        tracker.log(step=step, metrics={"x": step})
    tracker.finish()
    assert len((tmp_path / "metrics.jsonl").read_text().strip().splitlines()) == 4


# --- Self-designed adversarial cases -----------------------------------------------------
# These probe the decision points the brief flags as "don't guess": empty-harness division,
# a topic bucket with zero skills, wandb succeeding at init but failing later, duplicate steps,
# non-JSON-serializable metric values, unusual skill ids, and post-finish/repeat-finish calls.


def test_harness_state_on_empty_store_does_not_divide_by_zero(tmp_path):
    """An empty harness (e.g. before the first adaptation batch) must report zeroed-out
    scale metrics, never raise ZeroDivisionError on mean_skill_tokens."""
    store = SkillStore(tmp_path / "store")

    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log_harness_state(step=0, store=store)
    tracker.finish()

    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["harness/n_general"] == 0
    assert row["harness/n_topic_total"] == 0
    assert row["harness/total_tokens"] == 0
    assert row["harness/mean_skill_tokens"] == 0


def test_topic_with_no_skills_is_not_fabricated_as_a_zero_entry(tmp_path):
    """The store has no notion of "the set of all possible topics" -- only skills that
    actually exist. A topic that currently holds zero skills must simply be absent from the
    logged row, not synthesized as harness/n_topic/<topic> = 0 (which would require the
    tracker to hardcode a topic list it has no business knowing about)."""
    store = SkillStore(tmp_path / "store")
    store._write_skill(Skill(id="sk_0001", name="t-1", level="topic", topic="algebra",
                             trigger="T.", lesson="- L.", failure_mode="A."))

    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log_harness_state(step=0, store=store)
    tracker.finish()

    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["harness/n_topic/algebra"] == 1
    assert "harness/n_topic/number_theory" not in row


def test_wandb_log_failure_after_successful_init_does_not_raise(tmp_path, monkeypatch):
    """wandb.init() can succeed and wandb.log() can still blow up mid-run (dropped
    connection, quota, etc.). That must never propagate out of .log() and abort the
    experiment -- the jsonl copy must still be written."""
    fake_wandb = types.ModuleType("wandb")
    calls = {"log": 0}

    class FakeRun:
        def log(self, metrics, step=None):
            calls["log"] += 1
            raise RuntimeError("simulated wandb outage")

        def finish(self):
            pass

    fake_wandb.init = lambda **kwargs: FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    # This test is about log() failing mid-run, so it needs init to actually be reached; without
    # a credential the tracker now short-circuits before init (see the no-credentials tests).
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")

    tracker = HarnessTracker(tmp_path, enabled=True, project="p", run_name="r")
    assert tracker.wandb_run is not None

    tracker.log(step=1, metrics={"x": 1})  # must not raise despite FakeRun.log() blowing up
    tracker.finish()

    assert calls["log"] == 1
    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["x"] == 1


def test_logging_the_same_step_twice_appends_two_separate_lines(tmp_path):
    """log() is an append-only writer, not an upsert keyed by step -- two calls with the
    same step number must produce two jsonl rows, not overwrite or merge."""
    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log(step=5, metrics={"x": 1})
    tracker.log(step=5, metrics={"x": 2})
    tracker.finish()

    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert rows[0]["x"] == 1
    assert rows[1]["x"] == 2


def test_metrics_with_numpy_scalars_are_json_serializable(tmp_path):
    """Real accuracy/utility metrics computed with numpy will hand log() numpy scalar
    types (np.float64/np.int64), which plain json.dumps() rejects. These must round-trip
    as ordinary Python numbers, not crash the writer."""
    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log(step=0, metrics={"score": np.float64(0.5), "count": np.int64(3)})
    tracker.finish()

    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["score"] == 0.5
    assert row["count"] == 3


def test_skill_id_containing_slash_does_not_break_jsonl_logging(tmp_path):
    """Skill ids minted by the store are always sk_%04d (no slash), but the tracker must
    not assume that -- a per-skill metric key built as f"usage/skill_hit/{skill.id}" must
    still produce a valid, flat jsonl row even if id contains "/". (This is a real jsonl
    dict key, not a wandb panel path, so a slash cannot "break" nesting here -- but a naive
    implementation that tried to split on "/" for grouping could still misbehave.)"""

    class _FakeStore:
        def __init__(self, skills):
            self._skills = skills

        def all(self):
            return list(self._skills)

    skill = Skill(id="sk/0001", name="weird", level="general", topic=None,
                  trigger="T.", lesson="- L.", failure_mode="A.", n_tokens=10,
                  n_selected=2, n_selected_success=1)
    store = _FakeStore([skill])

    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log_harness_state(step=0, store=store)
    tracker.finish()

    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["usage/skill_hit/sk/0001"] == 2
    assert row["usage/skill_utility/sk/0001"] == skill.utility()


def test_finish_is_idempotent(tmp_path):
    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.log(step=0, metrics={"x": 1})
    tracker.finish()
    tracker.finish()  # must not raise
    assert tracker.wandb_run is None


def test_log_after_finish_still_writes_jsonl(tmp_path):
    """finish() only tears down the wandb side; the jsonl writer has no persistent handle
    to close, so a stray log() call after finish() must keep working rather than raise."""
    tracker = HarnessTracker(tmp_path, enabled=False)
    tracker.finish()
    tracker.log(step=0, metrics={"x": 1})

    row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert row["x"] == 1


def test_creates_run_dir_if_missing(tmp_path):
    run_dir = tmp_path / "nested" / "run1"
    tracker = HarnessTracker(run_dir, enabled=False)
    tracker.log(step=0, metrics={"x": 1})
    tracker.finish()
    assert (run_dir / "metrics.jsonl").exists()


# --- wandb without credentials -----------------------------------------------------------------
#
# wandb is entirely optional here -- the assignment never asks for it, and metrics.jsonl is the
# real record. But an *enabled* wandb on a machine that has never run `wandb login` is the default
# state of a fresh checkout, and it must not be able to hurt the run.


def test_wandb_enabled_without_credentials_degrades_to_jsonl(tmp_path, monkeypatch):
    """Measured on wandb 0.30 under a real PTY: `wandb.init()` blocks on an interactive API-key
    prompt and then raises `KeyboardInterrupt` -- a BaseException, which `except Exception` does
    NOT catch. The tracker constructor propagated it and killed the process before a single
    metrics.jsonl line was written.
    """
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    import wandb

    def prompting_init(**kwargs):
        raise KeyboardInterrupt  # what wandb actually raises when the prompt gets no input

    monkeypatch.setattr(wandb, "init", prompting_init)

    tracker = HarnessTracker(tmp_path, project="p", enabled=True)
    assert tracker.wandb_run is None

    tracker.log(0, {"adapt/pass_final": 1})
    tracker.finish()
    assert json.loads((tmp_path / "metrics.jsonl").read_text().strip()) == {"step": 0, "adapt/pass_final": 1}


def test_wandb_init_is_not_even_attempted_without_credentials(tmp_path, monkeypatch):
    """Catching the interrupt is not enough: reaching `wandb.init()` at all means blocking on a
    terminal prompt (4s, measured) and dumping a signup banner into the run log. With no
    credentials and no explicit offline mode there is nothing to log in to, so skip it."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr("alphaapollo.core.harness.tracker._netrc_has_wandb", lambda: False)

    called = []
    import wandb
    monkeypatch.setattr(wandb, "init", lambda **kw: called.append(kw))

    tracker = HarnessTracker(tmp_path, project="p", enabled=True)
    assert tracker.wandb_run is None
    assert called == [], "wandb.init must not be reached when there is no way to authenticate"


def test_an_api_key_in_the_environment_does_reach_wandb_init(tmp_path, monkeypatch):
    """The skip must be a credential check, not a blanket disable -- a configured machine still
    gets its wandb run."""
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")

    called = []
    import wandb
    monkeypatch.setattr(wandb, "init", lambda **kw: called.append(kw) or "RUN")

    tracker = HarnessTracker(tmp_path, project="p", enabled=True)
    assert tracker.wandb_run == "RUN"
    assert called and called[0]["project"] == "p"


def test_adversarial_offline_mode_needs_no_credentials(tmp_path, monkeypatch):
    """WANDB_MODE=offline is a legitimate configuration that logs locally and never
    authenticates; a credential check that ignored it would silently disable it."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr("alphaapollo.core.harness.tracker._netrc_has_wandb", lambda: False)

    called = []
    import wandb
    monkeypatch.setattr(wandb, "init", lambda **kw: called.append(kw) or "RUN")

    tracker = HarnessTracker(tmp_path, project="p", enabled=True)
    assert tracker.wandb_run == "RUN"


def test_adversarial_a_keyboard_interrupt_during_finish_does_not_lose_the_run(tmp_path, monkeypatch):
    """finish() runs in the driver's `finally`; an exception there would mask whatever actually
    ended the run."""
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")

    class InterruptingRun:
        def log(self, *a, **k):
            pass

        def finish(self):
            raise KeyboardInterrupt

    import wandb
    monkeypatch.setattr(wandb, "init", lambda **kw: InterruptingRun())

    tracker = HarnessTracker(tmp_path, project="p", enabled=True)
    tracker.finish()
    assert tracker.wandb_run is None


# --- run identity ------------------------------------------------------------------------------
#
# Six runs land in the same wandb project (3 arms x adaptation/held-out). Without an explicit
# name and group they arrive as random wandb nicknames in arbitrary order, and the one thing the
# dashboard exists for -- putting the three arms on the same axes -- becomes manual guesswork.


def test_the_run_name_and_group_reach_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    captured = {}

    import wandb
    monkeypatch.setattr(wandb, "init", lambda **kw: captured.update(kw) or "RUN")

    HarnessTracker(tmp_path, project="p", run_name="adapt-evo", group="adaptation",
                   config={"arm": "evo"}, enabled=True)

    assert captured["name"] == "adapt-evo"
    assert captured["group"] == "adaptation"
    assert captured["project"] == "p"
    assert captured["config"] == {"arm": "evo"}


def test_no_group_is_sent_when_none_is_configured(tmp_path, monkeypatch):
    """An explicit group=None is fine for wandb, but sending the key at all when the caller never
    asked for grouping would be a behavioural change for existing single-run usage."""
    monkeypatch.setenv("WANDB_API_KEY", "deadbeef")
    captured = {}

    import wandb
    monkeypatch.setattr(wandb, "init", lambda **kw: captured.update(kw) or "RUN")

    HarnessTracker(tmp_path, project="p", enabled=True)
    assert "group" not in captured


def test_adversarial_run_identity_does_not_require_wandb_to_be_reachable(tmp_path, monkeypatch):
    """Naming must never become a reason a jsonl-only run fails."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr("alphaapollo.core.harness.tracker._netrc_has_wandb", lambda: False)

    tracker = HarnessTracker(tmp_path, project="p", run_name="adapt-evo", group="adaptation", enabled=True)
    tracker.log(0, {"adapt/pass_final": 1})
    assert tracker.wandb_run is None
    assert (tmp_path / "metrics.jsonl").exists()
