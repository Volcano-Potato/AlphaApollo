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
