"""The backfill must reproduce `analysis.py`'s numbers exactly and never invent its own.

The real wandb is never imported. `build_payload` is deliberately pure, and the upload path is
driven through a fake module installed into ``sys.modules`` -- which is not just a convenience for
a checkout without wandb: the assertions that matter (``resume="must"``, the custom step metric,
that a failed ``finish()`` is not recorded as a success) are about *how* the call is made, and a
real client would hide exactly those behind a network.
"""

import json
import sys
import types

import pytest

from alphaapollo.core.harness import analysis
from alphaapollo.core.harness.wandb_backfill import (
    MARKER_NAME,
    PREVIEW_NAME,
    STEP_METRIC,
    Payload,
    add_cross_arm,
    build_payload,
    detect_project,
    find_state_root,
    main,
    push,
    read_run_id,
)


def problem_row(step, r0=0, final=0, error=0, topic="algebra"):
    return {"step": step, "adapt/pass1_round0": r0, "adapt/pass_final": final,
            "adapt/error": error, "topic": topic}


def write_run(tmp_path, name, rows, harness=None, accounting=None, run_id="abc123"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in rows]
    lines += [json.dumps(h) for h in (harness or [])]
    if accounting:
        lines.append(json.dumps(accounting))
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if run_id:
        (d / "wandb_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    return d


def write_selection_log(run_dir, dirname, rows):
    state = run_dir / dirname
    state.mkdir(parents=True, exist_ok=True)
    (state / "selection_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return state


class FakeRun:
    """Records the calls `push` makes, and can be told to fail at any one of them."""

    def __init__(self, fail_on_log=False, fail_on_finish=False):
        self.logged = []
        self.summary = {}
        self.defined = []
        self.finished = None
        self._fail_on_log = fail_on_log
        self._fail_on_finish = fail_on_finish

    def define_metric(self, name, step_metric=None):
        self.defined.append((name, step_metric))

    def log(self, row):
        if self._fail_on_log:
            raise RuntimeError("network died mid-upload")
        self.logged.append(row)

    def finish(self, exit_code=None):
        if self._fail_on_finish:
            raise RuntimeError("sync failed")
        self.finished = exit_code


class FakeTable:
    def __init__(self, columns, data):
        self.columns, self.data = columns, data


def install_fake_wandb(monkeypatch, run=None, init_error=None):
    """Put a fake ``wandb`` in ``sys.modules`` and hand back the run `push` will be given."""
    run = run if run is not None else FakeRun()
    calls = {}

    def init(**kwargs):
        calls.update(kwargs)
        if init_error is not None:
            raise init_error
        return run

    module = types.ModuleType("wandb")
    module.init = init
    module.Table = FakeTable
    monkeypatch.setitem(sys.modules, "wandb", module)
    return run, calls


# --- the curve analysis.py gained for this ----------------------------------------------------


def test_cumulative_curve_is_a_running_mean_over_completed_problems():
    run = analysis.RunData(name="x")
    run.problems = {r["step"]: r for r in
                    [problem_row(0, final=1), problem_row(1), problem_row(2, final=1),
                     problem_row(3, final=1)]}
    curve = analysis.cumulative_curve(run)
    assert [p["problem_idx"] for p in curve] == [0, 1, 2, 3]
    assert [p["cum_pass_final"] for p in curve] == [1.0, 0.5, 2 / 3, 0.75]


def test_cumulative_curve_skips_crashed_problems_without_shifting_the_x_axis():
    """A crashed problem is not a wrong answer, so it must not enter the mean -- but the surviving
    points must stay on their real stream positions, or the curve no longer lines up with the
    per-problem rows the original run already logged."""
    run = analysis.RunData(name="x")
    run.problems = {r["step"]: r for r in
                    [problem_row(0, final=1), problem_row(1, error=1), problem_row(2, final=1)]}
    curve = analysis.cumulative_curve(run)
    assert [p["problem_idx"] for p in curve] == [0, 2]
    assert [p["cum_pass_final"] for p in curve] == [1.0, 1.0]


def test_rolling_window_forgets_the_early_stream_while_the_cumulative_mean_does_not():
    run = analysis.RunData(name="x")
    run.problems = {i: problem_row(i, final=1 if i < 10 else 0) for i in range(20)}
    last = analysis.cumulative_curve(run, window=5)[-1]
    assert last["cum_pass_final"] == 0.5
    assert last["roll_pass_final"] == 0.0
    assert last["n_roll"] == 5


# --- payload ----------------------------------------------------------------------------------


def test_payload_series_matches_the_curve_and_is_keyed_by_problem_index(tmp_path):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1), problem_row(1), problem_row(2)])
    payload = build_payload(d, name="evo", window=5)
    assert [r[STEP_METRIC] for r in payload.series] == [0, 1, 2]
    assert payload.series[0]["analysis/cum_pass_final"] == 1.0
    assert payload.series[-1]["analysis/cum_pass_final"] == pytest.approx(1 / 3)
    # The window is part of the key, so two backfills at different windows cannot silently
    # overwrite each other's curve on the same dashboard.
    assert "analysis/roll5_pass_final" in payload.series[0]


def test_headline_and_per_topic_summary_agree_with_analysis(tmp_path):
    rows = [problem_row(0, final=1, topic="algebra"), problem_row(1, topic="algebra"),
            problem_row(2, final=1, topic="geometry")]
    d = write_run(tmp_path, "adapt-evo", rows)
    payload = build_payload(d, name="evo")
    run = analysis.load_run(d)

    assert payload.summary["analysis/overall/pass_final"] == analysis.overall(run)["pass_final"]
    assert payload.summary["analysis/topic/algebra/pass_final"] == 0.5
    assert payload.summary["analysis/topic/geometry/pass_final"] == 1.0
    assert payload.summary["analysis/topic/algebra/n"] == 2


def test_per_topic_table_carries_n_so_a_one_problem_topic_is_not_read_as_a_result(tmp_path):
    d = write_run(tmp_path, "adapt-evo",
                  [problem_row(0, final=1, topic="geometry"), problem_row(1, topic="algebra")])
    columns, data = build_payload(d, name="evo").tables["analysis/by_topic"]
    assert columns[:2] == ["topic", "n"]
    assert dict((row[0], row[1]) for row in data) == {"geometry": 1, "algebra": 1}


def test_cost_is_split_solver_side_from_management_side(tmp_path):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0), problem_row(1)],
                  accounting={"step": 1, "calls/solver": 20, "calls/reflect": 4,
                              "calls/solver_side_total": 20, "calls/mgmt_side_total": 4,
                              "tokens/solver_in": 900, "tokens/solver_out": 100})
    summary = build_payload(d, name="evo").summary
    assert summary["analysis/cost/calls_solver"] == 20
    assert summary["analysis/cost/calls_mgmt"] == 4
    assert summary["analysis/cost/calls_by_role/reflect"] == 4
    assert summary["analysis/cost/calls_per_problem"] == 12.0


def test_injected_context_and_skill_usage_come_from_the_selection_log(tmp_path):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1), problem_row(1)])
    write_selection_log(d, "store", [
        {"problem_idx": 0, "skill_ids": ["sk_0001"], "n_tokens": 100, "success": True},
        {"problem_idx": 1, "skill_ids": ["sk_0001", "sk_0002"], "n_tokens": 200, "success": False},
    ])
    payload = build_payload(d, name="evo")
    assert payload.summary["analysis/injected/mean_tokens"] == 150.0
    assert payload.summary["analysis/injected/n_with_injection"] == 2
    usage = {row[0]: row[1:3] for row in payload.tables["analysis/skill_usage"][1]}
    assert usage["sk_0001"] == [2, 1]
    assert usage["sk_0002"] == [1, 0]


def test_a_heldout_run_ignores_the_adaptation_rows_it_inherited(tmp_path):
    """`run_experiments.sh` gives a frozen run its state by copying the adaptation arm's whole
    directory, so the selection log it starts from already holds that arm's rows -- and since each
    phase numbers problems from zero, the held-out rows land *on top of* adaptation positions
    rather than after them. This is the real on-disk layout of `heldout-evo/store`.
    """
    d = write_run(tmp_path, "heldout-evo", [problem_row(0, final=1), problem_row(1)])
    write_selection_log(d, "store", [
        # inherited: 4 adaptation rows, two of which collide with the held-out positions
        {"problem_idx": 0, "skill_ids": ["sk_old"], "n_tokens": 1000, "success": True},
        {"problem_idx": 1, "skill_ids": ["sk_old"], "n_tokens": 1000, "success": True},
        {"problem_idx": 2, "skill_ids": ["sk_old"], "n_tokens": 1000, "success": True},
        {"problem_idx": 3, "skill_ids": ["sk_old"], "n_tokens": 1000, "success": True},
        # this run's own, appended afterwards
        {"problem_idx": 0, "skill_ids": ["sk_0001"], "n_tokens": 100, "success": True},
        {"problem_idx": 1, "skill_ids": ["sk_0001"], "n_tokens": 200, "success": False},
    ])
    payload = build_payload(d, name="evo")

    assert payload.summary["analysis/injected/n_problems"] == 2
    assert payload.summary["analysis/injected/mean_tokens"] == 150.0
    assert payload.summary["analysis/injected/max_tokens"] == 200
    # The inherited skill never entered a held-out context and must not appear at all.
    assert [row[0] for row in payload.tables["analysis/skill_usage"][1]] == ["sk_0001"]


def test_a_crashed_problem_still_counts_toward_injected_cost(tmp_path):
    """Its skills were selected and its tokens were paid; scoping to `completed` would quietly
    understate exactly the cost the assignment asks to be reported."""
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1), problem_row(1, error=1)])
    write_selection_log(d, "store", [
        {"problem_idx": 0, "skill_ids": ["sk_0001"], "n_tokens": 100, "success": True},
        {"problem_idx": 1, "skill_ids": ["sk_0001"], "n_tokens": 300, "success": False},
    ])
    summary = build_payload(d, name="evo").summary
    assert summary["analysis/injected/n_problems"] == 2
    assert summary["analysis/injected/mean_tokens"] == 200.0
    # ...while the accuracy summary still reports only the problem that produced a rollout.
    assert summary["analysis/overall/n"] == 1


def test_the_raw_arms_pool_is_found_as_readily_as_the_evo_arms_store(tmp_path):
    evo = write_run(tmp_path, "adapt-evo", [problem_row(0)])
    write_selection_log(evo, "store", [{"problem_idx": 0, "skill_ids": [], "n_tokens": 0}])
    raw = write_run(tmp_path, "adapt-raw", [problem_row(0)])
    write_selection_log(raw, "pool", [{"problem_idx": 0, "skill_ids": ["p_1"], "n_tokens": 300}])
    baseline = write_run(tmp_path, "adapt-baseline", [problem_row(0)])

    assert find_state_root(evo).name == "store"
    assert find_state_root(raw).name == "pool"
    # Baseline holds no cross-problem state at all; that is the arm's definition, not a gap.
    assert find_state_root(baseline) is None
    assert "analysis/injected/mean_tokens" not in build_payload(baseline).summary


def test_a_run_with_no_problem_rows_yields_an_empty_payload_rather_than_raising(tmp_path):
    d = tmp_path / "never-ran"
    d.mkdir()
    payload = build_payload(d)
    assert payload.series == [] and payload.summary == {}


# --- cross-arm --------------------------------------------------------------------------------


def test_the_comparable_rate_is_taken_on_the_problems_every_arm_completed(tmp_path):
    """Arms that lost different problems to the same rate-limit storm have different denominators;
    `analysis/common/*` is what makes wandb's runs table a fair comparison of them."""
    dirs = {
        "baseline": write_run(tmp_path, "adapt-baseline",
                              [problem_row(0, final=1), problem_row(1), problem_row(2, final=1)]),
        "evo": write_run(tmp_path, "adapt-evo",
                         [problem_row(0, final=1), problem_row(1, final=1),
                          problem_row(2, error=1)]),
    }
    payloads = {a: build_payload(d, name=a) for a, d in dirs.items()}
    runs = {a: analysis.load_run(d, a) for a, d in dirs.items()}
    add_cross_arm(payloads, runs, {})

    # Problem 2 is gone from the intersection because the evo arm crashed on it.
    assert payloads["baseline"].summary["analysis/common/n"] == 2
    assert payloads["baseline"].summary["analysis/common/pass_final"] == 0.5
    assert payloads["evo"].summary["analysis/common/pass_final"] == 1.0
    # ...and baseline's own-set rate is left alongside it, unchanged, rather than replaced.
    assert payloads["baseline"].summary["analysis/overall/pass_final"] == pytest.approx(2 / 3)


def test_no_redundant_common_curve_when_no_arm_lost_a_problem(tmp_path):
    dirs = {a: write_run(tmp_path, f"adapt-{a}", [problem_row(0, final=1), problem_row(1)])
            for a in ("baseline", "evo")}
    payloads = {a: build_payload(d, name=a) for a, d in dirs.items()}
    runs = {a: analysis.load_run(d, a) for a, d in dirs.items()}
    add_cross_arm(payloads, runs, {})
    assert all(STEP_METRIC in r and "analysis/common/cum_pass_final" not in r
               for r in payloads["evo"].series)


def test_transfer_cases_are_attached_to_the_evo_run_only(tmp_path):
    baseline = write_run(tmp_path, "adapt-baseline", [problem_row(0), problem_row(1, final=1)])
    evo = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1), problem_row(1)])
    state = write_selection_log(evo, "store", [
        {"problem_idx": 0, "skill_ids": ["sk_0001"], "n_tokens": 90, "success": True},
        {"problem_idx": 1, "skill_ids": ["sk_0002"], "n_tokens": 90, "success": False},
    ])
    dirs = {"baseline": baseline, "evo": evo}
    payloads = {a: build_payload(d, name=a) for a, d in dirs.items()}
    runs = {a: analysis.load_run(d, a) for a, d in dirs.items()}
    add_cross_arm(payloads, runs, {"evo": state})

    assert payloads["evo"].summary["analysis/transfer/n_positive"] == 1
    assert payloads["evo"].summary["analysis/transfer/n_negative"] == 1
    assert "analysis/transfer_positive" in payloads["evo"].tables
    assert not any(k.startswith("analysis/transfer") for k in payloads["baseline"].tables)


def test_cross_arm_is_a_no_op_with_a_single_arm(tmp_path):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    payloads = {"evo": build_payload(d, name="evo")}
    add_cross_arm(payloads, {"evo": analysis.load_run(d, "evo")}, {})
    assert not any(k.startswith("analysis/common") for k in payloads["evo"].summary)


# --- push -------------------------------------------------------------------------------------


def test_dry_run_writes_the_payload_and_never_marks_the_run_as_pushed(tmp_path):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1), problem_row(1)])
    assert push(d, build_payload(d, name="evo"), project="p", dry_run=True) == "dry_run"
    preview = json.loads((d / PREVIEW_NAME).read_text())
    assert preview["summary"]["analysis/overall/pass_final"] == 0.5
    # No marker: a dry run must leave the real push still available.
    assert not (d / MARKER_NAME).exists()


def test_a_successful_push_resumes_the_original_run_and_sends_every_part(tmp_path, monkeypatch):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1), problem_row(1)],
                  run_id="deadbeef")
    write_selection_log(d, "store", [
        {"problem_idx": 0, "skill_ids": ["sk_0001"], "n_tokens": 100, "success": True},
        {"problem_idx": 1, "skill_ids": ["sk_0001"], "n_tokens": 100, "success": False},
    ])
    run, calls = install_fake_wandb(monkeypatch)

    assert push(d, build_payload(d, name="evo"), project="proj", entity="ent") == "pushed"

    # `resume="must"`: "allow" would silently create a fresh empty run under a mistyped id.
    assert calls["id"] == "deadbeef" and calls["resume"] == "must"
    assert calls["project"] == "proj" and calls["entity"] == "ent"
    # The custom x-axis is declared before anything is logged against it.
    assert (STEP_METRIC, None) in run.defined
    assert ("analysis/*", STEP_METRIC) in run.defined
    series = [r for r in run.logged if STEP_METRIC in r]
    assert [r[STEP_METRIC] for r in series] == [0, 1]
    tables = {k: v for r in run.logged for k, v in r.items() if isinstance(v, FakeTable)}
    assert "analysis/by_topic" in tables and "analysis/skill_usage" in tables
    assert run.summary["analysis/overall/pass_final"] == 0.5
    assert "analysis/backfilled_at" in run.summary
    assert run.finished is None  # finish() called with no exit_code, i.e. cleanly


def test_a_repair_push_sends_summary_and_tables_without_re_appending_curves(tmp_path, monkeypatch):
    """The only safe way to correct a scalar on a pushed run: summary keys and tables overwrite by
    key, history does not."""
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1), problem_row(1)])
    run, _ = install_fake_wandb(monkeypatch)

    assert push(d, build_payload(d, name="evo"), project="p", include_series=False) == "pushed"

    assert run.defined == []
    assert not [r for r in run.logged if STEP_METRIC in r]
    assert run.summary["analysis/overall/pass_final"] == 0.5
    assert json.loads((d / MARKER_NAME).read_text())["included_series"] is False


def test_the_marker_records_completion_only_after_finish_returns(tmp_path, monkeypatch):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    install_fake_wandb(monkeypatch)
    push(d, build_payload(d, name="evo"), project="p")
    marker = json.loads((d / MARKER_NAME).read_text())
    assert marker["state"] == "complete"
    assert marker["run_id"] == "abc123" and marker["digest"]


def test_a_failed_upload_is_reported_as_failed_and_blocks_a_plain_retry(tmp_path, monkeypatch):
    """A half-uploaded run is the dangerous case: the remote already holds some appended rows, so
    retrying would stack the rest on top rather than replace them."""
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    install_fake_wandb(monkeypatch, run=FakeRun(fail_on_log=True))
    assert push(d, build_payload(d, name="evo"), project="p") == "failed"
    marker = json.loads((d / MARKER_NAME).read_text())
    assert marker["state"] == "failed"
    # Recorded before the first row went out, so it survives being killed mid-upload.
    assert marker["series_sent"] is True

    install_fake_wandb(monkeypatch)
    assert push(d, build_payload(d, name="evo"), project="p") == "skipped"
    # --force says "I meant it", which cannot make a duplicated curve correct: wandb offers no way
    # to remove the first copy, so this stays refused.
    assert push(d, build_payload(d, name="evo"), project="p", force=True) == "skipped"
    # The repair path is still open, because summary and tables overwrite by key.
    assert push(d, build_payload(d, name="evo"), project="p", force=True,
                include_series=False) == "pushed"


def test_force_cannot_re_append_history_to_a_run_that_already_has_it(tmp_path, monkeypatch):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1), problem_row(1)])
    install_fake_wandb(monkeypatch)
    assert push(d, build_payload(d, name="evo"), project="p") == "pushed"

    run, _ = install_fake_wandb(monkeypatch)
    assert push(d, build_payload(d, name="evo"), project="p", force=True) == "skipped"
    assert run.logged == []


def test_a_repair_does_not_clear_the_record_that_history_was_sent(tmp_path, monkeypatch):
    """`--skip_series` sends no history of its own; if it reset the flag it would hand the next
    --force a licence to duplicate exactly the curves it was careful not to touch."""
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1), problem_row(1)])
    install_fake_wandb(monkeypatch)
    push(d, build_payload(d, name="evo"), project="p")

    install_fake_wandb(monkeypatch)
    assert push(d, build_payload(d, name="evo"), project="p", force=True,
                include_series=False) == "pushed"
    assert json.loads((d / MARKER_NAME).read_text())["series_sent"] is True

    install_fake_wandb(monkeypatch)
    assert push(d, build_payload(d, name="evo"), project="p", force=True) == "skipped"


def test_a_run_whose_history_never_left_the_machine_may_still_get_its_curves(tmp_path,
                                                                            monkeypatch):
    """The refusal is about duplicating history, not about having failed once. A push that died
    before sending a single row must not lock the run out of ever getting one."""
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    install_fake_wandb(monkeypatch, init_error=RuntimeError("401"))
    assert push(d, build_payload(d, name="evo"), project="p") == "failed"
    assert json.loads((d / MARKER_NAME).read_text())["series_sent"] is False

    run, _ = install_fake_wandb(monkeypatch)
    assert push(d, build_payload(d, name="evo"), project="p", force=True) == "pushed"
    assert [r[STEP_METRIC] for r in run.logged if STEP_METRIC in r] == [0]


def test_a_marker_from_before_the_flag_existed_is_read_conservatively(tmp_path, monkeypatch):
    """Being wrong in this direction costs a refusal the operator can resolve by looking at the
    run; in the other direction it costs a permanently doubled curve."""
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    (d / MARKER_NAME).write_text(json.dumps({"state": "complete", "n_series": 144}) + "\n",
                                 encoding="utf-8")
    install_fake_wandb(monkeypatch)
    assert push(d, build_payload(d, name="evo"), project="p", force=True) == "skipped"


def test_force_does_not_override_the_run_lock(tmp_path, monkeypatch):
    """The two concerns are held by two files on purpose: --force overrides the record of what was
    pushed, never the lock, or two operators who both typed --force append two copies."""
    from alphaapollo.core.harness.resume import RunAlreadyActive

    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    run, _ = install_fake_wandb(monkeypatch)

    def held_by_someone_else(run_dir):
        raise RunAlreadyActive("held by live process 4242")

    monkeypatch.setattr("alphaapollo.core.harness.wandb_backfill.acquire_run_lock",
                        held_by_someone_else)

    # Contention is a failure, not a skip: exiting 0 here is the bug this whole check is about.
    assert push(d, build_payload(d, name="evo"), project="p", force=True) == "failed"
    assert run.logged == []


def test_the_lock_is_released_even_when_the_push_fails(tmp_path, monkeypatch):
    from alphaapollo.core.harness.resume import LOCK_FILENAME

    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    install_fake_wandb(monkeypatch, run=FakeRun(fail_on_finish=True))
    assert push(d, build_payload(d, name="evo"), project="p") == "failed"
    assert not (d / LOCK_FILENAME).exists()


def test_a_finish_that_raises_is_not_recorded_as_a_successful_push(tmp_path, monkeypatch):
    """wandb uploads on finish, so a finish that raises means the payload may never have left this
    machine -- a marker asserting otherwise would send the next attempt down the wrong path."""
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    install_fake_wandb(monkeypatch, run=FakeRun(fail_on_finish=True))
    assert push(d, build_payload(d, name="evo"), project="p") == "failed"
    assert json.loads((d / MARKER_NAME).read_text())["state"] == "failed"


def test_an_init_that_cannot_authenticate_is_a_failure_not_a_skip(tmp_path, monkeypatch):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    install_fake_wandb(monkeypatch, init_error=KeyboardInterrupt())
    # KeyboardInterrupt is what wandb raises from its interactive credential prompt, and it is a
    # BaseException that `except Exception` would miss.
    assert push(d, build_payload(d, name="evo"), project="p") == "failed"


def test_a_completed_push_is_refused_because_wandb_history_is_append_only(tmp_path, monkeypatch):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    install_fake_wandb(monkeypatch)
    assert push(d, build_payload(d, name="evo"), project="p") == "pushed"
    assert push(d, build_payload(d, name="evo"), project="p") == "skipped"


def test_a_second_process_cannot_slip_past_the_marker_before_it_is_written(tmp_path, monkeypatch):
    """The marker is created with O_EXCL *before* the first upload, so it is the lock as well as
    the record -- an existence check after the upload would leave both processes through."""
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    outcomes = []

    class Reentrant(FakeRun):
        def log(self, row):
            # Stands in for a concurrent invocation arriving while this upload is in flight.
            if not outcomes:
                outcomes.append(push(d, build_payload(d, name="evo"), project="p"))
            super().log(row)

    install_fake_wandb(monkeypatch, run=Reentrant())
    assert push(d, build_payload(d, name="evo"), project="p") == "pushed"
    assert outcomes == ["skipped"]


def test_a_run_that_was_never_tracked_by_wandb_is_skipped_not_fatal(tmp_path):
    """jsonl-only runs are a supported outcome of `HarnessTracker` -- there is simply no run to
    attach anything to, and saying so must not abort the other five runs."""
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)], run_id=None)
    assert read_run_id(d) is None
    assert push(d, build_payload(d, name="evo"), project="p") == "skipped"


def test_nothing_is_pushed_for_an_empty_payload(tmp_path):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0)])
    assert push(d, Payload(name="evo"), project="p") == "skipped"


def test_the_project_is_recovered_from_the_config_wandb_itself_stored(tmp_path):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0)])
    files = d / "wandb" / "run-20260916_131930-abc123" / "files"
    files.mkdir(parents=True)
    (files / "config.yaml").write_text(
        "wandb:\n    value:\n        project: alphaapollo-evo-harness\n        group: adapt\n",
        encoding="utf-8")
    assert detect_project(d) == "alphaapollo-evo-harness"


def test_an_unrecoverable_project_is_reported_rather_than_guessed(tmp_path):
    d = write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    assert push(d, build_payload(d, name="evo")) == "failed"
    # Nothing was attempted, so nothing may be recorded as attempted.
    assert not (d / MARKER_NAME).exists()


# --- CLI --------------------------------------------------------------------------------------


def write_phase(tmp_path, phase, rows=None, arms=analysis.ARMS):
    return {arm: write_run(tmp_path, f"{phase}-{arm}", rows or [problem_row(0, final=1)])
            for arm in arms}


def test_the_cli_exits_non_zero_when_a_run_fails_to_upload(tmp_path, monkeypatch):
    """`pushed 0 run(s)` on a totally failed backfill still read as success to a shell."""
    write_phase(tmp_path, "adapt")
    install_fake_wandb(monkeypatch, init_error=RuntimeError("401 unauthorized"))
    with pytest.raises(SystemExit) as exit_info:
        main(root=str(tmp_path), project="p", phases="adapt")
    assert exit_info.value.code == 1


def test_the_cli_exits_zero_when_every_run_is_merely_already_pushed(tmp_path, monkeypatch):
    """A deliberate skip is not a failure; conflating the two would make the exit code useless."""
    for d in write_phase(tmp_path, "adapt").values():
        (d / MARKER_NAME).write_text(
            json.dumps({"state": "complete", "series_sent": True}) + "\n", encoding="utf-8")
    install_fake_wandb(monkeypatch)
    main(root=str(tmp_path), project="p", phases="adapt")  # no SystemExit


def test_only_restricts_the_push_but_not_the_cross_arm_comparison(tmp_path, monkeypatch):
    """A filtered run's `analysis/common/*` must still be a comparison against the other arms --
    otherwise repairing one run would quietly redefine what its headline number means."""
    write_run(tmp_path, "adapt-baseline", [problem_row(0, final=1), problem_row(1)])
    write_run(tmp_path, "adapt-raw", [problem_row(0), problem_row(1)])
    write_run(tmp_path, "adapt-evo", [problem_row(0, final=1), problem_row(1, final=1)])
    run, calls = install_fake_wandb(monkeypatch)

    main(root=str(tmp_path), project="p", phases="adapt", only="adapt-evo")

    assert calls["id"] == "abc123"
    assert run.summary["analysis/common/pass_final"] == 1.0
    assert (tmp_path / "adapt-evo" / MARKER_NAME).exists()
    assert not (tmp_path / "adapt-baseline" / MARKER_NAME).exists()


# --- preflight --------------------------------------------------------------------------------


def test_a_missing_arm_stops_the_backfill_before_the_other_two_are_uploaded(tmp_path, monkeypatch):
    """The failure this guards against is not a crash; it is a cheerful summary after pushing two
    of three runs, where a per-run skip is invisible next to the successes."""
    write_phase(tmp_path, "adapt", arms=("baseline", "evo"))
    run, _ = install_fake_wandb(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        main(root=str(tmp_path), project="p", phases="adapt")
    assert exit_info.value.code == 1
    assert run.logged == [] and not (tmp_path / "adapt-evo" / MARKER_NAME).exists()


def test_an_empty_root_is_an_error_not_a_successful_no_op(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        main(root=str(tmp_path), project="p")
    assert exit_info.value.code == 1


def test_a_requested_phase_that_is_entirely_absent_fails(tmp_path, monkeypatch):
    write_phase(tmp_path, "adapt")
    install_fake_wandb(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        main(root=str(tmp_path), project="p", phases="adapt,heldout")
    assert exit_info.value.code == 1


def test_a_run_without_a_wandb_id_fails_preflight_rather_than_being_skipped(tmp_path, monkeypatch):
    """Five successes and one silent skip is exactly the shape of an incomplete backfill that
    reports success."""
    write_phase(tmp_path, "adapt")
    (tmp_path / "adapt-raw" / "wandb_run_id.txt").unlink()
    run, _ = install_fake_wandb(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        main(root=str(tmp_path), project="p", phases="adapt")
    assert exit_info.value.code == 1
    assert run.logged == []


def test_an_empty_metrics_file_fails_preflight(tmp_path, monkeypatch):
    write_phase(tmp_path, "adapt")
    (tmp_path / "adapt-raw" / "metrics.jsonl").write_text("", encoding="utf-8")
    install_fake_wandb(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        main(root=str(tmp_path), project="p", phases="adapt")
    assert exit_info.value.code == 1


def test_an_unresolvable_project_fails_preflight_before_anything_is_attempted(tmp_path,
                                                                             monkeypatch):
    write_phase(tmp_path, "adapt")
    install_fake_wandb(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        main(root=str(tmp_path), phases="adapt")  # no --project, and no wandb config on disk
    assert exit_info.value.code == 1


def test_allow_missing_accepts_a_partly_run_experiment(tmp_path, monkeypatch):
    """A mid-experiment backfill is a real use, but it has to be asked for."""
    write_phase(tmp_path, "adapt", arms=("baseline", "evo"))
    run, _ = install_fake_wandb(monkeypatch)
    main(root=str(tmp_path), project="p", phases="adapt", allow_missing=True)
    assert run.summary["analysis/overall/pass_final"] == 1.0


def test_a_dry_run_reports_preflight_problems_without_refusing_to_preview(tmp_path):
    """Nothing is uploaded, so there is nothing for an incomplete target set to corrupt -- and a
    preview is the natural way to look at an experiment that is still running."""
    write_phase(tmp_path, "adapt", arms=("evo",))
    main(root=str(tmp_path), project="p", phases="adapt", dry_run=True)
    assert (tmp_path / "adapt-evo" / PREVIEW_NAME).exists()


def test_an_only_filter_that_matches_nothing_is_an_error_not_a_silent_no_op(tmp_path):
    write_run(tmp_path, "adapt-evo", [problem_row(0, final=1)])
    with pytest.raises(SystemExit) as exit_info:
        main(root=str(tmp_path), project="p", phases="adapt", only="adapt-typo")
    assert exit_info.value.code == 1


def test_only_does_not_require_the_arms_it_did_not_name(tmp_path, monkeypatch):
    """An explicit --only is its own completeness claim: these named runs, all of them."""
    write_phase(tmp_path, "adapt")
    install_fake_wandb(monkeypatch)
    main(root=str(tmp_path), project="p", phases="adapt,heldout", only="adapt-evo")
    assert (tmp_path / "adapt-evo" / MARKER_NAME).exists()
