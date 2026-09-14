import json

import pytest

from alphaapollo.core.harness.resume import (
    PROGRESS_FILENAME,
    ResumeMismatch,
    fingerprint,
    plan_resume,
    prepare_resume,
    read_progress,
    truncate_jsonl,
    write_progress,
)

FP = {"stream_path": "s.parquet", "arm": "evo", "batch_size": 4, "seed": 1234, "frozen": False}


def write_lines(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# --- progress marker ----------------------------------------------------------------------


def test_no_marker_means_a_fresh_run(tmp_path):
    plan = plan_resume(tmp_path, fp=FP, batch_size=4)
    assert plan.is_fresh and plan.start_batch == 0 and plan.first_problem_idx == 0


def test_marker_round_trips(tmp_path):
    write_progress(tmp_path, completed_batches=3, fp=FP)
    assert read_progress(tmp_path) == {"completed_batches": 3, "fingerprint": FP}


def test_plan_resumes_at_the_batch_after_the_last_completed_one(tmp_path):
    write_progress(tmp_path, completed_batches=3, fp=FP)
    plan = plan_resume(tmp_path, fp=FP, batch_size=4)
    assert plan.start_batch == 3
    assert plan.first_problem_idx == 12, "batch 3 begins at problem 3*4"
    assert not plan.is_fresh


def test_write_progress_leaves_no_temp_file_behind(tmp_path):
    write_progress(tmp_path, completed_batches=1, fp=FP)
    assert [p.name for p in tmp_path.iterdir()] == [PROGRESS_FILENAME]


def test_a_corrupt_marker_restarts_from_zero_rather_than_raising(tmp_path):
    """The marker is written atomically, so a corrupt one means the very first write was
    interrupted -- indistinguishable from a run that never completed a batch."""
    (tmp_path / PROGRESS_FILENAME).write_text("{not json", encoding="utf-8")
    assert read_progress(tmp_path) is None
    assert plan_resume(tmp_path, fp=FP, batch_size=4).is_fresh


# --- fingerprint --------------------------------------------------------------------------


@pytest.mark.parametrize("field,value", [("arm", "raw"), ("stream_path", "other.parquet"),
                                         ("batch_size", 8), ("seed", 7), ("frozen", True)])
def test_resuming_into_a_different_run_is_refused(tmp_path, field, value):
    """Splicing two experiments together produces a run whose halves came from different
    configs -- unrecoverable after the fact, so this refuses instead of degrading."""
    write_progress(tmp_path, completed_batches=2, fp=FP)
    with pytest.raises(ResumeMismatch) as exc:
        plan_resume(tmp_path, fp={**FP, field: value}, batch_size=4)
    assert field in str(exc.value)


def test_max_workers_is_not_part_of_the_identity_of_a_run():
    """A throughput knob may change between attempts -- resuming with fewer workers after a
    rate-limit failure is exactly the intended use."""
    assert "max_workers" not in fingerprint({"max_workers": 5, "arm": "evo"})


def test_fingerprint_of_a_config_missing_fields_is_stable():
    assert fingerprint({}) == dict.fromkeys(
        ("stream_path", "arm", "batch_size", "seed", "frozen"), None)


# --- truncation ---------------------------------------------------------------------------


def test_truncate_drops_only_rows_at_or_past_the_cutoff(tmp_path):
    path = write_lines(tmp_path / "m.jsonl", [{"step": i} for i in range(8)])
    assert truncate_jsonl(path, key="step", first_dropped=4) == 4
    kept = [json.loads(line)["step"] for line in path.read_text().splitlines()]
    assert kept == [0, 1, 2, 3]


def test_truncate_is_a_no_op_when_nothing_is_past_the_cutoff(tmp_path):
    path = write_lines(tmp_path / "m.jsonl", [{"step": i} for i in range(4)])
    before = path.read_text()
    assert truncate_jsonl(path, key="step", first_dropped=4) == 0
    assert path.read_text() == before


def test_truncate_on_a_missing_file_is_harmless(tmp_path):
    assert truncate_jsonl(tmp_path / "absent.jsonl", key="step", first_dropped=0) == 0


def test_rows_without_the_key_are_kept(tmp_path):
    """metrics.jsonl mixes per-problem rows with per-batch accounting rows; a row carrying no
    comparable position is not one this function can judge, and dropping it would lose
    accounting the rerun does not regenerate."""
    path = write_lines(tmp_path / "m.jsonl", [{"step": 0}, {"note": "x"}, {"step": 9}])
    assert truncate_jsonl(path, key="step", first_dropped=4) == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"step": 0}, {"note": "x"}]


def test_a_boolean_is_not_treated_as_a_position(tmp_path):
    """bool is a subclass of int in Python; `True >= 1` would silently drop this row."""
    path = write_lines(tmp_path / "m.jsonl", [{"step": True}, {"step": 9}])
    assert truncate_jsonl(path, key="step", first_dropped=1) == 1
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [True]


def test_a_torn_final_line_is_kept_as_evidence(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps({"step": 0}) + "\n" + '{"step": 9', encoding="utf-8")
    truncate_jsonl(path, key="step", first_dropped=4)
    assert '{"step": 9' in path.read_text()


def test_truncate_leaves_no_temp_file_behind(tmp_path):
    path = write_lines(tmp_path / "m.jsonl", [{"step": i} for i in range(8)])
    truncate_jsonl(path, key="step", first_dropped=4)
    assert [p.name for p in tmp_path.iterdir()] == ["m.jsonl"]


# --- prepare_resume, the composed path ------------------------------------------------------


def test_prepare_resume_drops_the_aborted_batch_partial_rows(tmp_path):
    """A crash midway through batch 3 leaves rows for the problems that did finish. They are
    about to run again, so leaving them would double-count every one of them."""
    write_progress(tmp_path, completed_batches=3, fp=FP)
    metrics = write_lines(tmp_path / "metrics.jsonl", [{"step": i} for i in range(14)])
    selections = write_lines(tmp_path / "selection_log.jsonl",
                             [{"problem_idx": i} for i in range(14)])

    plan = prepare_resume(tmp_path, fp=FP, batch_size=4,
                          partial_logs={str(metrics): "step", str(selections): "problem_idx"})

    assert plan.start_batch == 3
    assert len(metrics.read_text().splitlines()) == 12
    assert len(selections.read_text().splitlines()) == 12


def test_prepare_resume_touches_nothing_on_a_fresh_run(tmp_path):
    metrics = write_lines(tmp_path / "metrics.jsonl", [{"step": i} for i in range(4)])
    before = metrics.read_text()
    plan = prepare_resume(tmp_path, fp=FP, batch_size=4, partial_logs={str(metrics): "step"})
    assert plan.is_fresh
    assert metrics.read_text() == before, "a fresh run must never truncate a pre-existing log"
