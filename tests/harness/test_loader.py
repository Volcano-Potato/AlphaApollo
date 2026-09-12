import pandas as pd
import pytest

from alphaapollo.core.harness.loader import batches, interleave, load_stream

TOPICS = ("algebra", "number_theory", "combinatorics", "geometry")


def make_parquet(tmp_path):
    rows = []
    for i in range(8):
        rows.append({
            "extra_info": {"question": f"Q{i}", "ground_truth": str(i),
                           "topic": TOPICS[i % 4], "problem_shape": "shape",
                           "technique": "divisibility-counting", "year": 2018 + i // 4,
                           "contest": "I", "number": i}
        })
    path = tmp_path / "stream.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_loader_preserves_topic_and_year(tmp_path):
    problems = load_stream(make_parquet(tmp_path))
    assert problems[0]["topic"] == "algebra"
    assert problems[0]["year"] == 2018
    assert problems[0]["technique"] == "divisibility-counting"


def test_loader_always_supplies_gt_traj(tmp_path):
    """run_problem reads current_problem["gt_traj"] unconditionally."""
    for problem in load_stream(make_parquet(tmp_path)):
        assert "gt_traj" in problem


def test_loader_assigns_sequential_problem_idx(tmp_path):
    problems = load_stream(make_parquet(tmp_path))
    assert [p["problem_idx"] for p in problems] == list(range(8))


def make_pool(per_topic=10):
    pool = []
    for t_i, topic in enumerate(TOPICS):
        for i in range(per_topic):
            pool.append({"problem_idx": t_i * per_topic + i, "topic": topic,
                         "year": 2018 + i // 5, "number": i, "question": f"{topic}-{i}"})
    return pool


def test_interleave_gives_every_batch_two_of_each_topic():
    stream = interleave(make_pool(), batch_size=8)
    for batch in batches(stream, 8):
        counts = {t: sum(1 for p in batch if p["topic"] == t) for t in TOPICS}
        assert counts == {t: 2 for t in TOPICS}


def test_interleave_preserves_within_topic_chronology():
    stream = interleave(make_pool(), batch_size=8)
    for topic in TOPICS:
        seq = [p["number"] for p in stream if p["topic"] == topic]
        assert seq == sorted(seq)


def test_interleave_renumbers_problem_idx_to_stream_position():
    stream = interleave(make_pool(), batch_size=8)
    assert [p["problem_idx"] for p in stream] == list(range(len(stream)))


def test_interleave_is_deterministic():
    assert [p["question"] for p in interleave(make_pool(), 8)] == \
           [p["question"] for p in interleave(make_pool(), 8)]


def test_interleave_stops_when_a_topic_runs_out():
    pool = [p for p in make_pool() if not (p["topic"] == "geometry" and p["number"] >= 2)]
    stream = interleave(pool, batch_size=8)
    assert len(stream) == 8, "only one full balanced batch is possible"


def test_batches_drops_the_incomplete_tail():
    assert len(batches(list(range(20)), 8)) == 2


def test_interleave_rejects_a_batch_size_not_divisible_by_topic_count():
    with pytest.raises(ValueError):
        interleave(make_pool(), batch_size=6)


# --- Adversarial cases (self-designed, not in the brief) ---------------------------------


def test_topic_count_exactly_equals_one_batch_share():
    """A topic with exactly batch_size // n_topics problems should still form one full batch."""
    pool = make_pool(per_topic=2)  # batch_size=8, 4 topics -> 2 per batch share
    stream = interleave(pool, batch_size=8)
    assert len(stream) == 8


def test_topic_count_one_short_of_batch_share_yields_empty_stream():
    pool = [p for p in make_pool(per_topic=2) if not (p["topic"] == "geometry" and p["number"] == 1)]
    stream = interleave(pool, batch_size=8)
    assert stream == []


def test_topic_with_zero_problems_yields_empty_stream():
    pool = [p for p in make_pool() if p["topic"] != "geometry"]
    stream = interleave(pool, batch_size=8)
    assert stream == []


def test_extra_info_missing_fills_blank_fields(tmp_path):
    path = tmp_path / "missing_extra_info.parquet"
    pd.DataFrame([{"other_col": 1}, {"other_col": 2}]).to_parquet(path)
    problems = load_stream(path)
    assert len(problems) == 2
    for p in problems:
        assert p["gt_traj"] == ""
        assert p["question"] == ""
        assert p["topic"] == ""


def test_extra_info_not_a_dict_fills_blank_fields(tmp_path):
    path = tmp_path / "non_dict_extra_info.parquet"
    pd.DataFrame([{"extra_info": "not-a-dict"}, {"extra_info": None}]).to_parquet(path)
    problems = load_stream(path)
    assert len(problems) == 2
    for p in problems:
        assert p["gt_traj"] == ""
        assert p["topic"] == ""


def test_year_as_string_is_preserved_not_coerced(tmp_path):
    rows = [{"extra_info": {"question": "Q", "ground_truth": "1", "topic": "algebra",
                             "problem_shape": "s", "technique": "t", "year": "2018",
                             "contest": "I", "number": 1}}]
    path = tmp_path / "string_year.parquet"
    pd.DataFrame(rows).to_parquet(path)
    problems = load_stream(path)
    assert problems[0]["year"] == "2018"


def test_interleave_sorts_correctly_with_mixed_year_types():
    """year/contest as strings vs ints must not raise TypeError during sort."""
    pool = make_pool(per_topic=3)
    for p in pool:
        if p["topic"] == "algebra":
            p["year"] = str(p["year"])
    stream = interleave(pool, batch_size=8)
    assert len(stream) >= 0  # must not raise


def test_interleave_handles_missing_contest_key():
    """make_pool() never sets 'contest' at all -- sorting must not KeyError on it."""
    pool = make_pool(per_topic=3)
    assert all("contest" not in p for p in pool)
    stream = interleave(pool, batch_size=8)
    assert isinstance(stream, list)


def test_interleave_handles_none_contest_value():
    pool = make_pool(per_topic=3)
    for p in pool:
        p["contest"] = None if p["number"] % 2 == 0 else "I"
    stream = interleave(pool, batch_size=8)
    assert isinstance(stream, list)


def test_load_stream_duplicate_problem_idx_not_produced(tmp_path):
    """load_stream must assign fresh sequential idx even if extra_info carried one."""
    rows = [{"extra_info": {"question": "Q0", "problem_idx": 99}},
            {"extra_info": {"question": "Q1", "problem_idx": 99}}]
    path = tmp_path / "dup.parquet"
    pd.DataFrame(rows).to_parquet(path)
    problems = load_stream(path)
    assert [p["problem_idx"] for p in problems] == [0, 1]


def test_load_stream_empty_parquet(tmp_path):
    path = tmp_path / "empty.parquet"
    pd.DataFrame({"extra_info": pd.Series(dtype=object)}).to_parquet(path)
    assert load_stream(path) == []


def test_interleave_empty_input_returns_empty_list():
    assert interleave([]) == []


def test_batches_empty_input_returns_empty_list():
    assert batches([], 8) == []
