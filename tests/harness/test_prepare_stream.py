import pandas as pd
import pytest

from alphaapollo.core.harness.loader import load_stream
from alphaapollo.data_preprocess.prepare_harness_stream import (
    ANSWER_RE,
    build_rows,
    normalise_aime_row,
    write_stream,
)


def aime_row(year=2018, part="I", number=1, answer="042", question=None):
    # Distinct question text per row by default: build_rows de-duplicates by question, so a
    # shared default would silently collapse multi-row fixtures and make ordering tests
    # unfalsifiable. Tests that care about duplicates pass `question=` explicitly.
    question = f"Find n for {year}-{part}-{number}." if question is None else question
    return {"ID": f"{year}-{part}-{number}", "Year": year, "Problem Number": str(number),
            "Question": question, "Answer": answer, "Part": part}


def test_a_clean_row_becomes_a_stream_problem():
    row = normalise_aime_row(aime_row(), data_source="gneubig/aime-1983-2024")
    assert row["question"] == "Find n for 2018-I-1."
    assert row["ground_truth"] == "042"
    assert row["year"] == 2018
    assert row["contest"] == "I"
    assert row["number"] == 1


def test_an_answer_that_is_not_a_plain_aime_integer_is_dropped():
    """AIME answers are integers 0-999. 2022-II-8 ships as "080 or 081 (both were accepted)":
    real data, one row, and the scorer compares strings -- keeping it would mark every attempt
    wrong regardless of arm, quietly biasing one problem against all three."""
    assert normalise_aime_row(aime_row(answer="080 or 081 (both were accepted)"), data_source="x") is None


@pytest.mark.parametrize("answer", ["", "   ", "1000", "-1", "12.5", "n/a"])
def test_other_malformed_answers_are_dropped_too(answer):
    assert normalise_aime_row(aime_row(answer=answer), data_source="x") is None


@pytest.mark.parametrize("answer", ["0", "42", "042", "999", " 7 "])
def test_valid_aime_answers_survive(answer):
    assert ANSWER_RE.fullmatch(answer.strip())
    assert normalise_aime_row(aime_row(answer=answer), data_source="x") is not None


def test_a_row_with_no_question_is_dropped():
    assert normalise_aime_row(aime_row(question="   "), data_source="x") is None


def test_the_stream_is_ordered_by_year_then_contest_then_number():
    """Assignment: problems arrive in a fixed order, split by year, never shuffled-then-split.
    Contest is compared as I < II, and number numerically -- a plain string sort would put
    problem 10 before problem 2."""
    raw = [aime_row(2019, "II", 2), aime_row(2018, "I", 10), aime_row(2018, "I", 2),
           aime_row(2018, "II", 1), aime_row(2019, "I", 15)]
    rows = build_rows(raw, data_source="x")
    assert [(r["year"], r["contest"], r["number"]) for r in rows] == [
        (2018, "I", 2), (2018, "I", 10), (2018, "II", 1), (2019, "I", 15), (2019, "II", 2)]


def test_problem_idx_is_assigned_after_ordering_and_is_contiguous():
    raw = [aime_row(2019, "I", 1), aime_row(2018, "I", 1)]
    rows = build_rows(raw, data_source="x")
    assert [r["problem_idx"] for r in rows] == [0, 1]
    assert rows[0]["year"] == 2018


def test_written_parquet_round_trips_through_the_harness_loader(tmp_path):
    """The written file has to satisfy BOTH readers: the harness's own load_stream and upstream's
    run_problem, which indexes question/ground_truth/gt_traj/data_source unconditionally."""
    rows = build_rows([aime_row(2018, "I", 1), aime_row(2018, "I", 2)], data_source="ds")
    path = write_stream(rows, tmp_path / "stream.parquet")

    problems = load_stream(path)
    assert len(problems) == 2
    for problem in problems:
        for key in ("question", "ground_truth", "gt_traj", "data_source", "topic", "year"):
            assert key in problem
        assert problem["data_source"] == "ds"


def test_written_parquet_keeps_upstreams_own_columns(tmp_path):
    """`load_informal_math_data` and the verl-shaped pipeline read these top-level columns; a
    stream file that only satisfies load_stream would break any other consumer."""
    rows = build_rows([aime_row()], data_source="ds")
    df = pd.read_parquet(write_stream(rows, tmp_path / "s.parquet"))
    for column in ("data_source", "prompt", "ability", "reward_model", "extra_info", "env_kwargs"):
        assert column in df.columns
    assert df.iloc[0]["reward_model"]["ground_truth"] == "042"
    assert df.iloc[0]["env_kwargs"]["question"] == "Find n for 2018-I-1."


def test_adversarial_the_ground_truth_never_reaches_the_topic_label(tmp_path):
    """Topic labels are analysis metadata now, but they are still produced by a model call, and
    that call must only ever see the question."""
    seen = []

    class RecordingAgent:
        def get_action_from_gpt(self, obs):
            seen.append(obs)
            return "algebra"

    rows = build_rows([aime_row(answer="042", question="Find n.")], data_source="ds",
                      agent=RecordingAgent())
    assert rows[0]["topic"] == "algebra"
    assert seen and "Find n." in seen[0]
    assert "042" not in seen[0]


def test_without_an_agent_topics_are_left_blank_rather_than_guessed(tmp_path):
    rows = build_rows([aime_row()], data_source="ds")
    assert rows[0]["topic"] == ""


def test_adversarial_a_human_label_is_preferred_over_the_model(tmp_path):
    """MathArena's 2025 split ships human problem_type; no call should be spent overriding it."""
    class ExplodingAgent:
        def get_action_from_gpt(self, obs):
            raise AssertionError("must not be called for an already-labelled row")

    raw = [{**aime_row(), "topic": "number_theory"}]
    rows = build_rows(raw, data_source="ds", agent=ExplodingAgent())
    assert rows[0]["topic"] == "number_theory"


def test_adversarial_duplicate_questions_across_years_are_dropped_once(tmp_path):
    """A repeated problem would be solved twice with different harness states, making the
    per-problem record ambiguous and double-counting it in the success rate."""
    rows = build_rows([aime_row(2018, "I", 1, question="Same."),
                       aime_row(2019, "I", 1, question="Same.")], data_source="ds")
    assert len(rows) == 1
    assert rows[0]["year"] == 2018, "the earlier occurrence is the one that survives"


def test_adversarial_writing_an_empty_stream_raises_rather_than_producing_a_dead_file(tmp_path):
    """A silently-empty stream would make every arm score 0/0 and look like a completed run."""
    with pytest.raises(ValueError):
        write_stream([], tmp_path / "empty.parquet")
