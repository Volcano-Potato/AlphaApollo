from alphaapollo.core.harness.guard import _NUMBER, validate_skill
from alphaapollo.core.harness.schema import CandidateMemory

QUESTION = ("Find the number of ordered pairs of positive integers (a, b) such that "
            "a + b = 1000 and neither a nor b has a zero digit.")
GT = "738"


def candidate(**kw) -> CandidateMemory:
    base = dict(
        trigger="Counting integer pairs under digit constraints.",
        lesson="- Enumerate a small analogue in python before generalizing.",
        failure_mode="Extrapolating without numeric verification.",
        scope_hint="topic", topic="number_theory", evidence=["p_0001:symbolic_slip"],
    )
    base.update(kw)
    return CandidateMemory(**base)


def check(c):
    return validate_skill(c, question_texts=[QUESTION], ground_truths=[GT])


def test_clean_candidate_is_accepted():
    assert check(candidate()) == (True, None)


def test_non_english_is_rejected():
    ok, reason = check(candidate(lesson="- 先用 python 暴力枚举小范围再推广到一般情况。"))
    assert ok is False and reason == "non_english"


def test_verbatim_question_span_is_rejected():
    leaked = "- Find the number of ordered pairs of positive integers such that a + b"
    ok, reason = check(candidate(lesson=leaked))
    assert ok is False and reason == "question_overlap"


def test_ground_truth_digits_next_to_an_assertion_cue_are_rejected():
    ok, reason = check(candidate(failure_mode="Forgetting that the answer is 738."))
    assert ok is False and reason == "answer_leak"


def test_numbers_unrelated_to_ground_truth_are_allowed():
    ok, reason = check(candidate(lesson="- Brute-force the range n <= 50 first."))
    assert ok is True and reason is None


def test_overlong_lesson_is_rejected():
    ok, reason = check(candidate(lesson="- " + " ".join(["word"] * 61)))
    assert ok is False and reason == "lesson_too_long"


def test_empty_section_is_rejected():
    ok, reason = check(candidate(trigger="   "))
    assert ok is False and reason == "empty_section"


def test_a_bound_that_merely_coincides_with_a_ground_truth_is_kept_but_flagged():
    """Measured on the real adaptation pool: 4% of AIME answers are exactly the
    round numbers a skill uses as a bound, so a bare equality test rejects
    21-28% of the enumerate-first skills this project exists to learn."""
    ok, reason = validate_skill(candidate(lesson="- Brute-force the range n <= 50 first."),
                                question_texts=[QUESTION], ground_truths=["50"])
    assert ok is True and reason == "numeric_coincidence"


def test_an_assertion_cue_on_another_line_does_not_trigger_a_reject():
    leaky_looking = "- Verify your answer numerically.\n- Brute-force the range n <= 50 first."
    ok, reason = validate_skill(candidate(lesson=leaky_looking),
                                question_texts=[QUESTION], ground_truths=["50"])
    assert ok is True and reason == "numeric_coincidence"


def test_assertion_cue_on_the_same_line_still_rejects_across_all_three_sections():
    for field in ("trigger", "lesson", "failure_mode"):
        c = candidate(**{field: "The solution equals 50 in this family."})
        ok, reason = validate_skill(c, question_texts=[QUESTION], ground_truths=["50"])
        assert (ok, reason) == (False, "answer_leak"), f"missed a leak in {field}"


# _NUMBER underpins both answer_leak and numeric_coincidence, so it gets its own direct
# regression coverage instead of being exercised only indirectly through validate_skill().
# The negative lookahead used to be `(?![\w.])`, which -- meaning to exclude decimals like
# "738.5" -- also excluded any number immediately followed by a sentence-ending period
# ("The answer is 50."), making that the single most natural way to write a leak invisible
# to every downstream check. `(?!\.?\d)(?!\w)` excludes only a "." that is itself followed
# by a digit (an actual decimal point) while still blocking identifier-embedded digit runs.
def test_number_regex_sees_a_digit_run_immediately_before_a_sentence_period():
    assert _NUMBER.findall("The answer is 50.") == ["50"]


def test_number_regex_sees_a_digit_run_before_a_period_with_no_trailing_context():
    assert _NUMBER.findall("n <= 50.") == ["50"]


def test_number_regex_sees_a_bare_digit_run_followed_only_by_a_period():
    assert _NUMBER.findall("50.") == ["50"]


def test_number_regex_still_excludes_a_true_decimal():
    assert _NUMBER.findall("738.5") == []


def test_number_regex_still_excludes_a_leading_zero_decimal():
    assert _NUMBER.findall("0.375") == []


def test_number_regex_still_excludes_digits_embedded_in_an_identifier():
    assert _NUMBER.findall("abc738") == []
    assert _NUMBER.findall("738abc") == []


def test_number_regex_still_excludes_a_dotted_version_string():
    assert _NUMBER.findall("v1.2.3") == []


def test_number_regex_still_matches_parenthesized_and_punctuated_numbers():
    assert _NUMBER.findall("(50)") == ["50"]
    assert _NUMBER.findall("50, and") == ["50"]
    assert _NUMBER.findall("answer: 50") == ["50"]
