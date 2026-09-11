import pytest

from alphaapollo.core.harness.schema import OPS, CandidateMemory, Skill, SkillEdit, skill_from_markdown, skill_to_markdown


def make_skill(**kw) -> Skill:
    base = dict(
        id="sk_0007", name="enumerate-before-generalize", level="topic",
        topic="number_theory",
        trigger="Counting integers subject to divisibility or congruence conditions.",
        lesson="- Brute-force small range in python first.\n- Check the closed form back against the enumeration.",
        failure_mode="Extrapolating a small-range pattern without numeric verification.",
        evidence=["p_0023:symbolic_slip"], created_at=23, revised_at=[41],
        n_selected=12, n_selected_success=7, n_tokens=94,
    )
    base.update(kw)
    return Skill(**base)


def test_markdown_round_trip_preserves_every_field():
    original = make_skill()
    restored = skill_from_markdown(skill_to_markdown(original))
    assert restored == original


def test_markdown_uses_the_three_section_layout():
    text = skill_to_markdown(make_skill())
    assert "## When to use" in text
    assert "## Strategy" in text
    assert "## Avoid" in text


def test_utility_is_laplace_smoothed():
    assert make_skill(n_selected=0, n_selected_success=0).utility() == pytest.approx(0.5)
    assert make_skill(n_selected=12, n_selected_success=7).utility() == pytest.approx(8 / 14)


def test_general_skill_has_no_topic():
    s = make_skill(level="general", topic=None)
    assert skill_from_markdown(skill_to_markdown(s)).topic is None


def test_markdown_round_trip_survives_triple_dash_inside_a_field_value():
    original = make_skill(name="foo --- bar")
    restored = skill_from_markdown(skill_to_markdown(original))
    assert restored == original


def test_skill_edit_rejects_unknown_op():
    with pytest.raises(ValueError):
        SkillEdit(op="FROBNICATE", actor="topic_curator", reason="x")


def test_candidate_memory_rejects_unknown_scope_hint():
    with pytest.raises(ValueError):
        CandidateMemory(trigger="t", lesson="l", failure_mode="f",
                        scope_hint="sideways", topic=None, evidence=[])


def test_candidate_memory_defaults_to_a_new_skill_proposal():
    c = CandidateMemory(trigger="t", lesson="l", failure_mode="f",
                        scope_hint="topic", topic="number_theory", evidence=[])
    assert c.action_hint == "NEW" and c.target_id is None


def test_candidate_memory_rejects_unknown_action_hint():
    with pytest.raises(ValueError):
        CandidateMemory(trigger="t", lesson="l", failure_mode="f", scope_hint="topic",
                        topic="number_theory", evidence=[], action_hint="OBLITERATE")


def test_ops_frozen_set_is_exactly_the_five_operators():
    assert OPS == frozenset({"ADD", "MERGE", "REVISE", "DELETE", "SKIP"})
