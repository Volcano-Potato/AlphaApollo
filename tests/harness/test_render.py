from alphaapollo.core.harness.render import NEUTRAL_SYSTEM_PROMPT, count_tokens, render_harness
from alphaapollo.core.harness.schema import Skill


def skill(i: int, level: str = "topic") -> Skill:
    return Skill(id=f"sk_{i:04d}", name=f"skill-{i}", level=level,
                 topic=None if level == "general" else "number_theory",
                 trigger=f"Trigger {i}.", lesson=f"- Lesson {i}.",
                 failure_mode=f"Avoid {i}.")


def test_count_tokens_is_monotonic_and_positive():
    assert count_tokens("") == 0
    assert 0 < count_tokens("hello world") < count_tokens("hello world again and again")


def test_empty_harness_renders_the_neutral_prompt_not_an_empty_string():
    out = render_harness([])
    assert out == NEUTRAL_SYSTEM_PROMPT
    assert out, "empty system_prompt would suppress the system message entirely"


def test_render_groups_general_before_topic():
    out = render_harness([skill(1, "topic"), skill(2, "general")])
    assert out.index("skill-2") < out.index("skill-1")


def test_render_includes_all_three_sections_of_each_skill():
    out = render_harness([skill(1)])
    assert "Trigger 1." in out and "- Lesson 1." in out and "Avoid 1." in out


def test_render_never_includes_frontmatter_metadata():
    out = render_harness([skill(1)])
    for leaked in ("n_selected", "evidence", "created_at", "sk_0001"):
        assert leaked not in out
