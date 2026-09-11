import json

import pytest

from alphaapollo.core.harness.schema import Skill
from alphaapollo.core.harness.store import SkillStore


def skill(i: int, level: str = "topic", topic: str = "number_theory") -> Skill:
    return Skill(id=f"sk_{i:04d}", name=f"skill-{i}", level=level,
                 topic=None if level == "general" else topic,
                 trigger=f"Trigger {i}.", lesson=f"- Lesson {i}.",
                 failure_mode=f"Avoid {i}.", created_at=i)


def test_skills_survive_a_fresh_store_instance(tmp_path):
    store = SkillStore(tmp_path)
    store._write_skill(skill(1))
    store._write_skill(skill(2, level="general"))

    reopened = SkillStore(tmp_path)
    got = {s.id: s for s in reopened.all()}
    assert set(got) == {"sk_0001", "sk_0002"}
    assert got["sk_0002"].level == "general" and got["sk_0002"].topic is None
    assert got["sk_0001"].lesson == "- Lesson 1."


def test_counters_survive_reload(tmp_path):
    store = SkillStore(tmp_path)
    s = skill(1)
    s.n_selected, s.n_selected_success = 12, 7
    store._write_skill(s)

    restored = SkillStore(tmp_path).all()[0]
    assert (restored.n_selected, restored.n_selected_success) == (12, 7)


def test_log_event_appends_one_json_line_each_call(tmp_path):
    store = SkillStore(tmp_path)
    store.log_event(problem_idx=1, batch=0, op="ADD", skill_id="sk_0001",
                    actor="topic_curator", reason="new pattern", accepted=True,
                    reject_reason=None)
    store.log_event(problem_idx=2, batch=0, op="ADD", skill_id=None,
                    actor="topic_curator", reason="dup", accepted=False,
                    reject_reason="question_overlap")

    lines = (tmp_path / "harness_log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    second = json.loads(lines[1])
    assert second["accepted"] is False and second["reject_reason"] == "question_overlap"


def test_selection_log_is_a_separate_file(tmp_path):
    store = SkillStore(tmp_path)
    store.log_selection(problem_idx=41, batch=5, topic="number_theory",
                        selected=[{"skill_id": "sk_0007", "score": 0.72, "tokens": 94}],
                        total_inject_tokens=611, pass1_round0=0, pass_final=1)
    payload = json.loads((tmp_path / "selection_log.jsonl").read_text().strip())
    assert payload["selected"][0]["skill_id"] == "sk_0007"
    assert not (tmp_path / "harness_log.jsonl").exists()


def test_snapshot_is_isolated_from_later_mutations(tmp_path):
    store = SkillStore(tmp_path)
    store._write_skill(skill(1))
    frozen = store.snapshot()
    store._write_skill(skill(2))
    assert len(frozen) == 1 and len(store.all()) == 2


def test_next_id_is_monotonic_across_reloads(tmp_path):
    store = SkillStore(tmp_path)
    store._write_skill(skill(1))
    store._write_skill(skill(9))
    assert SkillStore(tmp_path)._next_id() == "sk_0010"


def test_write_skill_rejects_a_general_skill_with_a_topic(tmp_path):
    """level == 'general' <=> topic is None. A general-level skill carrying a topic string
    is an inconsistent write and must never reach disk silently -- see the deferred review
    note from Task 1's schema review, which asked for this check on the SkillStore write
    path rather than assuming callers always pass a consistent (level, topic) pair."""
    store = SkillStore(tmp_path)
    bad = skill(1, level="topic")  # constructed with topic="number_theory" by the helper
    bad.level = "general"  # now inconsistent: general level, non-None topic
    with pytest.raises(ValueError, match="general"):
        store._write_skill(bad)


def test_write_skill_rejects_a_topic_skill_without_a_topic(tmp_path):
    store = SkillStore(tmp_path)
    bad = skill(1, level="topic")
    bad.topic = None  # now inconsistent: topic level, missing topic
    with pytest.raises(ValueError, match="topic"):
        store._write_skill(bad)
