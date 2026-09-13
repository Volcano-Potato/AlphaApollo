import json

import pytest

from alphaapollo.core.harness.export import build_report, export_harness
from alphaapollo.core.harness.schema import CandidateMemory, SkillEdit
from alphaapollo.core.harness.store import Caps, SkillStore


def seeded_store(tmp_path):
    """A store carrying one general skill, one topic skill, and one recorded rejection."""
    store = SkillStore(tmp_path / "store", caps=Caps(general=5, per_topic=5))
    store.apply(
        [
            SkillEdit(op="ADD", actor="general_curator", reason="broadly useful",
                      payload=CandidateMemory(trigger="when a count seems to explode",
                                              lesson="- bound the search space first",
                                              failure_mode="brute-forcing an infinite range",
                                              scope_hint="general", topic=None, evidence=["p_3", "p_5"])),
            SkillEdit(op="ADD", actor="topic_curator", reason="localized procedure",
                      payload=CandidateMemory(trigger="when solving a congruence",
                                              lesson="- check small residues numerically",
                                              failure_mode="assuming uniform residues",
                                              scope_hint="topic", topic="number_theory",
                                              evidence=["p_7"])),
            SkillEdit(op="SKIP", actor="topic_curator", reason="not referenced by the curator's reply",
                      payload=CandidateMemory(trigger="t", lesson="l", failure_mode="a",
                                              scope_hint="topic", topic="algebra", evidence=["p_9"])),
        ],
        problem_idx=0, batch=0, question_texts=[], ground_truths=[],
    )
    store.log_selection(problem_idx=11, topic="number_theory", skill_ids=["sk_0002"],
                        n_tokens=84, success=True)
    store.log_selection(problem_idx=12, topic="algebra", skill_ids=["sk_0001"],
                        n_tokens=61, success=False)
    return store


def test_export_writes_both_a_readable_harness_and_a_machine_readable_summary(tmp_path):
    store = seeded_store(tmp_path)
    out = export_harness(store.root, tmp_path / "out")

    assert (out / "harness.md").exists()
    assert (out / "summary.json").exists()
    assert (out / "evolution.jsonl").exists()


def test_the_exported_harness_separates_the_two_layers(tmp_path):
    """General vs topic is the harness's defining structure; an export that flattens it loses
    the thing the two-layer design exists to show."""
    store = seeded_store(tmp_path)
    text = (export_harness(store.root, tmp_path / "out") / "harness.md").read_text()

    assert "## General skills" in text
    assert "number_theory" in text
    assert "bound the search space first" in text
    assert "check small residues numerically" in text


def test_the_summary_counts_accepted_and_rejected_candidates_separately(tmp_path):
    store = seeded_store(tmp_path)
    summary = json.loads((export_harness(store.root, tmp_path / "out") / "summary.json").read_text())

    assert summary["edits"]["accepted"] == 2
    assert summary["edits"]["rejected"] == 1
    assert summary["skills"]["n_general"] == 1
    assert summary["skills"]["n_topic"] == 1


def test_the_summary_reports_per_skill_usage_and_the_injected_token_cost(tmp_path):
    """Skill usage frequency and injected-context token cost are both required deliverables."""
    store = seeded_store(tmp_path)
    summary = json.loads((export_harness(store.root, tmp_path / "out") / "summary.json").read_text())

    assert summary["usage"]["sk_0002"]["n_selected"] == 1
    assert summary["usage"]["sk_0002"]["n_selected_success"] == 1
    assert summary["usage"]["sk_0001"]["n_selected_success"] == 0
    assert summary["selection"]["total_injected_tokens"] == 145
    assert summary["selection"]["mean_injected_tokens"] == pytest.approx(72.5)


def test_a_rejected_candidate_keeps_its_source_problem_in_the_export(tmp_path):
    """The rejection record is the only trace a rejected candidate ever leaves."""
    store = seeded_store(tmp_path)
    out = export_harness(store.root, tmp_path / "out")
    rejections = [json.loads(line) for line in (out / "evolution.jsonl").read_text().splitlines()
                  if not json.loads(line)["accepted"]]

    assert len(rejections) == 1
    assert rejections[0]["source_problems"] == [9]


def test_export_of_an_empty_store_is_a_valid_empty_report(tmp_path):
    """The baseline arm never creates a store, but a frozen or wholly-rejecting run can leave an
    empty one; the export must produce a report rather than crash on the deliverable path."""
    store = SkillStore(tmp_path / "store", caps=Caps())
    out = export_harness(store.root, tmp_path / "out")

    summary = json.loads((out / "summary.json").read_text())
    assert summary["skills"]["n_general"] == 0
    assert summary["edits"]["accepted"] == 0
    assert (out / "harness.md").read_text().strip()


def test_adversarial_the_export_never_carries_a_ground_truth_answer(tmp_path):
    """The export is a shipped artifact -- the last place a leak would be noticed. Skills are
    guard-checked on the way in, but this asserts the export path itself adds no new channel
    (e.g. by dumping a raw candidate or a question text alongside the skill)."""
    store = seeded_store(tmp_path)
    out = export_harness(store.root, tmp_path / "out")

    blob = (out / "harness.md").read_text() + (out / "summary.json").read_text()
    assert "ground_truth" not in blob
    assert "Matches ground truth" not in blob


def test_adversarial_build_report_is_pure_and_touches_no_disk(tmp_path):
    """Separating the report from the writing is what lets the numbers be asserted directly and
    reused (README tables, slides) without a filesystem round-trip."""
    store = seeded_store(tmp_path)
    before = sorted(p.name for p in (tmp_path / "store").iterdir())
    report = build_report(store)
    assert report["skills"]["n_general"] == 1
    assert sorted(p.name for p in (tmp_path / "store").iterdir()) == before


def test_adversarial_a_selection_log_naming_a_deleted_skill_does_not_crash_the_export(tmp_path):
    """Usage is joined from the selection log onto live skills; a skill deleted by a later
    curator cycle still appears in earlier selection lines."""
    store = seeded_store(tmp_path)
    store.log_selection(problem_idx=13, topic="algebra", skill_ids=["sk_9999"], n_tokens=10, success=True)

    summary = json.loads((export_harness(store.root, tmp_path / "out") / "summary.json").read_text())
    assert summary["usage"]["sk_9999"]["n_selected"] == 1
    assert summary["usage"]["sk_9999"]["present_in_final_harness"] is False
    assert summary["usage"]["sk_0002"]["present_in_final_harness"] is True


def test_adversarial_export_is_rerunnable_over_the_same_destination(tmp_path):
    """Re-exporting after a longer run must overwrite, not append a second copy of the log."""
    store = seeded_store(tmp_path)
    export_harness(store.root, tmp_path / "out")
    out = export_harness(store.root, tmp_path / "out")

    assert len((out / "evolution.jsonl").read_text().strip().splitlines()) == 3


def test_provenance_survives_the_whole_chain_from_arm_to_export(tmp_path):
    """End-to-end for the provenance guarantee: the `p_<idx>` tag is attached by
    EvoHarnessArm.observe, carried through Reflect's candidate, bound by the curator, written by
    store.apply, and must still name the right problem in the exported bundle. Each link is unit
    tested; this asserts they actually compose.
    """
    from alphaapollo.core.harness.arms import EvoHarnessArm

    reflection = ("TOPIC: number theory\nSCOPE: topic\nTRIGGER: Counting under congruence constraints.\n"
                  "LESSON:\n- Enumerate a small range before generalizing.\n"
                  "AVOID: Extrapolating without numeric verification.")

    class ScriptedAgent:
        def __init__(self, replies):
            self.replies = list(replies)

        def get_action_from_gpt(self, obs):
            return self.replies.pop(0) if self.replies else "NO_PROPOSALS"

    arm = EvoHarnessArm(store_root=tmp_path / "store",
                        agent=ScriptedAgent([reflection, "ADD: 1\nREASON: useful", "NO_PATTERNS"]))
    problem = {"problem_idx": 42, "question": "Count integers divisible by seven.",
               "topic": "number_theory", "problem_shape": "counting-with-constraints"}
    result = {"pass_final": 0, "pass1_round0": 0, "final_answer_given": "412",
              "verifier_feedback": "The modular step is wrong.", "tool_errors": "",
              "reasoning_excerpt": "I assumed uniform residues.", "round_count": 3}

    arm.begin_batch(0)
    arm.system_prompt_for(problem)
    arm.observe(problem, result)
    arm.end_batch(0)

    out = export_harness(arm.store.root, tmp_path / "out")
    evolution = [json.loads(line) for line in (out / "evolution.jsonl").read_text().splitlines()]

    assert evolution, "the curator's decisions must reach the exported log"
    assert all(record["source_problems"] == [42] for record in evolution)
    # The accepted skill also keeps the tag in its own evidence, readable in harness.md.
    assert "p_42" in (out / "harness.md").read_text()
