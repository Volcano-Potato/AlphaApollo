import json

from alphaapollo.core.harness.schema import CandidateMemory, SkillEdit
from alphaapollo.core.harness.store import Caps, SkillStore

QUESTIONS = ["Find the number of ordered pairs of positive integers."]
GTS = ["738"]


def cand(n: int, scope: str = "topic", topic: str = "number_theory") -> CandidateMemory:
    return CandidateMemory(trigger=f"Trigger {n}.", lesson=f"- Lesson {n}.",
                           failure_mode=f"Avoid {n}.", scope_hint=scope,
                           topic=None if scope == "general" else topic,
                           evidence=[f"p_{n:04d}:symbolic_slip"])


def add(n: int, scope: str = "topic") -> SkillEdit:
    return SkillEdit(op="ADD", actor="topic_curator", reason=f"new {n}", payload=cand(n, scope))


def apply(store, edits):
    return store.apply(edits, problem_idx=1, batch=0, question_texts=QUESTIONS, ground_truths=GTS)


# ---------------------------------------------------------------------------
# Brief-specified tests (contract-level behavior: method/field names below are
# load-bearing for later tasks).
# ---------------------------------------------------------------------------


def test_add_creates_a_skill_and_logs_acceptance(tmp_path):
    store = SkillStore(tmp_path)
    results = apply(store, [add(1)])
    assert len(store.all()) == 1 and results[0]["accepted"] is True
    logged = json.loads((tmp_path / "harness_log.jsonl").read_text().strip())
    assert logged["op"] == "ADD" and logged["accepted"] is True


def test_add_beyond_topic_capacity_is_filtered_and_logged(tmp_path):
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=2))
    apply(store, [add(1), add(2)])
    results = apply(store, [add(3)])

    assert len(store.all()) == 2, "capacity must be a hard bound, not a prompt suggestion"
    assert results[0]["accepted"] is False
    assert results[0]["reject_reason"] == "capacity_full"


def test_general_and_topic_capacities_are_independent(tmp_path):
    store = SkillStore(tmp_path, caps=Caps(general=1, per_topic=1))
    apply(store, [add(1, "topic"), add(2, "general")])
    assert len(store.all()) == 2


def test_delete_then_add_within_one_batch_frees_a_slot(tmp_path):
    """两阶段 apply 的核心：同 batch 内 DELETE 腾出的槽位必须能被 ADD 占用。"""
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=1))
    apply(store, [add(1)])
    victim = store.all()[0].id

    apply(store, [add(2), SkillEdit(op="DELETE", actor="general_curator",
                                    reason="stale", skill_id=victim)])

    remaining = store.all()
    assert len(remaining) == 1
    assert remaining[0].id != victim, "the ADD should have taken the freed slot"


def test_revise_updates_content_and_records_the_problem_index(tmp_path):
    store = SkillStore(tmp_path)
    apply(store, [add(1)])
    sid = store.all()[0].id

    store.apply([SkillEdit(op="REVISE", actor="topic_curator", reason="sharpen",
                           skill_id=sid, payload=cand(99))],
                problem_idx=41, batch=5, question_texts=QUESTIONS, ground_truths=GTS)

    revised = store.all()[0]
    assert revised.id == sid and revised.lesson == "- Lesson 99." and 41 in revised.revised_at


def test_guard_rejection_does_not_mutate_the_store_but_is_logged(tmp_path):
    store = SkillStore(tmp_path)
    leaky = SkillEdit(op="ADD", actor="topic_curator", reason="leak",
                      payload=cand(1).__class__(trigger="T.", lesson="- The answer is 738.",
                                                failure_mode="A.", scope_hint="topic",
                                                topic="number_theory", evidence=[]))
    results = apply(store, [leaky])
    assert store.all() == [] and results[0]["reject_reason"] == "answer_leak"
    assert json.loads((tmp_path / "harness_log.jsonl").read_text().strip())["accepted"] is False


def test_skip_is_a_noop_but_still_logged(tmp_path):
    store = SkillStore(tmp_path)
    apply(store, [SkillEdit(op="SKIP", actor="topic_curator", reason="low confidence")])
    assert store.all() == []
    assert len((tmp_path / "harness_log.jsonl").read_text().strip().splitlines()) == 1


def test_every_edit_produces_exactly_one_log_line(tmp_path):
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=1))
    apply(store, [add(1), add(2), SkillEdit(op="SKIP", actor="x", reason="y")])
    assert len((tmp_path / "harness_log.jsonl").read_text().strip().splitlines()) == 3


def test_name_is_a_readable_slug_not_a_level_id_concatenation(tmp_path):
    """The reference implementation in the brief minted ``name=f"{level}-{sid}"`` (e.g.
    "topic-sk_0007"), which is pure noise both in the rendered system message (render.py
    emits "### {skill.name}") and in on-disk filenames used for README case studies. The
    name must instead be a readable slug derived from the trigger text."""
    store = SkillStore(tmp_path)
    edit = SkillEdit(op="ADD", actor="topic_curator", reason="new",
                     payload=CandidateMemory(trigger="Counting integers subject to divisibility conditions.",
                                             lesson="- Enumerate residues mod small primes.",
                                             failure_mode="Avoid double counting boundary cases.",
                                             scope_hint="topic", topic="number_theory", evidence=["p_0001:x"]))
    apply(store, [edit])
    skill = store.all()[0]
    assert skill.name != f"topic-{skill.id}"
    assert skill.id not in skill.name
    assert all(ch.islower() or ch.isdigit() or ch == "-" for ch in skill.name)
    assert skill.name.startswith("counting-integers")


# ---------------------------------------------------------------------------
# Self-designed adversarial cases (see task-5-report.md for narrative discussion of each).
# ---------------------------------------------------------------------------


def test_adversarial_shuffled_batch_delete_add_revise_and_skip_interleaved(tmp_path):
    """Attack: throw DELETE, ADD, REVISE and SKIP into one batch in an order that is
    deliberately NOT phase-sorted (ADD listed first, DELETE buried in the middle), on a
    per_topic cap of exactly 1. If the two-phase split were implemented by "process in
    list order" instead of "partition then process", this would incorrectly reject the
    ADD (capacity looks full at the time it's encountered)."""
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=1))
    apply(store, [add(1)])
    old_id = store.all()[0].id

    results = apply(store, [
        add(2),  # ADD listed first in input order
        SkillEdit(op="SKIP", actor="x", reason="noise"),
        SkillEdit(op="DELETE", actor="general_curator", reason="stale", skill_id=old_id),
    ])

    remaining = store.all()
    assert len(remaining) == 1 and remaining[0].id != old_id
    add_result = next(r for r in results if r["op"] == "ADD")
    assert add_result["accepted"] is True


def test_adversarial_add_count_exactly_at_cap_then_one_over_in_same_call(tmp_path):
    """Attack: submit exactly cap-many ADDs plus one extra, all in a single apply() call
    (not spread across calls, which is the easier case already covered by the brief's
    test). The occupancy check must be re-evaluated after each ADD within the same
    phase-2 loop, not computed once up front from a stale count."""
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=2))
    results = apply(store, [add(1), add(2), add(3)])
    accepted = [r["accepted"] for r in results]
    assert accepted == [True, True, False]
    assert results[2]["reject_reason"] == "capacity_full"
    assert len(store.all()) == 2


def test_adversarial_revise_and_delete_on_the_same_skill_in_one_batch(tmp_path):
    """Attack: a curator emits both REVISE and DELETE against the same skill_id in one
    batch (e.g. two independent per-topic/general curators disagreeing about a skill in
    the same cycle). Both ops fall into phase 1, so behavior is order-dependent on the
    input list; document exactly what happens rather than leaving it to accident.
    Here DELETE-then-REVISE: the REVISE must fail cleanly (unknown_skill_id), never raise,
    and the deletion must stick."""
    store = SkillStore(tmp_path)
    apply(store, [add(1)])
    sid = store.all()[0].id

    results = apply(store, [
        SkillEdit(op="DELETE", actor="general_curator", reason="stale", skill_id=sid),
        SkillEdit(op="REVISE", actor="topic_curator", reason="sharpen", skill_id=sid, payload=cand(99)),
    ])

    assert store.all() == []
    delete_result, revise_result = results
    assert delete_result["accepted"] is True
    assert revise_result["accepted"] is False
    assert revise_result["reject_reason"] == "unknown_skill_id"


def test_adversarial_revise_before_delete_in_input_order_still_ends_deleted(tmp_path):
    """Same attack, reversed input order: REVISE listed before DELETE. Since both are
    phase-1 ops processed in list order, the REVISE should succeed first and then the
    DELETE removes the (just-revised) skill anyway -- the end state is deletion either
    way, but the intermediate REVISE result differs, which is worth pinning down."""
    store = SkillStore(tmp_path)
    apply(store, [add(1)])
    sid = store.all()[0].id

    results = apply(store, [
        SkillEdit(op="REVISE", actor="topic_curator", reason="sharpen", skill_id=sid, payload=cand(99)),
        SkillEdit(op="DELETE", actor="general_curator", reason="stale", skill_id=sid),
    ])

    assert store.all() == []
    revise_result, delete_result = results
    assert revise_result["accepted"] is True
    assert delete_result["accepted"] is True


def test_adversarial_delete_of_unknown_skill_id_is_rejected_not_raised(tmp_path):
    """Attack: DELETE targeting a skill_id that was never created (typo'd id, or a
    curator racing against a rollback). Must not raise; must be logged as a rejection."""
    store = SkillStore(tmp_path)
    results = apply(store, [SkillEdit(op="DELETE", actor="x", reason="y", skill_id="sk_9999")])
    assert results[0]["accepted"] is False
    assert results[0]["reject_reason"] == "unknown_skill_id"
    assert store.all() == []


def test_adversarial_merge_with_no_payload_at_all_is_rejected_not_raised(tmp_path):
    """Attack: MERGE with payload=None (the curator forgot to attach a candidate, or a
    serialization bug dropped it). Must not raise (e.g. AttributeError on None.trigger);
    must be a clean, logged rejection distinct from the guard's own reject reasons."""
    store = SkillStore(tmp_path)
    apply(store, [add(1)])
    sid = store.all()[0].id

    results = apply(store, [SkillEdit(op="MERGE", actor="topic_curator", reason="combine", skill_id=sid, payload=None)])
    assert results[0]["accepted"] is False
    assert results[0]["reject_reason"] == "missing_payload"
    # the target skill must be untouched
    assert store.all()[0].lesson == "- Lesson 1."


def test_adversarial_merge_with_empty_evidence_list_still_updates_text_fields(tmp_path):
    """Attack (softer): MERGE with a *structurally present* but content-thin payload
    (evidence=[]). This is not rejected by the guard (empty evidence is not an
    "empty_section" the way an empty trigger/lesson/failure_mode would be) -- confirm
    the merge still goes through and existing evidence survives the union rather than
    being wiped out by an empty replacement."""
    store = SkillStore(tmp_path)
    apply(store, [add(1)])
    sid = store.all()[0].id
    original_evidence = list(store.all()[0].evidence)
    assert original_evidence  # sanity: add(1) seeds non-empty evidence

    thin = CandidateMemory(trigger="Refined trigger.", lesson="- Refined lesson.",
                           failure_mode="Refined avoid.", scope_hint="topic",
                           topic="number_theory", evidence=[])
    results = apply(store, [SkillEdit(op="MERGE", actor="topic_curator", reason="thin merge", skill_id=sid, payload=thin)])

    assert results[0]["accepted"] is True
    merged = store.all()[0]
    assert merged.lesson == "- Refined lesson."
    assert set(original_evidence) <= set(merged.evidence)


def test_adversarial_topic_scope_hint_with_missing_topic_is_rejected_not_a_crash(tmp_path):
    """Attack: a candidate declares scope_hint="topic" but leaves topic empty/None
    (CandidateMemory.__post_init__ only validates scope_hint/action_hint are in their
    enums -- it does NOT enforce the scope_hint=="topic" => topic-is-not-None invariant
    that Skill/_write_skill assumes). Materializing this naively would build a Skill
    with level="topic", topic=None and _write_skill's _validate_level_topic would raise
    ValueError, which -- absent a guard -- would propagate out of apply() and abort the
    whole batch, violating the "must never break the baseline" constraint."""
    store = SkillStore(tmp_path)
    bad = CandidateMemory(trigger="Trigger.", lesson="- Lesson.", failure_mode="Avoid.",
                          scope_hint="topic", topic=None, evidence=["p_0001:x"])
    results = apply(store, [SkillEdit(op="ADD", actor="topic_curator", reason="bad scope", payload=bad)])
    assert results[0]["accepted"] is False
    assert results[0]["reject_reason"] == "missing_topic"
    assert store.all() == []


def test_adversarial_write_skill_raising_mid_batch_does_not_abort_the_batch(tmp_path):
    """Attack: simulate an unexpected exception inside the write path (e.g. a disk error,
    or any bug we haven't anticipated) on the first of two ADDs. The batch must still
    produce one log line per edit and must still process the second edit -- a single
    bad edit must never break the underlying baseline run."""
    store = SkillStore(tmp_path)
    real_write_skill = store._write_skill
    call_count = {"n": 0}

    def flaky_write_skill(skill):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated disk failure")
        return real_write_skill(skill)

    store._write_skill = flaky_write_skill
    results = apply(store, [add(1), add(2)])

    assert len(results) == 2
    assert results[0]["accepted"] is False
    assert results[1]["accepted"] is True
    assert len(store.all()) == 1
    logged = [json.loads(line) for line in (tmp_path / "harness_log.jsonl").read_text().strip().splitlines()]
    assert len(logged) == 2


def test_adversarial_accepted_numeric_coincidence_guard_note_reaches_the_log(tmp_path):
    """Attack: a candidate whose lesson happens to contain a number equal to a ground
    truth, but without an assertion cue (so the guard accepts it with the advisory tag
    "numeric_coincidence" rather than rejecting it as answer_leak). The report needs an
    accurate count of how often this happens, so the tag must survive all the way into
    harness_log.jsonl's guard_note field, not just the in-memory return dict."""
    store = SkillStore(tmp_path)
    coincidental = CandidateMemory(
        trigger="Trigger.",
        lesson="- Enumerate candidates up to 738 and check each modulus.",
        failure_mode="Avoid.", scope_hint="topic", topic="number_theory", evidence=["p_0001:x"],
    )
    results = apply(store, [SkillEdit(op="ADD", actor="topic_curator", reason="coincidence", payload=coincidental)])

    assert results[0]["accepted"] is True
    assert results[0]["guard_note"] == "numeric_coincidence"
    logged = json.loads((tmp_path / "harness_log.jsonl").read_text().strip())
    assert logged["guard_note"] == "numeric_coincidence"
    assert len(store.all()) == 1
