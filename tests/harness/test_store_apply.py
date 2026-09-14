import json

from alphaapollo.core.harness.schema import CandidateMemory, SkillEdit
from alphaapollo.core.harness.store import Caps, SkillStore

QUESTIONS = ["Find the number of ordered pairs of positive integers."]
GTS = ["738"]


def cand(n: int, scope: str = "topic", topic: str = "number_theory") -> CandidateMemory:
    # Evidence tags are exactly `p_<problem_idx>` -- the form `EvoHarnessArm.observe` actually
    # writes (arms.py) and the form found in real skill files on disk. An earlier version of this
    # helper used an invented `p_0001:symbolic_slip` shape that the system never produces, which
    # silently made `source_problems` empty in every test using it.
    # A general-scoped candidate carries two problems because that is what a general skill
    # legitimately rests on; a topic-scoped one needs only its own.
    evidence = [f"p_{n}"] if scope != "general" else [f"p_{n}", f"p_{n + 100}"]
    return CandidateMemory(trigger=f"Trigger {n}.", lesson=f"- Lesson {n}.",
                           failure_mode=f"Avoid {n}.", scope_hint=scope,
                           topic=None if scope == "general" else topic,
                           evidence=evidence)


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


def test_every_log_line_names_the_problems_that_produced_the_candidate(tmp_path):
    """`problem_idx` on a harness_log line is the BATCH index, not a problem's: apply() takes one
    value per call while its edits are pooled from every failed problem in the batch (see
    arms.py). So without this field the log cannot answer "what did problem N propose, and was it
    accepted?" -- which the assignment requires. Provenance rides in on the candidate's evidence
    tags, which _bind_payloads guarantees are present on every candidate-derived edit.
    """
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    edit = SkillEdit(op="ADD", actor="topic_curator", reason="useful",
                     payload=CandidateMemory(trigger="t", lesson="l", failure_mode="a",
                                             scope_hint="topic", topic="algebra",
                                             evidence=["p_3", "p_7"]))
    store.apply([edit], problem_idx=0, batch=0, question_texts=[], ground_truths=[])

    line = json.loads((tmp_path / "harness_log.jsonl").read_text().strip())
    assert line["source_problems"] == [3, 7]


def test_a_rejected_candidate_is_still_traceable_to_its_problem(tmp_path):
    """The rejection path is the one that matters most here -- an accepted skill keeps its
    evidence in the skill file, a rejected one exists nowhere but this log line."""
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    edit = SkillEdit(op="SKIP", actor="general_curator", reason="not referenced by the curator's reply",
                     payload=CandidateMemory(trigger="t", lesson="l", failure_mode="a",
                                             scope_hint="general", topic=None, evidence=["p_11"]))
    store.apply([edit], problem_idx=0, batch=0, question_texts=[], ground_truths=[])

    line = json.loads((tmp_path / "harness_log.jsonl").read_text().strip())
    assert line["accepted"] is False
    assert line["source_problems"] == [11]


def test_adversarial_an_edit_with_no_candidate_payload_logs_an_empty_provenance(tmp_path):
    """DELETE is harness maintenance, not a response to one candidate; it must log an empty list
    rather than omitting the key, so every line has the same shape for downstream analysis."""
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    store.apply([SkillEdit(op="DELETE", actor="general_curator", reason="stale", skill_id="sk_0001")],
                problem_idx=0, batch=0, question_texts=[], ground_truths=[])

    line = json.loads((tmp_path / "harness_log.jsonl").read_text().strip())
    assert line["source_problems"] == []


def test_adversarial_malformed_evidence_tags_do_not_break_logging(tmp_path):
    """Evidence is also where the guard records non-provenance notes; anything that is not a
    `p_<int>` tag must be ignored rather than crashing the audit write."""
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    edit = SkillEdit(op="ADD", actor="topic_curator", reason="useful",
                     payload=CandidateMemory(trigger="t", lesson="l", failure_mode="a",
                                             scope_hint="topic", topic="algebra",
                                             evidence=["p_3", "p_notanumber", "freeform note", "p_5"]))
    store.apply([edit], problem_idx=0, batch=0, question_texts=[], ground_truths=[])

    line = json.loads((tmp_path / "harness_log.jsonl").read_text().strip())
    assert line["source_problems"] == [3, 5]


# --- the general layer must actually be cross-problem ------------------------------------------


def _general_add(evidence, trigger="when counting pairs under a constraint"):
    return SkillEdit(op="ADD", actor="general_curator", reason="looks broad",
                     payload=CandidateMemory(trigger=trigger, lesson="- bound the search",
                                             failure_mode="brute force", scope_hint="general",
                                             topic=None, evidence=list(evidence)))


def test_a_general_skill_backed_by_a_single_problem_is_refused(tmp_path):
    """GENERAL_CURATOR_PROMPT already says "Each general skill must address a pattern seen in at
    least 2 (2+) different problems" -- and nothing enforced it, so the model ignored it. The
    first real 12-problem batch produced three general skills whose evidence was `[0]`, `[2]` and
    `[3]`: three single-problem lessons filed as cross-task patterns.

    Two things go wrong when that is allowed. The general layer stops being a *layer* -- it
    becomes a second copy of the topic layer, and indeed two of those three skills were
    verbatim duplicates of topic skills minted from the same candidate in the same batch. And
    both copies then compete for the injection budget, so the same text can be injected twice.

    Enforced in code rather than by asking the model again, for the same reason the token budget
    is: an instruction the model may ignore is not a constraint.
    """
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    [record] = store.apply([_general_add(["p_3"])], problem_idx=0, batch=0,
                           question_texts=[], ground_truths=[])

    assert record["accepted"] is False
    assert record["reject_reason"] == "general_needs_two_problems"
    assert store.all() == []


def test_a_general_skill_seen_in_two_problems_is_accepted(tmp_path):
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    [record] = store.apply([_general_add(["p_3", "p_7"])], problem_idx=0, batch=0,
                           question_texts=[], ground_truths=[])

    assert record["accepted"] is True
    assert len(store.all()) == 1


def test_the_same_problem_named_twice_is_still_one_problem(tmp_path):
    """Evidence is a list, and a candidate merged from two proposals of the same problem would
    otherwise satisfy the rule without any cross-problem support at all."""
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    [record] = store.apply([_general_add(["p_3", "p_3"])], problem_idx=0, batch=0,
                           question_texts=[], ground_truths=[])

    assert record["accepted"] is False
    assert record["reject_reason"] == "general_needs_two_problems"


def test_a_topic_skill_from_one_problem_is_still_fine(tmp_path):
    """A localized procedure learned from a single failure is exactly what the topic layer is
    for; this rule must not leak across to it."""
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    edit = SkillEdit(op="ADD", actor="topic_curator", reason="localized",
                     payload=CandidateMemory(trigger="t", lesson="- l", failure_mode="a",
                                             scope_hint="topic", topic="number_theory",
                                             evidence=["p_3"]))
    [record] = store.apply([edit], problem_idx=0, batch=0, question_texts=[], ground_truths=[])
    assert record["accepted"] is True


def test_adversarial_merging_into_a_general_skill_is_not_blocked(tmp_path):
    """MERGE adds a problem's evidence to a skill that already cleared the bar; re-checking the
    incoming payload alone would reject every legitimate reinforcement."""
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    store.apply([_general_add(["p_1", "p_2"])], problem_idx=0, batch=0,
                question_texts=[], ground_truths=[])
    target = store.all()[0].id

    merge = SkillEdit(op="MERGE", actor="general_curator", reason="same pattern again",
                      skill_id=target,
                      payload=CandidateMemory(trigger="t2", lesson="- l2", failure_mode="a2",
                                              scope_hint="general", topic=None, evidence=["p_9"]))
    [record] = store.apply([merge], problem_idx=0, batch=1, question_texts=[], ground_truths=[])
    assert record["accepted"] is True
    assert sorted(store.all()[0].evidence) == ["p_1", "p_2", "p_9"]


def test_adversarial_a_general_add_with_no_evidence_at_all_is_refused(tmp_path):
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    [record] = store.apply([_general_add([])], problem_idx=0, batch=0,
                           question_texts=[], ground_truths=[])
    assert record["accepted"] is False
    assert record["reject_reason"] == "general_needs_two_problems"


# --- a curator may only edit its own layer ------------------------------------------------------


def test_a_general_curator_cannot_rewrite_a_topic_skill(tmp_path):
    """Observed in a real run: GeneralCurator is shown only general skills, hallucinated the id of
    a topic skill, and `_apply_one` happily rewrote it -- because it looked the target up by id
    and never checked which layer it belonged to. Both curators then accumulated evidence and
    content into the same skill, so the two layers stopped being independent and the
    General-Only/Topic-Only ablation the paper reports would have been measuring one tangled
    layer.

    `_bind_payloads` already forces every payload's scope_hint to the issuing curator's own layer,
    so the payload is a reliable statement of who issued the edit.
    """
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    apply(store, [add(1, "topic")])
    topic_id = store.all()[0].id

    cross_layer = SkillEdit(op="MERGE", actor="general_curator", reason="hallucinated id",
                            skill_id=topic_id,
                            payload=CandidateMemory(trigger="OVERWRITTEN", lesson="- x",
                                                    failure_mode="y", scope_hint="general",
                                                    topic=None, evidence=["p_9", "p_10"]))
    [record] = apply(store, [cross_layer])

    assert record["accepted"] is False
    assert record["reject_reason"] == "wrong_layer"
    assert store.all()[0].trigger != "OVERWRITTEN", "the topic skill must be untouched"


def test_a_topic_curator_cannot_rewrite_a_general_skill(tmp_path):
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    apply(store, [add(2, "general")])
    general_id = store.all()[0].id

    cross_layer = SkillEdit(op="REVISE", actor="topic_curator", reason="hallucinated id",
                            skill_id=general_id,
                            payload=CandidateMemory(trigger="OVERWRITTEN", lesson="- x",
                                                    failure_mode="y", scope_hint="topic",
                                                    topic="number_theory", evidence=["p_9"]))
    [record] = apply(store, [cross_layer])

    assert record["accepted"] is False
    assert record["reject_reason"] == "wrong_layer"
    assert store.all()[0].trigger != "OVERWRITTEN"


def test_an_in_layer_merge_is_unaffected(tmp_path):
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    apply(store, [add(1, "topic")])
    topic_id = store.all()[0].id

    same_layer = SkillEdit(op="MERGE", actor="topic_curator", reason="same pattern",
                           skill_id=topic_id,
                           payload=CandidateMemory(trigger="REFINED", lesson="- x",
                                                   failure_mode="y", scope_hint="topic",
                                                   topic="number_theory", evidence=["p_9"]))
    [record] = apply(store, [same_layer])

    assert record["accepted"] is True
    assert store.all()[0].trigger == "REFINED"


def test_adversarial_a_topic_curator_cannot_hop_between_topics(tmp_path):
    """Same failure one level down: both skills are topic-scoped, but a curator invoked for
    `algebra` must not rewrite the `number_theory` bucket's skill."""
    store = SkillStore(tmp_path, caps=Caps(general=5, per_topic=5))
    apply(store, [add(1, "topic")])  # topic=number_theory
    victim = store.all()[0].id

    other_topic = SkillEdit(op="MERGE", actor="topic_curator", reason="wrong bucket",
                            skill_id=victim,
                            payload=CandidateMemory(trigger="OVERWRITTEN", lesson="- x",
                                                    failure_mode="y", scope_hint="topic",
                                                    topic="algebra", evidence=["p_9"]))
    [record] = apply(store, [other_topic])

    assert record["accepted"] is False
    assert record["reject_reason"] == "wrong_layer"
    assert store.all()[0].trigger != "OVERWRITTEN"


# --- candidate text survives the decision that rejected it ---------------------------------


def test_every_log_line_carries_the_candidate_text_behind_it(tmp_path):
    store = SkillStore(tmp_path / "s", caps=Caps(general=5, per_topic=5))
    apply(store, [add(1)])
    line = json.loads(store.harness_log.read_text().splitlines()[0])
    assert line["candidate"]["lesson"] == "- Lesson 1."
    assert line["candidate"]["trigger"] == "Trigger 1."
    assert line["candidate"]["failure_mode"] == "Avoid 1."
    assert line["candidate"]["scope_hint"] == "topic"
    assert line["candidate"]["evidence"] == ["p_1"]


def test_a_rejected_candidates_text_is_recorded_not_just_its_verdict(tmp_path):
    """This is the case the field exists for. An accepted candidate keeps its text in the skill
    file; a rejected one existed only in memory, and reconstructing it later costs a full re-run.
    """
    store = SkillStore(tmp_path / "s", caps=Caps(general=5, per_topic=1))
    apply(store, [add(1)])                      # fills the single topic slot
    records = apply(store, [add(2)])            # rejected: no room
    assert records[0]["accepted"] is False
    rejected = json.loads(store.harness_log.read_text().splitlines()[-1])
    assert rejected["accepted"] is False
    assert rejected["candidate"]["lesson"] == "- Lesson 2.", \
        "the text the curator proposed and the store refused must outlive the call"


def test_a_skipped_candidate_keeps_its_text_too(tmp_path):
    """SKIP is the curator declining a proposal -- the single most interesting rejection to be
    able to read back, since it is the curator's judgement rather than a capacity limit."""
    store = SkillStore(tmp_path / "s", caps=Caps(general=5, per_topic=5))
    apply(store, [SkillEdit(op="SKIP", actor="topic_curator", reason="too vague",
                            payload=cand(3))])
    line = json.loads(store.harness_log.read_text().splitlines()[0])
    assert line["op"] == "SKIP" and line["candidate"]["lesson"] == "- Lesson 3."


def test_an_edit_with_no_candidate_behind_it_logs_a_null_rather_than_omitting_the_key(tmp_path):
    """DELETE is harness maintenance, not a response to a proposal. Every line keeps one shape
    so downstream analysis never has to special-case a missing key."""
    store = SkillStore(tmp_path / "s", caps=Caps(general=5, per_topic=5))
    apply(store, [add(1)])
    apply(store, [SkillEdit(op="DELETE", actor="topic_curator", reason="stale",
                            skill_id="sk_0001")])
    line = json.loads(store.harness_log.read_text().splitlines()[-1])
    assert "candidate" in line and line["candidate"] is None


def test_the_candidate_record_never_carries_a_question_or_an_answer(tmp_path):
    """The log is an audit trail, not a second channel into the method -- but it is written from
    the same apply() call that receives question_texts and ground_truths for leak checking, so
    this pins that none of that leaks into the record."""
    store = SkillStore(tmp_path / "s", caps=Caps(general=5, per_topic=5))
    apply(store, [add(1)])
    line = store.harness_log.read_text()
    assert QUESTIONS[0] not in line and GTS[0] not in line
