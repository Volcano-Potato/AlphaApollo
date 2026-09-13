from alphaapollo.core.harness.evolver import GeneralCurator, TopicCurator, _bind_payloads, parse_curator_output
from alphaapollo.core.harness.schema import CandidateMemory, Skill
from alphaapollo.core.harness.store import Caps


class StubAgent:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def get_action_from_gpt(self, obs):
        self.prompts.append(obs)
        return self.reply


class ExplodingAgent:
    def get_action_from_gpt(self, obs):
        raise RuntimeError("API is down")


def cand(n: int, scope="topic") -> CandidateMemory:
    return CandidateMemory(trigger=f"Trigger {n}.", lesson=f"- Lesson {n}.",
                           failure_mode=f"Avoid {n}.", scope_hint=scope,
                           topic="number_theory" if scope == "topic" else None,
                           evidence=[f"p_{n:04d}:slip"])


def skill(n: int, level="topic") -> Skill:
    return Skill(id=f"sk_{n:04d}", name=f"{level}-{n}", level=level,
                 topic=None if level == "general" else "number_theory",
                 trigger=f"T{n}.", lesson=f"- L{n}.", failure_mode=f"A{n}.")


ACCEPT = """ADD: 1
REASON: distinct and actionable"""

MERGE = """MERGE: 1 INTO sk_0003
REASON: overlaps existing guidance
TRIGGER: Counting integers under congruence constraints.
LESSON:
- Enumerate before generalizing.
AVOID: Skipping numeric verification."""


def test_parse_add_maps_to_the_candidate_index():
    edits = parse_curator_output(ACCEPT, actor="topic_curator")
    assert len(edits) == 1 and edits[0].op == "ADD" and edits[0].reason


def test_parse_merge_carries_the_target_id_and_new_content():
    edits = parse_curator_output(MERGE, actor="topic_curator")
    assert edits[0].op == "MERGE" and edits[0].skill_id == "sk_0003"
    assert "Enumerate before generalizing." in edits[0].payload.lesson


def test_parse_delete_needs_no_payload():
    edits = parse_curator_output("DELETE: sk_0004\nREASON: stale", actor="general_curator")
    assert edits[0].op == "DELETE" and edits[0].skill_id == "sk_0004" and edits[0].payload is None


def test_parse_skip_is_recorded_not_dropped():
    edits = parse_curator_output("SKIP: 1\nREASON: too vague", actor="topic_curator")
    assert edits[0].op == "SKIP"


def test_parse_no_proposals_yields_no_edits():
    assert parse_curator_output("NO_PROPOSALS", actor="topic_curator") == []
    assert parse_curator_output("NO_PATTERNS", actor="general_curator") == []


def test_parse_ignores_unparseable_noise():
    assert parse_curator_output("I think maybe we should keep things as they are.",
                                actor="topic_curator") == []


def test_topic_curator_binds_add_payload_to_the_right_candidate():
    agent = StubAgent(ACCEPT)
    edits = TopicCurator().curate(agent, existing=[], candidates=[cand(1), cand(2)],
                                  topic="number_theory", caps=Caps())
    assert edits[0].payload.lesson == "- Lesson 1."
    assert edits[0].payload.topic == "number_theory"


def test_topic_curator_declares_the_remaining_slots_in_the_prompt():
    agent = StubAgent("NO_PROPOSALS")
    TopicCurator().curate(agent, existing=[skill(1), skill(2)], candidates=[cand(1)],
                          topic="number_theory", caps=Caps(general=5, per_topic=5))
    assert "2/5" in agent.prompts[0]


def test_topic_curator_with_no_candidates_makes_no_model_call():
    agent = StubAgent(ACCEPT)
    assert TopicCurator().curate(agent, existing=[], candidates=[],
                                 topic="number_theory", caps=Caps()) == []
    assert agent.prompts == []


def test_general_curator_forces_general_scope_on_its_payloads():
    agent = StubAgent(ACCEPT)
    edits = GeneralCurator().curate(agent, existing=[], candidates=[cand(1, "topic")],
                                    caps=Caps())
    assert edits[0].payload.scope_hint == "general" and edits[0].payload.topic is None


def test_general_curator_requires_a_pattern_across_at_least_two_problems():
    agent = StubAgent("NO_PATTERNS")
    GeneralCurator().curate(agent, existing=[], candidates=[cand(1), cand(2)], caps=Caps())
    assert "2+" in agent.prompts[0] or "at least 2" in agent.prompts[0]


def test_topic_curator_prompt_carries_the_generalizability_test():
    """Paper Appendix E.2: apply a generalizability test such as usefulness
    for multiple unseen tasks.

    NOTE: the brief's own draft of this test asserted the substring "unseen", but
    TOPIC_CURATOR_PROMPT's verbatim text (mandated word-for-word, see the brief's
    "必须逐字采用" list) reads "problems you have never seen" -- "never seen", not
    "unseen". The two cannot both hold: "unseen" is not a substring of "never seen".
    Fixed here to check the phrase that actually appears in the mandated prompt text,
    confirmed identical to GENERAL_CURATOR_PROMPT's wording of the same test in intent
    (see the companion GeneralCurator prompt, which does use "unseen problems").
    """
    agent = StubAgent("NO_PROPOSALS")
    TopicCurator().curate(agent, existing=[], candidates=[cand(1)],
                          topic="number_theory", caps=Caps())
    assert "never seen" in agent.prompts[0].lower()
    assert "generalizability test" in agent.prompts[0].lower()


def test_general_curator_forbids_context_specific_references():
    """Paper Appendix E.3: general skills must avoid context-specific references."""
    agent = StubAgent("NO_PATTERNS")
    GeneralCurator().curate(agent, existing=[], candidates=[cand(1)], caps=Caps())
    assert "context-specific" in agent.prompts[0].lower()


def test_enhance_candidates_surface_their_target_in_the_prompt():
    agent = StubAgent("NO_PROPOSALS")
    enhancing = cand(1)
    enhancing.action_hint, enhancing.target_id = "ENHANCE", "sk_0042"
    TopicCurator().curate(agent, existing=[skill(42)], candidates=[enhancing],
                          topic="number_theory", caps=Caps())
    assert "ENHANCE" in agent.prompts[0] and "sk_0042" in agent.prompts[0]


def test_curator_degrades_to_noop_when_the_model_call_raises():
    """设计文档 §5.5:skill 更新失败必须降级为 no-op,不能破坏求解路径。"""
    assert TopicCurator().curate(ExplodingAgent(), existing=[], candidates=[cand(1)],
                                 topic="number_theory", caps=Caps()) == []
    assert GeneralCurator().curate(ExplodingAgent(), existing=[], candidates=[cand(1)],
                                   caps=Caps()) == []


# ---------------------------------------------------------------------------
# Think-block hijack regression (task-9 brief): the same reasoning-model
# <think>...</think> prefix that hijacked parse_reflection() in task 8 is even
# more dangerous here -- a decoy instruction head inside the think block would
# turn into a *real* SkillEdit against the live harness (e.g. a bogus DELETE).
# ---------------------------------------------------------------------------


def _wrap_think(reasoning: str, reply: str) -> str:
    return f"<think>\n{reasoning}\n</think>\n{reply}"


def test_think_block_containing_a_decoy_delete_does_not_hijack_the_real_instruction():
    reasoning = "Let me consider my options.\nDELETE: sk_0003\nActually, that seems too aggressive."
    wrapped = _wrap_think(reasoning, ACCEPT)
    edits = parse_curator_output(wrapped, actor="topic_curator")
    assert len(edits) == 1
    assert edits[0].op == "ADD"


def test_think_block_decoy_delete_does_hijack_when_not_stripped():
    """Sanity check that the attack is real: parsing the *unstripped* text (bypassing
    _strip_reasoning) picks up the decoy DELETE as the first instruction head, proving
    the regression test above is exercising a genuine defense and not a no-op. The decoy
    must sit on its own physical line (exactly like a reasoning model rehearsing a
    line-anchored field while thinking out loud) -- a decoy merely embedded mid-sentence
    would never match _HEAD's line-start anchor in the first place, which would make this
    "attack" fake from the start."""
    from alphaapollo.core.harness.evolver import _HEAD

    reasoning = "Let me consider my options.\nDELETE: sk_0003\nActually, that seems too aggressive."
    wrapped = _wrap_think(reasoning, ACCEPT)
    # Bypass the stripping step deliberately, mirroring what parse_curator_output would
    # do if it forgot to call _strip_reasoning first.
    raw_heads = _HEAD.findall(wrapped)
    assert raw_heads[0][0] == "DELETE" and raw_heads[0][1] == "sk_0003", (
        "expected the decoy DELETE inside the think block to be the first head match "
        "on the unstripped text -- if this fails, the attack scenario itself is stale"
    )


# ---------------------------------------------------------------------------
# Self-authored adversarial cases (task-9 brief, "自己设计至少 3 个对抗用例").
# ---------------------------------------------------------------------------


def test_instruction_head_inside_reason_text_is_not_a_new_instruction():
    """"REASON: better than a plain ADD: 1" must not be parsed as a second, nested
    instruction -- only a line that *starts* with one of the five verbs counts as a head."""
    text = "ADD: 2\nREASON: better than a plain ADD: 1"
    edits = parse_curator_output(text, actor="topic_curator")
    assert len(edits) == 1
    assert edits[0].op == "ADD" and edits[0].reason.strip() == "better than a plain ADD: 1"


def test_merge_into_a_malformed_target_is_still_parsed_structurally():
    """A malformed/nonexistent MERGE target is not this function's job to validate --
    that belongs to the store's apply() (unknown_skill_id rejection). parse_curator_output
    must not raise or silently drop the edit merely because the id looks odd."""
    text = "MERGE: 1 INTO not-a-real-id\nREASON: overlaps\nTRIGGER: T.\nLESSON:\n- L.\nAVOID: A."
    edits = parse_curator_output(text, actor="topic_curator")
    assert len(edits) == 1
    assert edits[0].op == "MERGE" and edits[0].skill_id == "not-a-real-id"


def test_same_candidate_referenced_by_both_add_and_skip_yields_both_edits_unfiltered():
    """parse_curator_output does not deduplicate/arbitrate conflicting references to the
    same candidate number -- that is intentionally left to the caller (or to the store,
    which processes edits independently); the parser's job is only to structurally decode
    what the model said, not to resolve contradictions."""
    text = "ADD: 1\nREASON: keep it\n\nSKIP: 1\nREASON: too vague"
    edits = parse_curator_output(text, actor="topic_curator")
    assert [e.op for e in edits] == ["ADD", "SKIP"]


def test_add_with_zero_candidate_number_is_out_of_range_and_dropped():
    """Candidate numbers are 1-based; 0 has no corresponding 0-based index and must be
    dropped rather than wrapping around to candidates[-1]."""
    agent = StubAgent("ADD: 0\nREASON: bogus index")
    edits = TopicCurator().curate(agent, existing=[], candidates=[cand(1)],
                                  topic="number_theory", caps=Caps())
    # The malformed ADD is dropped, and the candidate it failed to name is accounted for as a
    # recorded rejection rather than vanishing -- see the candidate-provenance tests below.
    assert [e.op for e in edits] == ["SKIP"]
    assert not any(e.op == "ADD" for e in edits)


def test_add_with_negative_candidate_number_is_dropped():
    agent = StubAgent("ADD: -1\nREASON: bogus index")
    edits = TopicCurator().curate(agent, existing=[], candidates=[cand(1)],
                                  topic="number_theory", caps=Caps())
    # The malformed ADD is dropped, and the candidate it failed to name is accounted for as a
    # recorded rejection rather than vanishing -- see the candidate-provenance tests below.
    assert [e.op for e in edits] == ["SKIP"]
    assert not any(e.op == "ADD" for e in edits)


def test_add_with_non_numeric_candidate_token_is_dropped():
    agent = StubAgent("ADD: one\nREASON: bogus index")
    edits = TopicCurator().curate(agent, existing=[], candidates=[cand(1)],
                                  topic="number_theory", caps=Caps())
    # The malformed ADD is dropped, and the candidate it failed to name is accounted for as a
    # recorded rejection rather than vanishing -- see the candidate-provenance tests below.
    assert [e.op for e in edits] == ["SKIP"]
    assert not any(e.op == "ADD" for e in edits)


def test_add_with_out_of_range_candidate_number_is_dropped_not_half_built():
    agent = StubAgent("ADD: 5\nREASON: only one candidate exists")
    edits = TopicCurator().curate(agent, existing=[], candidates=[cand(1)],
                                  topic="number_theory", caps=Caps())
    # The malformed ADD is dropped, and the candidate it failed to name is accounted for as a
    # recorded rejection rather than vanishing -- see the candidate-provenance tests below.
    assert [e.op for e in edits] == ["SKIP"]
    assert not any(e.op == "ADD" for e in edits)


def test_merge_missing_one_of_the_three_payload_sections_is_dropped():
    """MERGE/REVISE require TRIGGER + LESSON + AVOID; a reply missing AVOID entirely must
    not produce a half-built payload."""
    text = "MERGE: 1 INTO sk_0003\nREASON: overlaps\nTRIGGER: T.\nLESSON:\n- L."
    edits = parse_curator_output(text, actor="topic_curator")
    assert edits == []


def test_model_wraps_the_format_in_explanatory_prose():
    text = """Sure, here is my curation decision:

ADD: 1
REASON: distinct and actionable

Let me know if you would like anything else!"""
    edits = parse_curator_output(text, actor="topic_curator")
    assert len(edits) == 1 and edits[0].op == "ADD"


def test_reason_written_in_chinese_is_still_captured():
    text = "ADD: 1\nREASON: 这个候选是独立且可执行的"
    edits = parse_curator_output(text, actor="topic_curator")
    assert len(edits) == 1 and edits[0].op == "ADD"
    assert "独立" in edits[0].reason


# --- candidate provenance --------------------------------------------------------------------
#
# The assignment requires every problem's candidate updates and accept/reject decisions to be
# recorded. The first real end-to-end run showed three ways a candidate's source problem got
# lost on the way to harness_log.jsonl.


def _cand(text, evidence):
    return CandidateMemory(trigger=f"when {text}", lesson=f"do {text}", failure_mode=f"avoid {text}",
                           scope_hint="general", topic=None, evidence=list(evidence))


def test_revise_carries_the_provenance_of_the_candidates_that_prompted_it():
    """REVISE is the one op whose parsed payload has no candidate ordinal -- it is the curator
    reacting to the pool as a whole -- and _bind_payloads used to hand it `evidence=[]`, which
    made a REVISE untraceable to any problem. A real 4-problem run produced one.

    It is bound to every candidate in the pool, not one, because that is genuinely what the
    curator saw; claiming a single source would be a more precise lie.
    """
    candidates = [_cand("alpha", ["p_3"]), _cand("beta", ["p_7"])]
    edits = _bind_payloads(
        parse_curator_output("REVISE: sk_0002\nREASON: sharpen it\nTRIGGER: t\nLESSON: l\nAVOID: a", actor="general_curator"),
        candidates, level="general", topic=None)

    revises = [e for e in edits if e.op == "REVISE"]
    assert len(revises) == 1
    assert sorted(revises[0].payload.evidence) == ["p_3", "p_7"]
    # A REVISE claims no individual candidate, so both are still separately accounted for.
    assert sorted(e.payload.evidence for e in edits if e.op == "SKIP") == [["p_3"], ["p_7"]]


def test_a_candidate_the_curator_never_mentions_is_recorded_as_rejected():
    """store.py's own module docstring promises harness_log.jsonl holds "every candidate that was
    proposed but rejected". It did not: a candidate the curator simply failed to mention produced
    no edit, therefore no log line, therefore no trace that the problem ever proposed anything.

    Unreferenced candidates become explicit SKIPs so they travel the existing rejection path,
    with a reject reason that distinguishes them from a SKIP the curator actually asked for.
    """
    candidates = [_cand("alpha", ["p_3"]), _cand("beta", ["p_7"])]
    edits = _bind_payloads(
        parse_curator_output("ADD: 1\nREASON: useful", actor="general_curator"),
        candidates, level="general", topic=None)

    ops = [(e.op, e.payload.evidence) for e in edits]
    assert ("ADD", ["p_3"]) in ops
    dropped = [e for e in edits if e.op == "SKIP"]
    assert len(dropped) == 1
    assert dropped[0].payload.evidence == ["p_7"]
    assert "not referenced" in dropped[0].reason


def test_an_explicit_skip_is_not_relabelled_as_unreferenced():
    candidates = [_cand("alpha", ["p_3"])]
    edits = _bind_payloads(
        parse_curator_output("SKIP: 1\nREASON: too specific", actor="general_curator"),
        candidates, level="general", topic=None)
    assert len(edits) == 1
    assert edits[0].op == "SKIP"
    assert "not referenced" not in edits[0].reason
    assert edits[0].payload.evidence == ["p_3"]


def test_adversarial_every_candidate_referenced_means_no_synthetic_skips():
    candidates = [_cand("alpha", ["p_3"]), _cand("beta", ["p_7"])]
    edits = _bind_payloads(
        parse_curator_output("ADD: 1\nREASON: a\n\nADD: 2\nREASON: b", actor="general_curator"),
        candidates, level="general", topic=None)
    assert [e.op for e in edits] == ["ADD", "ADD"]


def test_adversarial_an_unresolvable_ordinal_does_not_consume_a_candidate():
    """An out-of-range ordinal is dropped by the binder. The candidates it did not name must
    still be accounted for rather than silently vanishing alongside it."""
    candidates = [_cand("alpha", ["p_3"])]
    edits = _bind_payloads(
        parse_curator_output("ADD: 9\nREASON: hallucinated ordinal", actor="general_curator"),
        candidates, level="general", topic=None)
    assert [e.op for e in edits] == ["SKIP"]
    assert edits[0].payload.evidence == ["p_3"]


def test_adversarial_merge_provenance_survives_binding():
    candidates = [_cand("alpha", ["p_3"])]
    edits = _bind_payloads(
        parse_curator_output("MERGE: 1 INTO sk_0002\nREASON: overlaps\nTRIGGER: t\nLESSON: l\nAVOID: a", actor="general_curator"),
        candidates, level="general", topic=None)
    assert [e.op for e in edits] == ["MERGE"]
    assert edits[0].payload.evidence == ["p_3"]


# --- a general ADD must be able to span the problems it generalises from -----------------------
#
# Found by running: after the "a general skill needs 2+ problems" rule landed, a 12-problem smoke
# rejected EVERY general ADD (9 of 9, all `general_needs_two_problems`) and the general layer came
# back empty. The rule was right; the mechanism could not satisfy it. `ADD: <candidate number>`
# takes ONE ordinal and copies that candidate's text verbatim, and each Reflect call yields one
# candidate carrying exactly one `p_<idx>` tag -- so a general ADD's evidence was always a single
# problem, structurally.
#
# That also exposed the deeper defect the rule had merely revealed: a curator told to "distil
# patterns ACROSS different topics" had no way to write a synthesis. Its only ADD form copied one
# candidate unchanged.


def test_an_add_may_cite_several_candidates_and_unions_their_evidence():
    candidates = [_cand("alpha", ["p_3"]), _cand("beta", ["p_7"]), _cand("gamma", ["p_9"])]
    edits = _bind_payloads(
        parse_curator_output("ADD: 1,3\nREASON: same failure in both", actor="general_curator"),
        candidates, level="general", topic=None)

    adds = [e for e in edits if e.op == "ADD"]
    assert len(adds) == 1
    assert sorted(adds[0].payload.evidence) == ["p_3", "p_9"]


def test_a_multi_candidate_add_may_carry_the_curators_own_synthesis():
    """Copying one candidate's wording is not distillation. When the curator supplies
    TRIGGER/LESSON/AVOID they win, which is what lets a general skill say something neither
    source candidate said."""
    candidates = [_cand("alpha", ["p_3"]), _cand("beta", ["p_7"])]
    reply = ("ADD: 1,2\nREASON: both mis-handled bounds\n"
             "TRIGGER: When a search space looks unbounded.\n"
             "LESSON:\n- Bound the range before enumerating.\nAVOID: Enumerating an open range.")
    edits = _bind_payloads(parse_curator_output(reply, actor="general_curator"),
                           candidates, level="general", topic=None)

    add = [e for e in edits if e.op == "ADD"][0]
    assert add.payload.trigger == "When a search space looks unbounded."
    assert "Bound the range before enumerating." in add.payload.lesson
    assert add.payload.failure_mode == "Enumerating an open range."
    assert sorted(add.payload.evidence) == ["p_3", "p_7"]


def test_a_single_candidate_add_still_copies_that_candidate():
    """The topic layer's normal case: one localized procedure learned from one failure."""
    candidates = [_cand("alpha", ["p_3"])]
    edits = _bind_payloads(parse_curator_output("ADD: 1\nREASON: useful", actor="topic_curator"),
                           candidates, level="topic", topic="number_theory")

    add = [e for e in edits if e.op == "ADD"][0]
    assert add.payload.trigger == "when alpha"
    assert add.payload.evidence == ["p_3"]


def test_every_cited_candidate_counts_as_referenced():
    """A candidate named inside a multi-ordinal ADD must not also be emitted as an unreferenced
    SKIP -- that would log it as both accepted and rejected."""
    candidates = [_cand("alpha", ["p_3"]), _cand("beta", ["p_7"])]
    edits = _bind_payloads(parse_curator_output("ADD: 1,2\nREASON: both", actor="general_curator"),
                           candidates, level="general", topic=None)
    assert [e.op for e in edits] == ["ADD"]


def test_adversarial_duplicate_and_out_of_range_ordinals_in_one_add():
    """`ADD: 1,1,9` -- a repeat must not double-count evidence, and a bad ordinal must not
    invalidate the whole edit when a good one is present."""
    candidates = [_cand("alpha", ["p_3"])]
    edits = _bind_payloads(parse_curator_output("ADD: 1,1,9\nREASON: x", actor="general_curator"),
                           candidates, level="general", topic=None)
    add = [e for e in edits if e.op == "ADD"][0]
    assert add.payload.evidence == ["p_3"]


def test_adversarial_an_add_naming_only_bad_ordinals_is_dropped():
    candidates = [_cand("alpha", ["p_3"])]
    edits = _bind_payloads(parse_curator_output("ADD: 8,9\nREASON: x", actor="general_curator"),
                           candidates, level="general", topic=None)
    assert not any(e.op == "ADD" for e in edits)
    assert [e.op for e in edits] == ["SKIP"], "the unnamed candidate is still accounted for"


def test_the_general_curator_is_shown_how_to_cite_several_candidates():
    """The capability is useless if the prompt never mentions it."""
    agent = StubAgent("NO_PATTERNS")
    GeneralCurator().curate(agent, existing=[], candidates=[cand(1), cand(2)], caps=Caps())
    prompt = agent.prompts[0]
    assert "ADD: <candidate numbers" in prompt or "comma" in prompt.lower()
