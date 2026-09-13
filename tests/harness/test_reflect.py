import inspect

import pytest

from alphaapollo.core.harness import reflect as reflect_mod
from alphaapollo.core.harness.reflect import _strip_reasoning, build_reflect_context, normalise_topic, parse_reflection, reflect, sanitize_feedback
from alphaapollo.core.harness.schema import Skill


class StubAgent:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def get_action_from_gpt(self, obs):
        self.prompts.append(obs)
        return self.reply


GOOD = """ACTION: NEW
TOPIC: number theory
SCOPE: topic
TRIGGER: Counting integers under congruence constraints.
LESSON:
- Enumerate a small range in python before generalizing.
AVOID: Extrapolating without numeric verification."""

ENHANCE = """ACTION: ENHANCE
TARGET: sk_0003
TOPIC: number theory
SCOPE: topic
TRIGGER: Counting integers under congruence constraints.
LESSON:
- Also check the modulus boundary case.
AVOID: Assuming residues are uniform."""


def a_skill(sid="sk_0003"):
    return Skill(id=sid, name=f"topic-{sid}", level="topic", topic="number_theory", trigger="Counting integers divisible by a modulus.", lesson="- Enumerate first.", failure_mode="Skipping verification.")


def context(**kw):
    base = dict(existing_topics=["number_theory"], problem_shape="counting-with-constraints",
                final_answer_given="412", outcome="failed",
                verifier_feedback="The modular step is wrong.", tool_errors="",
                reasoning_excerpt="I assumed the residues were uniform.", round_count=3,
                related_skills=[a_skill()])
    base.update(kw)
    return build_reflect_context(**base)


def test_module_never_references_ground_truth():
    src = inspect.getsource(reflect_mod)
    assert "ground_truth" not in src, "reflect.py must not be able to touch the ground truth"


def test_context_builder_takes_no_question_or_answer_key():
    params = set(inspect.signature(build_reflect_context).parameters)
    assert "question" not in params and "ground_truth" not in params
    assert params == {"existing_topics", "problem_shape", "final_answer_given", "outcome", "verifier_feedback", "tool_errors", "reasoning_excerpt", "round_count", "related_skills"}


def test_prompt_shows_the_related_existing_skills():
    """Paper Appendix E.1 lists "related existing skills" as a proposal input;
    without them the solver keeps re-proposing what the harness already has."""
    agent = StubAgent(GOOD)
    reflect(agent, context())
    assert "sk_0003" in agent.prompts[0]
    assert "Counting integers divisible by a modulus." in agent.prompts[0]


def test_prompt_handles_an_empty_related_skill_list():
    agent = StubAgent(GOOD)
    reflect(agent, context(related_skills=[]))
    assert agent.prompts[0]


def test_prompt_carries_the_aggressive_filter_rules():
    """Each substring corresponds to one of paper Appendix E.1's four "filter aggressively"
    categories: generic advice, basic tool usage, exact task replay, and anything that would
    not generalize to an unseen problem. Checked as substrings that actually occur in
    REFLECT_PROMPT verbatim, not phrases invented independently of the prompt text."""
    agent = StubAgent(GOOD)
    reflect(agent, context())
    prompt = agent.prompts[0]
    for rule in ("generic advice", "basic tool usage", "exact replay", "never seen"):
        assert rule in prompt.lower()


def test_parse_reflection_reads_the_action_hint():
    assert parse_reflection(GOOD).action_hint == "NEW"


def test_parse_reflection_reads_enhance_with_its_target():
    cand = parse_reflection(ENHANCE)
    assert cand.action_hint == "ENHANCE" and cand.target_id == "sk_0003"


def test_parse_reflection_returns_none_on_action_none():
    assert parse_reflection("ACTION: NONE\nnothing reusable here") is None


def test_missing_action_line_defaults_to_new():
    legacy = GOOD.split("\n", 1)[1]
    assert parse_reflection(legacy).action_hint == "NEW"


def test_sanitize_strips_the_matches_gt_channel():
    raw = "Result: 7\nMatches ground truth: True\nMatches GT: False\nLooks fine."
    cleaned = sanitize_feedback(raw)
    assert "Matches ground truth" not in cleaned and "Matches GT" not in cleaned
    assert "Looks fine." in cleaned


def test_minimal_feedback_level_drops_the_verifier_report():
    agent = StubAgent(GOOD)
    reflect(agent, context(), feedback_level="minimal")
    assert "The modular step is wrong." not in agent.prompts[0]


def test_standard_feedback_level_includes_the_verifier_report():
    agent = StubAgent(GOOD)
    reflect(agent, context(), feedback_level="standard")
    assert "The modular step is wrong." in agent.prompts[0]


def test_parse_reflection_extracts_all_four_fields():
    cand = parse_reflection(GOOD)
    assert cand.scope_hint == "topic"
    assert cand.trigger.startswith("Counting integers")
    assert "Enumerate a small range" in cand.lesson
    assert cand.failure_mode.startswith("Extrapolating")


def test_parse_reflection_returns_none_on_explicit_none():
    assert parse_reflection("SCOPE: none\nnothing useful here") is None


def test_parse_reflection_returns_none_on_malformed_output():
    assert parse_reflection("I am not going to follow the format.") is None


def test_reflect_returns_none_when_the_model_output_is_unparseable():
    assert reflect(StubAgent("garbage"), context()) is None


def test_reflect_prompt_demands_english():
    agent = StubAgent(GOOD)
    reflect(agent, context())
    assert "English" in agent.prompts[0]


# ---------------------------------------------------------------------------
# Self-authored adversarial cases (task-8 brief, "自己设计至少 3 个对抗用例").
# ---------------------------------------------------------------------------


def test_lesson_bullet_containing_the_word_avoid_does_not_confuse_the_boundary():
    """A LESSON bullet that uses the word "avoid" mid-sentence (not as a line-start
    "AVOID:" label) must not be mistaken for the AVOID boundary -- only a line that
    literally starts with "AVOID:" may terminate the LESSON block."""
    text = """ACTION: NEW
TOPIC: number theory
SCOPE: topic
TRIGGER: Modular counting problems.
LESSON:
- Don't avoid checking the boundary residue; it is often the special case.
AVOID: Assuming every residue class behaves identically."""
    cand = parse_reflection(text)
    assert cand is not None
    assert "avoid checking the boundary residue" in cand.lesson
    assert cand.failure_mode == "Assuming every residue class behaves identically."


def test_explanatory_preamble_and_epilogue_around_the_format_block():
    """LLMs routinely wrap the requested format in chatty prose ("Sure, here is my
    reflection:" ... "Hope that helps!"). The parser must still find the fields anywhere
    in the text, not only when the format starts at column 0 of the whole string."""
    text = """Sure, here is my reflection on this attempt:

ACTION: NEW
SCOPE: general
TRIGGER: Problems that reduce to counting residues modulo a small number.
LESSON:
- Brute-force enumerate the modulus range before trusting a closed form.
AVOID: Trusting a closed-form residue count without a small numeric check.

Let me know if you would like more detail!"""
    cand = parse_reflection(text)
    assert cand is not None
    assert cand.scope_hint == "general"
    assert cand.trigger.startswith("Problems that reduce")
    assert cand.failure_mode.startswith("Trusting a closed-form")


def test_enhance_without_a_target_line_does_not_raise_and_leaves_target_id_none():
    """ACTION: ENHANCE with no TARGET line at all (model forgot to name the id) must
    not crash; target_id degrades gracefully to None rather than raising KeyError/etc."""
    text = """ACTION: ENHANCE
TOPIC: number theory
SCOPE: topic
TRIGGER: Counting integers under congruence constraints.
LESSON:
- Also check the modulus boundary case.
AVOID: Assuming residues are uniform."""
    cand = parse_reflection(text)
    assert cand is not None
    assert cand.action_hint == "ENHANCE"
    assert cand.target_id is None


def test_field_value_containing_a_colon_is_captured_in_full():
    """TRIGGER/AVOID values may themselves contain a colon (e.g. "Ratio problems: convert
    units first."); the parser must not truncate at the first colon inside the value."""
    text = """ACTION: NEW
TOPIC: number theory
SCOPE: topic
TRIGGER: Ratio problems: convert units before comparing quantities.
LESSON:
- Normalize units first: meters vs. centimeters is a common trap.
AVOID: Comparing raw numbers: mismatched units silently give a wrong ratio.
"""
    cand = parse_reflection(text)
    assert cand is not None
    assert cand.trigger == "Ratio problems: convert units before comparing quantities."
    assert cand.failure_mode == "Comparing raw numbers: mismatched units silently give a wrong ratio."


def test_first_of_multiple_scope_lines_wins():
    """If the model emits more than one SCOPE: line (e.g. one in a stray explanatory
    aside), the first occurrence is authoritative -- selection must be deterministic,
    not "whichever the regex engine happens to prefer"."""
    text = """ACTION: NEW
SCOPE: general
TRIGGER: Counting integers under congruence constraints.
LESSON:
- Enumerate a small range before generalizing. (this is a general SCOPE: topic tip)
AVOID: Extrapolating without numeric verification."""
    cand = parse_reflection(text)
    assert cand is not None
    assert cand.scope_hint == "general"


def test_empty_and_whitespace_only_input_returns_none():
    assert parse_reflection("") is None
    assert parse_reflection("   \n\n   \t  ") is None


def test_lesson_with_empty_body_between_header_and_avoid_is_incomplete():
    """LESSON: immediately followed by AVOID: with nothing in between (model skipped the
    bullet entirely) leaves lesson empty after stripping -- this counts as an incomplete
    candidate, not a candidate with an empty lesson string."""
    text = """ACTION: NEW
TOPIC: number theory
SCOPE: topic
TRIGGER: Counting integers under congruence constraints.
LESSON:
AVOID: Extrapolating without numeric verification."""
    assert parse_reflection(text) is None


def test_sanitize_feedback_case_variants_leading_space_and_duplicate_in_one_line():
    raw = "  matches ground truth: TRUE\nFoo. Matches GT: yes Matches GT: no\nGenuinely useful feedback line."
    cleaned = sanitize_feedback(raw)
    assert "matches ground truth" not in cleaned.lower()
    assert "matches gt" not in cleaned.lower()
    assert "Genuinely useful feedback line." in cleaned


def test_missing_action_line_defaults_to_new_when_action_value_is_unrecognized():
    """An ACTION: line that names something other than NEW/ENHANCE/NONE (a reasoning model
    drifting off the requested vocabulary, e.g. "MAYBE") is treated the same as a malformed
    reply -- None, not a silent coercion to NEW or ENHANCE. Pinned explicitly so a future
    change cannot casually start accepting arbitrary ACTION values."""
    text = """ACTION: MAYBE
TOPIC: number theory
SCOPE: topic
TRIGGER: Counting integers under congruence constraints.
LESSON:
- Enumerate a small range in python before generalizing.
AVOID: Extrapolating without numeric verification."""
    assert parse_reflection(text) is None


# ---------------------------------------------------------------------------
# Coordinator fix-round 1/5, Finding 1: a <think>...</think> reasoning-model
# prefix (accounting.install_accounting preserves upstream's practice of
# prepending this to every reply) must never be allowed to hijack parsing via
# re.search's first-match-by-position semantics.
# ---------------------------------------------------------------------------


def _wrap_think(reasoning: str, reply: str) -> str:
    return f"<think>\n{reasoning}\n</think>\n{reply}"


def test_strip_reasoning_removes_a_single_closed_think_block():
    assert _strip_reasoning("<think>blah blah</think>REST") == "REST"


def test_strip_reasoning_removes_multiple_think_blocks():
    text = "<think>one</think>middle<think>two</think>tail"
    assert _strip_reasoning(text) == "middletail"


def test_strip_reasoning_is_case_insensitive():
    assert _strip_reasoning("<THINK>blah</THINK>REST") == "REST"
    assert _strip_reasoning("<Think>blah</think>REST") == "REST"


def test_strip_reasoning_drops_everything_after_an_unclosed_think_tag():
    text = "<think>never closes and just keeps going ACTION: NEW"
    assert _strip_reasoning(text) == ""


def test_strip_reasoning_is_a_noop_with_no_think_tags():
    plain = "plain text, no think tags at all"
    assert _strip_reasoning(plain) == plain


def test_think_prefix_with_no_interfering_keywords_still_parses():
    """The case the coordinator's manual test labeled "clean think prefix" -- must keep
    working after the fix, not just the decoy cases below."""
    cand = parse_reflection(_wrap_think("Let me work through this problem carefully.", GOOD))
    assert cand is not None
    assert cand.action_hint == "NEW"
    assert cand.scope_hint == "topic"


def test_think_block_containing_a_decoy_scope_line_does_not_hijack_scope():
    """Before the fix: the think block's "SCOPE: general" line (line-anchored, exactly like the
    real answer's own SCOPE line) was the first match by position and silently won, filing what
    should be a topic-level candidate as general-level with no error anywhere. After the fix:
    the think block is gone before SCOPE is searched for at all, so the real "SCOPE: topic" line
    in the formatted answer is the only match."""
    reasoning = "Drafting my answer.\nSCOPE: general\nActually, rethinking this below."
    cand = parse_reflection(_wrap_think(reasoning, GOOD))
    assert cand is not None
    assert cand.scope_hint == "topic"


def test_think_block_containing_action_none_does_not_discard_the_real_candidate():
    """Before the fix: the think block's own "ACTION: NONE" line (line-anchored, exactly like the
    real answer's own ACTION line) was the first match by position, so the real candidate was
    silently dropped (reflect() would just look like "no candidate this round" with nothing in
    the log to explain why)."""
    reasoning = "Let me think about this.\nACTION: NONE\nWait, actually there is a good lesson here."
    cand = parse_reflection(_wrap_think(reasoning, GOOD))
    assert cand is not None
    assert cand.action_hint == "NEW"


def test_think_block_containing_action_none_maybe_does_not_discard_the_real_candidate():
    reasoning = "ACTION: NONE maybe? Let me reconsider before deciding."
    cand = parse_reflection(_wrap_think(reasoning, GOOD))
    assert cand is not None
    assert cand.action_hint == "NEW"


def test_reflect_end_to_end_strips_a_think_prefixed_reply():
    """Integration-level check that reflect() (not just parse_reflection() in isolation) sees
    the benefit: the StubAgent's reply carries a decoy-laden think prefix exactly like a real
    served reasoning model would produce via accounting.install_accounting."""
    reasoning = "Considering options.\nACTION: NONE\nSCOPE: general\nActually, finalizing the real answer below."
    agent = StubAgent(_wrap_think(reasoning, GOOD))
    cand = reflect(agent, context())
    assert cand is not None
    assert cand.action_hint == "NEW"
    assert cand.scope_hint == "topic"
    assert cand.topic == "number_theory"


def test_sanitize_feedback_strips_a_think_block_before_scrubbing_the_gt_channel():
    """The verifier's report text flows through the same Agent.get_action_from_gpt path as
    everything else, so it can carry the same <think> wrapper -- and that wrapper could itself
    contain (a paraphrase of) the answer-matching line."""
    raw = "<think>\nMatches ground truth: True\n</think>\nResult looks internally consistent."
    cleaned = sanitize_feedback(raw)
    assert "<think>" not in cleaned.lower()
    assert "Matches ground truth" not in cleaned
    assert "Result looks internally consistent." in cleaned


# --- the model names its own topic (paper Appendix E.1) ---------------------------------------
#
# The proposal step's inputs are "evaluation result, verifier details or rubric feedback,
# trajectory signals, compressed trajectory, and related existing skills" -- no topic -- and the
# instruction is "Propose a reusable skill. Choose a broad topic, decide NEW, ENHANCE, or NONE".
# An earlier version of this module had the topic as an INPUT copied from a dataset label, which
# manufactured a dependency on per-problem topic annotations the method does not actually have.


def test_the_prompt_never_tells_the_model_what_topic_the_problem_is():
    agent = StubAgent(GOOD)
    reflect(agent, context())
    prompt = agent.prompts[0]
    assert "Topic: number_theory" not in prompt
    assert "Choose" in prompt or "TOPIC:" in prompt


def test_the_topic_comes_off_the_reply_not_the_context():
    """The context lists `algebra` as the only existing topic; the reply says number theory. The
    reply must win -- the context's list is a reuse hint, not an assignment."""
    cand = reflect(StubAgent(GOOD), context(existing_topics=["algebra"]))
    assert cand.topic == "number_theory"


def test_existing_topics_are_offered_so_the_model_can_reuse_one():
    """Free-form naming fragments the buckets ("number theory" / "modular arithmetic" /
    "divisibility"), and since the harness caps skills per topic, every bucket ends up holding
    one skill and the layer stops being a layer."""
    agent = StubAgent(GOOD)
    reflect(agent, context(existing_topics=["number_theory", "geometry"]))
    assert "number_theory" in agent.prompts[0]
    assert "geometry" in agent.prompts[0]


def test_a_topic_scoped_candidate_without_a_topic_name_is_rejected():
    """It would have no bucket to live in; filing it under a placeholder would create a key that
    looks legitimate while matching nothing."""
    text = GOOD.replace("TOPIC: number theory\n", "")
    assert parse_reflection(text) is None


def test_a_general_candidate_keeps_the_topic_it_arose_in():
    """Algorithm 1 feeds the same proposal pool to CompileTaskType and the cross-task step, so a
    proposal made while doing number theory still belongs in that topic's curator call even when
    the model judged the lesson itself cross-task. `_bind_payloads` nulls it later for whatever
    the GeneralCurator accepts."""
    cand = parse_reflection(GOOD.replace("SCOPE: topic", "SCOPE: general"))
    assert cand.scope_hint == "general"
    assert cand.topic == "number_theory"


@pytest.mark.parametrize("raw,expected", [
    ("number theory", "number_theory"),
    ("Number Theory", "number_theory"),
    ("  NUMBER-THEORY  ", "number_theory"),
    ("number theory problems", "number_theory"),
    ("Combinatorics.", "combinatorics"),
])
def test_topic_names_are_canonicalised_onto_one_bucket_key(raw, expected):
    assert normalise_topic(raw) == expected


@pytest.mark.parametrize("raw", [
    "", "   ", "none", "N/A", "unknown", "general",
    "counting lattice paths under a divisibility constraint",
])
def test_unusable_topic_names_are_refused_rather_than_truncated(raw):
    """Past three words the model has described this one problem instead of naming a topic;
    truncating would yield a plausible-looking key that still matches nothing else."""
    assert normalise_topic(raw) is None


def test_adversarial_a_think_block_naming_a_decoy_topic_does_not_win():
    """Same hijack shape already proven for SCOPE/ACTION: a reasoning model rehearses topic names
    while deliberating, and first-match-by-position would pick the rehearsal over the answer."""
    text = "<think>\nTOPIC: geometry\nHmm, no.\n</think>\n" + GOOD
    assert parse_reflection(text).topic == "number_theory"


def test_adversarial_an_empty_harness_still_renders_a_usable_topic_section():
    agent = StubAgent(GOOD)
    reflect(agent, context(existing_topics=[]))
    assert "{existing_topics}" not in agent.prompts[0]
    assert "none yet" in agent.prompts[0].lower()


def test_adversarial_duplicate_and_blank_existing_topics_are_de_duplicated():
    agent = StubAgent(GOOD)
    reflect(agent, context(existing_topics=["number_theory", "Number Theory", "", None, "geometry"]))
    assert agent.prompts[0].count("- number_theory") == 1
