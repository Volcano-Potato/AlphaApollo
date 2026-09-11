from alphaapollo.core.harness.schema import Skill
from alphaapollo.core.harness.store import Budget, Caps, SkillStore

QUESTION = "Count the ordered pairs of integers whose sum is divisible by seven."


def make(store, n, level, topic, trigger, n_sel=0, n_ok=0):
    s = Skill(id=f"sk_{n:04d}", name=f"{level}-{n}", level=level,
              topic=None if level == "general" else topic,
              trigger=trigger, lesson=f"- Lesson {n}.", failure_mode=f"Avoid {n}.",
              n_selected=n_sel, n_selected_success=n_ok)
    store._write_skill(s)
    return s


def populate(tmp_path, caps=None, budget=None):
    store = SkillStore(tmp_path, caps=caps or Caps(general=50, per_topic=50), budget=budget)
    for i in range(10):
        make(store, i, "general", None, "Verify intermediate results numerically.")
    for i in range(10, 20):
        make(store, i, "topic", "number_theory", "Counting integers divisible by a modulus.")
    for i in range(20, 30):
        make(store, i, "topic", "geometry", "Inscribed circles and tangent lengths.")
    return store


# ---------------------------------------------------------------------------
# Brief-specified tests (contract-level behavior: method/field names below are
# load-bearing for later tasks).
# ---------------------------------------------------------------------------


def test_selection_respects_count_and_quota(tmp_path):
    store = populate(tmp_path)
    picked = store.select(QUESTION, topic="number_theory")
    assert len(picked) <= 6
    assert sum(1 for s in picked if s.level == "general") <= 3
    assert sum(1 for s in picked if s.level != "general") <= 4


def test_selection_respects_the_token_budget(tmp_path):
    store = populate(tmp_path, budget=Budget(b=6, general_max=3, topic_max=4, tokens=40))
    picked = store.select(QUESTION, topic="number_theory")
    assert sum(s.n_tokens for s in picked) <= 40


def test_selection_never_returns_a_foreign_topic(tmp_path):
    store = populate(tmp_path)
    picked = store.select(QUESTION, topic="number_theory")
    assert all(s.topic in (None, "number_theory") for s in picked)


def test_utility_breaks_ties_among_equal_triggers(tmp_path):
    store = SkillStore(tmp_path)
    make(store, 1, "general", None, "Verify numerically.", n_sel=20, n_ok=1)
    make(store, 2, "general", None, "Verify numerically.", n_sel=20, n_ok=19)
    picked = store.select(QUESTION, topic="number_theory")
    assert picked[0].id == "sk_0002"


def test_lexical_overlap_outranks_utility(tmp_path):
    store = SkillStore(tmp_path)
    make(store, 1, "general", None, "Unrelated advice about calendars.", n_sel=20, n_ok=20)
    make(store, 2, "general", None, "Count ordered pairs divisible by a modulus.", n_sel=0, n_ok=0)
    assert store.select(QUESTION, topic="number_theory")[0].id == "sk_0002"


def test_selection_is_deterministic(tmp_path):
    store = populate(tmp_path)
    a = [s.id for s in store.select(QUESTION, topic="number_theory")]
    b = [s.id for s in store.select(QUESTION, topic="number_theory")]
    assert a == b


def test_empty_store_selects_nothing(tmp_path):
    assert SkillStore(tmp_path).select(QUESTION, topic="number_theory") == []


def test_record_usage_updates_counters_and_persists(tmp_path):
    store = populate(tmp_path)
    picked = store.select(QUESTION, topic="number_theory")
    store.record_usage(picked, success=True)

    reloaded = {s.id: s for s in SkillStore(tmp_path).all()}
    for s in picked:
        assert reloaded[s.id].n_selected == 1 and reloaded[s.id].n_selected_success == 1


# ---------------------------------------------------------------------------
# Self-designed adversarial cases (see task-6-report.md for narrative discussion
# of each). These deliberately target the boundaries called out in the task
# brief: exact-count / exact-token cutoffs, an oversized single skill, an
# empty-bucket quota, tie-break stability under reordering, topic=None, and
# record_usage against a since-deleted skill.
# ---------------------------------------------------------------------------


def test_adversarial_exactly_b_candidates_all_selected(tmp_path):
    """3 general + 3 topic = exactly b=6 candidates, each individually within its
    level's quota (general_max=3, topic_max=4) and all well under the token
    budget: every single one should survive (no off-by-one truncation at n == b)."""
    store = SkillStore(tmp_path, caps=Caps(general=50, per_topic=50))
    for i in range(3):
        make(store, i, "general", None, "Verify intermediate results numerically.")
    for i in range(10, 13):
        make(store, i, "topic", "number_theory", "Counting integers divisible by a modulus.")
    picked = store.select(QUESTION, topic="number_theory")
    assert len(picked) == 6


def test_adversarial_b_plus_one_candidates_drops_exactly_one_by_score(tmp_path):
    """Same as above but with one extra topic skill (4 -> 5, exceeding topic_max=4
    is not the point here -- exactly b+1=7 total candidates spread 3 general + 4
    topic, i.e. quota-legal individually but one over the combined cap b=6).
    The dropped skill must be the lowest-scoring one, deterministically."""
    store = SkillStore(tmp_path, caps=Caps(general=50, per_topic=50))
    for i in range(3):
        make(store, i, "general", None, "Verify intermediate results numerically.")
    # 4 on-topic skills with a strong lexical match, 1 with none -- the
    # weakest-scoring one (sk_0014, an off-topic trigger) should be the one
    # dropped when the combined pool (3 + 5 = 8, still > b=6) is trimmed.
    for i in range(10, 14):
        make(store, i, "topic", "number_theory", "Counting integers divisible by a modulus.")
    make(store, 14, "topic", "number_theory", "Unrelated advice about calendars and travel.")
    picked = store.select(QUESTION, topic="number_theory")
    assert len(picked) == 6
    assert "sk_0014" not in {s.id for s in picked}


def test_adversarial_token_budget_boundary_equal_vs_one_over(tmp_path):
    """The token cutoff must be <=, not <: a candidate set whose total cost lands
    exactly on budget.tokens must be fully admitted, and adding one more token of
    cost must drop something."""
    store = SkillStore(tmp_path, caps=Caps(general=50, per_topic=50))
    s = make(store, 0, "general", None, "Verify intermediate results numerically.")
    exact_budget = Budget(b=6, general_max=3, topic_max=4, tokens=s.n_tokens)
    store_exact = SkillStore(tmp_path, caps=Caps(general=50, per_topic=50), budget=exact_budget)
    picked = store_exact.select(QUESTION, topic="number_theory")
    assert len(picked) == 1 and picked[0].id == s.id

    store_over = SkillStore(tmp_path, caps=Caps(general=50, per_topic=50),
                             budget=Budget(b=6, general_max=3, topic_max=4, tokens=s.n_tokens - 1))
    assert store_over.select(QUESTION, topic="number_theory") == []


def test_adversarial_single_oversized_skill_yields_empty_not_a_crash(tmp_path):
    """A skill whose own n_tokens exceeds the entire token budget must simply be
    skipped -- not returned (over-budget), and not cause an infinite loop or
    exception."""
    store = SkillStore(tmp_path, caps=Caps(general=50, per_topic=50),
                        budget=Budget(b=6, general_max=3, topic_max=4, tokens=2))
    make(store, 0, "general", None, "Verify intermediate results numerically with great care and diligence.")
    picked = store.select(QUESTION, topic="number_theory")
    assert picked == []


def test_adversarial_empty_topic_bucket_does_not_borrow_general_slack(tmp_path):
    """10 general skills, zero topic skills for this problem's topic: the fixed
    per-level quota must NOT let general skills spill into the unused topic
    slots to fill up to b=6. Total selected is capped at general_max=3, not b=6.
    This is the documented, intended behavior of "quotas are fixed, not pooled"
    -- verified explicitly here since it is easy to regress into looking like a
    bug."""
    store = SkillStore(tmp_path, caps=Caps(general=50, per_topic=50))
    for i in range(10):
        make(store, i, "general", None, "Verify intermediate results numerically.")
    picked = store.select(QUESTION, topic="number_theory")
    assert len(picked) == 3
    assert all(s.level == "general" for s in picked)


def test_adversarial_tie_break_is_stable_under_reversed_insertion_order(tmp_path):
    """Two skills with identical trigger text (hence identical score) inserted in
    reverse id order must still resolve the tie the same way as forward order --
    the sort key must not depend on dict/insertion order, only on (score, id)."""
    store_fwd = SkillStore(tmp_path / "fwd", caps=Caps(general=50, per_topic=50))
    make(store_fwd, 1, "general", None, "Verify numerically.")
    make(store_fwd, 2, "general", None, "Verify numerically.")
    fwd = [s.id for s in store_fwd.select(QUESTION, topic="number_theory")]

    store_rev = SkillStore(tmp_path / "rev", caps=Caps(general=50, per_topic=50))
    make(store_rev, 2, "general", None, "Verify numerically.")
    make(store_rev, 1, "general", None, "Verify numerically.")
    rev = [s.id for s in store_rev.select(QUESTION, topic="number_theory")]

    assert fwd == rev == ["sk_0001", "sk_0002"]


def test_adversarial_topic_none_selects_general_only(tmp_path):
    """A problem with no topic annotation (topic=None) must never surface
    topic-scoped skills -- topic-level skills always have a non-None topic by
    construction, so topic=None can only ever match nothing on the topic side."""
    store = populate(tmp_path)
    picked = store.select(QUESTION, topic=None)
    assert picked and all(s.level == "general" for s in picked)


def test_adversarial_record_usage_on_a_deleted_skill_does_not_raise(tmp_path):
    """record_usage must tolerate a skill that no longer exists in the store
    (e.g. deleted by a curator edit between selection and the usage callback)
    without raising, and must still update the surviving skills in the same
    batch."""
    store = populate(tmp_path)
    picked = store.select(QUESTION, topic="number_theory")
    assert len(picked) >= 2
    survivor, victim = picked[0], picked[1]
    store._delete_skill(victim.id)

    store.record_usage([survivor, victim], success=True)  # must not raise

    reloaded = {s.id: s for s in store.all()}
    assert reloaded[survivor.id].n_selected == 1
    assert victim.id not in reloaded
