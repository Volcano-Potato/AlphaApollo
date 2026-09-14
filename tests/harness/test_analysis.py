import json

from alphaapollo.core.harness.analysis import (
    RunData,
    adaptation_curve,
    by_topic,
    compare_arms,
    cost,
    harness_growth,
    injected_context,
    load_run,
    overall,
    render,
    transfer_cases,
)


def problem_row(step, r0=0, final=0, error=0, topic="algebra"):
    return {"step": step, "adapt/pass1_round0": r0, "adapt/pass_final": final,
            "adapt/error": error, "topic": topic}


def write_run(tmp_path, name, rows, harness=None, accounting=None):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in rows]
    lines += [json.dumps(h) for h in (harness or [])]
    if accounting:
        lines.append(json.dumps(accounting))
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def make_run(rows, name="x"):
    run = RunData(name=name)
    run.problems = {r["step"]: r for r in rows}
    return run


# --- loading --------------------------------------------------------------------------------


def test_load_splits_the_three_kinds_of_row(tmp_path):
    d = write_run(tmp_path, "r",
                  [problem_row(0, final=1), problem_row(1)],
                  harness=[{"step": 1, "harness/n_general": 2, "harness/n_topic_total": 3}],
                  accounting={"step": 1, "calls/solver": 40, "calls/mgmt_side_total": 6})
    run = load_run(d)
    assert len(run.problems) == 2
    assert len(run.harness_series) == 1
    assert run.accounting["calls/solver"] == 40


def test_a_missing_run_directory_yields_an_empty_run_rather_than_raising(tmp_path):
    """Reporting mid-experiment, with only some arms finished, must work."""
    run = load_run(tmp_path / "never-ran")
    assert run.problems == {} and run.completed == set()


def test_a_corrupt_line_is_skipped_not_fatal(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "metrics.jsonl").write_text(json.dumps(problem_row(0, final=1)) + "\n{torn\n",
                                     encoding="utf-8")
    assert len(load_run(d).problems) == 1


# --- the completed/lost distinction ----------------------------------------------------------


def test_a_crashed_problem_is_excluded_from_every_rate():
    """It is not a wrong answer; counting it as one understates the arm."""
    run = make_run([problem_row(0, final=1), problem_row(1, final=0),
                    problem_row(2, error=1)])
    stats = overall(run)
    assert stats["n"] == 2 and stats["pass_final"] == 0.5 and stats["n_lost"] == 1


def test_arms_are_compared_on_the_problems_all_of_them_completed():
    """The arms run concurrently against one provider and lose different problems to the same
    rate-limit storm. Comparing their raw rates compares different problem sets."""
    runs = {
        "baseline": make_run([problem_row(0, final=1), problem_row(1, final=0),
                              problem_row(2, final=0)]),
        "evo": make_run([problem_row(0, final=1), problem_row(1, final=1),
                         problem_row(2, error=1)]),
    }
    comparison = compare_arms(runs)
    assert comparison["n_common"] == 2, "problem 2 died for evo, so it counts for neither"
    assert comparison["on_common"]["baseline"]["pass_final"] == 0.5
    assert comparison["on_common"]["evo"]["pass_final"] == 1.0
    assert comparison["dropped_by_arm"]["baseline"] == [2]


def test_per_arm_rates_over_their_own_sets_are_reported_alongside():
    """The two differing is itself a finding; reporting only the intersection hides it."""
    runs = {"a": make_run([problem_row(0, final=1), problem_row(1, error=1)]),
            "b": make_run([problem_row(0, final=0), problem_row(1, final=0)])}
    comparison = compare_arms(runs)
    assert comparison["on_own"]["a"]["pass_final"] == 1.0
    assert comparison["on_common"]["a"]["pass_final"] == 1.0
    assert comparison["on_own"]["b"]["pass_final"] == 0.0


def test_compare_arms_on_no_runs_at_all_is_harmless():
    assert compare_arms({"a": RunData(name="a")})["arms"] == {}


# --- the seven required results ---------------------------------------------------------------


def test_adaptation_curve_is_windowed_not_cumulative():
    """The question is whether the arm improves as it goes; a cumulative mean buries a late
    improvement under everything before it."""
    rows = [problem_row(i, final=0) for i in range(4)] + \
           [problem_row(i, final=1) for i in range(4, 8)]
    curve = adaptation_curve(make_run(rows), window=4)
    assert [c["window"] for c in curve] == ["0-3", "4-7"]
    assert [c["pass_final"] for c in curve] == [0.0, 1.0]


def test_adaptation_curve_reports_n_so_a_short_window_is_visible_as_short():
    rows = [problem_row(i, final=1) for i in range(4)] + [problem_row(4, final=0)]
    curve = adaptation_curve(make_run(rows), window=4)
    assert curve[1]["n"] == 1


def test_by_topic_reports_accuracy_not_harness_composition():
    """export.build_report's n_topic_by_topic counts *skills* per topic. This counts solved
    *problems* per topic -- a different required result that did not previously exist."""
    run = make_run([problem_row(0, final=1, topic="algebra"),
                    problem_row(1, final=0, topic="algebra"),
                    problem_row(2, final=1, topic="geometry")])
    breakdown = by_topic(run)
    assert breakdown["algebra"] == {"n": 2, "pass1_round0": 0.0, "pass_final": 0.5}
    assert breakdown["geometry"]["pass_final"] == 1.0


def test_unlabelled_topics_are_bucketed_rather_than_dropped():
    run = make_run([problem_row(0, final=1, topic="")])
    assert "(unlabelled)" in by_topic(run)


def test_cost_splits_solver_from_management_calls():
    run = make_run([problem_row(0)])
    run.accounting = {"calls/solver": 100, "calls/reflect": 8, "calls/topic_curator": 2,
                      "calls/mgmt_side_total": 10, "tokens/solver_in": 5, "tokens/solver_out": 7}
    c = cost(run)
    assert c["calls_solver"] == 100 and c["calls_mgmt"] == 10
    assert c["calls_by_role"]["reflect"] == 8
    assert c["tokens_in"] == 5 and c["tokens_out"] == 7


def test_unscoped_calls_are_surfaced_as_their_own_number():
    """A non-zero unscoped count is an instrumentation bug; folding it into either side would
    hide it, and it has been a real defect in this project before."""
    run = make_run([problem_row(0)])
    run.accounting = {"calls/solver": 10, "calls/unscoped": 3}
    assert cost(run)["calls_unscoped"] == 3


def test_harness_growth_is_ordered_by_stream_position():
    run = RunData(name="x")
    run.harness_series = [{"step": 15, "harness/n_general": 3, "harness/n_topic_total": 5,
                           "harness/total_tokens": 400, "harness/mean_skill_tokens": 50},
                          {"step": 7, "harness/n_general": 1, "harness/n_topic_total": 2,
                           "harness/total_tokens": 150, "harness/mean_skill_tokens": 50}]
    growth = harness_growth(run)
    assert [g["step"] for g in growth] == [7, 15]
    assert [g["n_general"] for g in growth] == [1, 3]


def test_injected_context_measures_what_entered_a_context_not_harness_size(tmp_path):
    """Early problems see an empty harness, so mean injected tokens is not the harness's size."""
    (tmp_path / "selection_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [
            {"problem_idx": 0, "n_tokens": 0}, {"problem_idx": 1, "n_tokens": 100},
            {"problem_idx": 2, "n_tokens": 200}]) + "\n", encoding="utf-8")
    stats = injected_context(tmp_path)
    assert stats["n_problems"] == 3 and stats["mean_tokens"] == 100.0
    assert stats["n_with_injection"] == 2 and stats["max_tokens"] == 200


def test_injected_context_without_a_selection_log_is_zero_not_an_error(tmp_path):
    assert injected_context(tmp_path)["n_problems"] == 0


# --- transfer cases ---------------------------------------------------------------------------


def make_selection_log(tmp_path, rows):
    (tmp_path / "selection_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


def test_transfer_cases_surface_both_directions(tmp_path):
    """A negative case is as required as a positive one, and strictly more interesting."""
    evo = make_run([problem_row(0, final=1), problem_row(1, final=0), problem_row(2, final=1)])
    base = make_run([problem_row(0, final=0), problem_row(1, final=1), problem_row(2, final=1)])
    store = make_selection_log(tmp_path, [
        {"problem_idx": 0, "skill_ids": ["sk_0001"], "n_tokens": 90},
        {"problem_idx": 1, "skill_ids": ["sk_0002"], "n_tokens": 80},
        {"problem_idx": 2, "skill_ids": ["sk_0003"], "n_tokens": 70}])
    cases = transfer_cases(evo, base, store)
    assert [c["problem_idx"] for c in cases["positive"]] == [0]
    assert [c["problem_idx"] for c in cases["negative"]] == [1]
    assert cases["positive"][0]["skill_ids"] == ["sk_0001"]


def test_a_problem_with_nothing_injected_is_not_a_transfer_case(tmp_path):
    """Without injection there was nothing to transfer, whichever way the outcome went."""
    evo = make_run([problem_row(0, final=1)])
    base = make_run([problem_row(0, final=0)])
    store = make_selection_log(tmp_path, [{"problem_idx": 0, "skill_ids": [], "n_tokens": 0}])
    cases = transfer_cases(evo, base, store)
    assert cases["positive"] == [] and cases["negative"] == []


def test_transfer_cases_point_at_the_trajectory_file_to_read(tmp_path):
    """The automated part is finding which problems are worth reading, not writing the case."""
    evo = make_run([problem_row(12, final=1)])
    base = make_run([problem_row(12, final=0)])
    store = make_selection_log(tmp_path, [{"problem_idx": 12, "skill_ids": ["sk_1"], "n_tokens": 5}])
    assert transfer_cases(evo, base, store)["positive"][0]["trajectory"] == \
        "trajectories/problem_0012.json"


def test_a_problem_only_one_arm_completed_is_never_a_transfer_case(tmp_path):
    evo = make_run([problem_row(0, final=1)])
    base = make_run([problem_row(0, error=1)])
    store = make_selection_log(tmp_path, [{"problem_idx": 0, "skill_ids": ["s"], "n_tokens": 5}])
    cases = transfer_cases(evo, base, store)
    assert cases["positive"] == [] and cases["negative"] == []


# --- rendering ---------------------------------------------------------------------------------


def test_render_covers_all_seven_required_results(tmp_path):
    evo = make_run([problem_row(i, final=i % 2, topic="algebra") for i in range(8)], "evo")
    evo.harness_series = [{"step": 7, "harness/n_general": 2, "harness/n_topic_total": 3,
                           "harness/total_tokens": 300, "harness/mean_skill_tokens": 60}]
    evo.accounting = {"calls/solver": 90, "calls/mgmt_side_total": 12, "tokens/solver_in": 1000}
    base = make_run([problem_row(i, final=0, topic="algebra") for i in range(8)], "baseline")
    store = make_selection_log(tmp_path, [
        {"problem_idx": i, "skill_ids": ["sk_0001"], "n_tokens": 100} for i in range(8)])

    out = render({"baseline": base, "evo": evo}, {}, {"evo": str(store)}, window=4)
    for heading in ("## 1. Adaptation stream", "## 2. Held-out set", "## 3. By topic",
                    "## 4. Harness size", "## 5. Injected context", "## 6. Skill usage",
                    "## 7. Transfer cases"):
        assert heading in out, f"missing required section: {heading}"
    assert "Not run yet" in out, "an unrun held-out phase must say so rather than show blanks"


def test_render_without_any_runs_does_not_crash(tmp_path):
    out = render({"evo": RunData(name="evo")}, {}, {}, window=4)
    assert "# Task C results" in out


# --- skill usage frequency ---------------------------------------------------------------------


def test_skill_usage_counts_injections_and_outcomes(tmp_path):
    from alphaapollo.core.harness.analysis import skill_usage
    make_selection_log(tmp_path, [
        {"problem_idx": 0, "skill_ids": ["sk_1", "sk_2"], "success": True},
        {"problem_idx": 1, "skill_ids": ["sk_1"], "success": False},
        {"problem_idx": 2, "skill_ids": [], "success": True}])
    usage = skill_usage(tmp_path)
    assert usage["sk_1"] == {"n_selected": 2, "n_selected_success": 1}
    assert usage["sk_2"] == {"n_selected": 1, "n_selected_success": 1}
    assert list(usage) == ["sk_1", "sk_2"], "ordered by frequency, so the tail is visible"


def test_skill_usage_counts_a_skill_a_later_curator_deleted(tmp_path):
    """It no longer exists to hold a counter, but its injections happened and cost tokens.
    Dropping them understates both usage and the harness's true injected-context cost."""
    from alphaapollo.core.harness.analysis import skill_usage
    make_selection_log(tmp_path, [{"problem_idx": 0, "skill_ids": ["sk_deleted"], "success": False}])
    assert skill_usage(tmp_path)["sk_deleted"]["n_selected"] == 1


def test_an_arm_that_never_ran_is_absent_from_the_cost_table_not_a_row_of_zeros(tmp_path):
    evo = make_run([problem_row(0, final=1)], "evo")
    evo.accounting = {"calls/solver": 10, "calls/mgmt_side_total": 2}
    out = render({"evo": evo, "raw": RunData(name="raw")}, {}, {}, window=4)
    cost_section = out.split("## 5.")[1].split("## 6.")[0]
    assert "| raw |" not in cost_section
