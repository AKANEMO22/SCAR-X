from scarx.evaluation.recovery_bank_bounds import finite_bound, simple_control


def row(ident, exception, schema, finish, retry):
    return {"case_id": ident, "initial_result": {"exception": exception},
            "outcomes": {k: {"verified_resolution": v} for k,v in (
                ("show_schema", schema), ("finish", finish), ("retry", retry))}}


def test_control_only_reads_actual_initial_exception():
    assert simple_control({"exception": None, "gold": "show_schema", "family": "fake"}) == "finish"
    assert simple_control({"exception": "Missing", "gold": "finish"}) == "show_schema"


def test_oracle_bound_does_not_count_unmeasured_success_or_create_learning():
    rows = [row("ok", None, False, True, False),
            row("repair", "BadFormat", False, False, True),
            row("unsolved", "Denied", False, False, False)]
    result = finite_bound(rows)
    assert result["simple_control_successes"] == 1
    assert result["best_measured_action_oracle_successes"] == 2
    assert result["max_gain_over_this_control"] == 1/3
    assert not result["control_frozen_before_formal_heldout_evaluation"]
