"""Finite observed action table bounds, not fitted policy or generalization proof."""


def simple_control(initial):
    # Uses reported outcome only. No taskfamily, gold, time or outcome table.
    return "show_schema" if initial.get("exception") is not None else "finish"


def finite_bound(rows, minimum_gain=.05):
    if not rows:
        raise ValueError("Expected measured action rows")
    successes, oracle, cover = 0, 0, []
    for row in rows:
        selected = simple_control(row["initial_result"])
        if selected not in row["outcomes"]:
            raise ValueError("Control action not measured")
        outcomes = [bool(v["verified_resolution"]) for v in row["outcomes"].values()]
        if not outcomes:
            raise ValueError("Missing measured action outcomes")
        control_success = bool(row["outcomes"][selected]["verified_resolution"])
        best_possible = any(outcomes)
        successes += control_success
        oracle += best_possible
        cover.append({"case_id": row["case_id"], "selected_action": selected,
                      "control_success": control_success, "any_measured_success": best_possible})
    count = len(rows)
    return {
        "cases": count, "simple_control_successes": successes,
        "best_measured_action_oracle_successes": oracle,
        "simple_control_rate": successes / count,
        "finite_oracle_rate": oracle / count,
        "max_gain_over_this_control": (oracle - successes) / count,
        "registered_minimum_gain": minimum_gain,
        "can_show_registered_gain_on_this_table": (oracle - successes) / count >= minimum_gain,
        "per_case": cover,
        "control_definition": "prototype: reported exception -> show_schema, else finish",
        "control_frozen_before_formal_heldout_evaluation": False,
        "scope": "Descriptive upper bound for measured actions/budgets only; not a trained policy or broad impossibility",
    }
