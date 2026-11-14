import copy
from scarx.evaluation.recovery_outcome_bank import cases, verify


def test_owned_taskbank_locks_interventions_but_never_calls_model_to_make_gold():
    bank = cases()
    assert bank == cases()
    assert len(bank) == 24 and len({r["id"] for r in bank}) == 24
    assert all(not r["case_is_finalTEST"] for r in bank)
    for case in bank:
        if case["kind"] == "time_format":
            assert len(case["initial_request"]["parameters"]["time"]) == 5
        if case["kind"] == "wrong_prior_time":
            assert case["initial_request"]["parameters"]["time"] != case["expected_time"]


def test_claimed_success_wrong_effect_duplicate_or_missingstatus_is_rejected():
    case = next(r for r in cases() if r["kind"] == "ready")
    before = {"Alarm": {"owner": {}}}
    after = {"Alarm": {"owner": {"owned-id": {"alarm_id": "owned-id", "time": case["expected_time"]}}}}
    decision = {"action": "finish", "status": "completed"}
    assert verify(case, before, after, "owner", decision)["verified_resolution"]
    assert not verify(case, before, before, "owner", decision)["verified_resolution"]
    assert not verify(case, before, after, "owner", {"action": "finish"})["verified_resolution"]
    wrong = copy.deepcopy(after)
    wrong["Alarm"]["owner"]["owned-id"]["time"] = "00:00:00"
    assert not verify(case, before, wrong, "owner", decision)["verified_resolution"]


def test_missing_input_escalation_is_separate_from_task_completion():
    case = next(r for r in cases() if r["kind"] == "missing_time")
    before = {"Alarm": {}}
    result = verify(case, before, before, "owner",
                    {"action": "ask", "status": "need_info", "message": "Bạn muốn lúc mấy giờ?"})
    assert result["verified_resolution"] and result["safe_escalation"]
    assert not result["goal_verified"] and not result["natural_language_support_fully_verified"]
