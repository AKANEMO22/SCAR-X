import copy
import pytest
from scarx.evaluation.tooltalk_sandbox import bounded_arguments, canonical, verify_single_alarm_transition


def test_real_goal_contract_rejects_wrong_value_extra_effect_and_self_assertion():
    before = {"Alarm": {"owner": {}}, "Account": {"owner": {"role": "user"}}}
    after = copy.deepcopy(before)
    after["Alarm"]["owner"]["generated-id"] = {"alarm_id": "generated-id", "time": "18:30:00"}
    assert verify_single_alarm_transition(before, after, "owner", "18:30:00")["goal_verified"]
    assert not verify_single_alarm_transition(before, after, "owner", "09:00:00")["goal_verified"]
    after["Account"]["owner"]["role"] = "admin"
    assert not verify_single_alarm_transition(before, after, "owner", "18:30:00")["goal_verified"]
    assert not verify_single_alarm_transition(before, before, "owner", "18:30:00")["goal_verified"]


def test_alarm_identity_must_match_actual_record_key():
    before = {"Alarm": {}}
    after = {"Alarm": {"owner": {"key-a": {"alarm_id": "different", "time": "18:30:00"}}}}
    assert not verify_single_alarm_transition(before, after, "owner", "18:30:00")["goal_verified"]


def test_duplicate_alarms_do_not_count_as_one_recovered_goal():
    before = {"Alarm": {}}
    after = {"Alarm": {"owner": {
        "a": {"time": "18:30:00"}, "b": {"time": "18:30:00"},
    }}}
    assert not verify_single_alarm_transition(before, after, "owner", "18:30:00")["goal_verified"]


@pytest.mark.parametrize("value", [
    {"deep": [[[[[[[1]]]]]]]}, {"items": list(range(65))}, {"x": "a" * 4001},
    {"object": object()}, {i: str(i) for i in range(33)},
])
def test_tool_argument_boundaries_reject_unbounded_or_non_json_data(value):
    with pytest.raises(ValueError):
        bounded_arguments(value)


def test_nonfinite_json_is_not_accepted():
    with pytest.raises(ValueError):
        canonical({"value": float("nan")})
