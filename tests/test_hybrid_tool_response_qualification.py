from scarx.evaluation.tool_response_qualification import classify


def test_only_declared_collection_order_is_ignored():
    a, b = {"events": [{"id": "a"}, {"id": "b"}]}, {"events": [{"id": "b"}, {"id": "a"}]}
    assert classify("QueryCalendar", a, b)["category"] == "known_collection_order_only"
    assert classify("Other", {"sequence": [1, 2]}, {"sequence": [2, 1]})["category"] == "unresolved_payload_difference"


def test_multiplicity_and_extra_records_are_not_hidden():
    a, b = {"events": [{"id": "a"}]}, {"events": [{"id": "a"}, {"id": "a"}]}
    result = classify("QueryCalendar", a, b)
    assert result["category"] == "actual_collection_strict_superset"
    assert not result["source_response_equivalent"]
    assert result["extra_record_counts"] == {"events": 1}
    assert classify("QueryCalendar", b, a)["category"] == "unresolved_payload_difference"


def test_known_ids_are_descriptive_and_wrong_field_is_not_renamed():
    result = classify("AddAlarm", {"alarm_id": "a"}, {"alarm_id": "b"})
    assert result["category"] == "generated_identifier_only"
    assert not result["source_response_equivalent"]
    assert classify("AddReminder", {"alarm_id": "a"}, {"reminder_id": "a"})["category"] == "response_key_schema_mismatch"
    assert classify("Unknown", {"id": "a"}, {"id": "b"})["category"] == "unresolved_payload_difference"


def test_other_changed_fields_prevent_the_identifier_only_category():
    result = classify("AddAlarm", {"alarm_id": "a", "time": "18:30"},
                      {"alarm_id": "b", "time": "09:00"})
    assert result["category"] == "unresolved_payload_difference"


def test_boolean_is_not_coerced_to_numeric_in_exact_or_identifier_comparison():
    assert classify("Other", {"value": True}, {"value": 1})["category"] == "unresolved_payload_difference"
    result = classify("AddAlarm", {"alarm_id": "a", "valid": True},
                      {"alarm_id": "b", "valid": 1})
    assert result["category"] == "unresolved_payload_difference"
