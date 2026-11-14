import copy
import pytest
from scarx.evaluation.tooltalk_schema import normalize_tool_schema, repair_pinned_calendar_enum


def document():
    return {"name": "OwnedRead", "description": "Fixture",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            "required": ["path"]}


def test_required_is_nested_without_mutating_upstream_document():
    original = document()
    before = copy.deepcopy(original)
    repaired = normalize_tool_schema(original)
    assert repaired["parameters"]["required"] == ["path"]
    assert "required" not in repaired
    assert original == before
    assert normalize_tool_schema(repaired) == repaired


@pytest.mark.parametrize("value", ["path", ["absent"], ["path", "path"], [True]])
def test_invalid_required_contract_is_not_silently_weakened(value):
    original = document()
    original["required"] = value
    with pytest.raises(ValueError):
        normalize_tool_schema(original)


def test_conflicting_inner_and_outer_contract_is_rejected():
    original = document()
    original["parameters"]["required"] = []
    with pytest.raises(ValueError):
        normalize_tool_schema(original)


def test_pinned_calendar_enum_is_moved_without_restricting_event_name():
    original = {"name": "CreateEvent", "parameters": {
        "type": "object", "properties": {
            "name": {"type": "string", "enum": ["meeting", "event"]},
            "event_type": {"type": "string"},
        }}}
    repaired, changed = repair_pinned_calendar_enum(original)
    assert changed
    assert "enum" not in repaired["parameters"]["properties"]["name"]
    assert repaired["parameters"]["properties"]["event_type"]["enum"] == ["meeting", "event"]
    assert original["parameters"]["properties"]["name"]["enum"] == ["meeting", "event"]


def test_unknown_calendar_schema_is_not_silently_patched():
    original = {"name": "CreateEvent", "parameters": {
        "type": "object", "properties": {
            "name": {"type": "string"}, "event_type": {"type": "string"},
        }}}
    with pytest.raises(ValueError):
        repair_pinned_calendar_enum(original)
