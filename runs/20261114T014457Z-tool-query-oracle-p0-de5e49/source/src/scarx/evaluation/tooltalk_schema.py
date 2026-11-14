"""Data-only ToolTalk schema normalization; not imported into serving."""

import copy


def normalize_tool_schema(document):
    value = copy.deepcopy(document)
    params = value.get("parameters")
    if not isinstance(params, dict) or params.get("type") != "object":
        raise ValueError("Expected object parameter schema")
    outer = value.pop("required", None)
    inner = params.get("required")
    if outer is not None and inner is not None and outer != inner:
        raise ValueError("Conflicting required declarations")
    required = inner if inner is not None else outer
    if required is not None:
        if (not isinstance(required, list)
                or not all(isinstance(k, str) for k in required)
                or len(set(required)) != len(required)
                or not set(required) <= set(params.get("properties", {}))):
            raise ValueError("Invalid required parameter declaration")
        params["required"] = required
    return value


def repair_pinned_calendar_enum(document):
    """Explicit adapted-harness patch for ToolTalk e05f4ce CreateEvent schema."""
    value = copy.deepcopy(document)
    if value.get("name") != "CreateEvent":
        return value, False
    properties = value["parameters"]["properties"]
    if (properties["name"].get("enum") != ["meeting", "event"]
            or "enum" in properties["event_type"]):
        raise ValueError("Pinned CreateEvent schema signature changed")
    properties["event_type"]["enum"] = properties["name"].pop("enum")
    return value, True
