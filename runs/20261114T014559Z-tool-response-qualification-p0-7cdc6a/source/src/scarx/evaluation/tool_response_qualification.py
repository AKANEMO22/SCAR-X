"""Descriptive data comparisons; never semantic truth, labels or live scoring."""

from collections import Counter
import json

COLLECTION_FIELDS = frozenset(("alarms", "events", "reminders", "emails", "messages", "users"))
GENERATED_IDS = {
    "AddAlarm": "alarm_id", "AddReminder": "reminder_id", "CreateEvent": "event_id",
    "SendEmail": "email_id", "SendMessage": "message_id",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def unordered_known(value, parent=None):
    if isinstance(value, dict):
        return {key: unordered_known(child, key) for key, child in value.items()}
    if isinstance(value, list):
        children = [unordered_known(child) for child in value]
        return sorted(children, key=canonical) if parent in COLLECTION_FIELDS else children
    return value


def classify(api, expected, actual):
    if canonical(expected) == canonical(actual):
        return {"category": "exact", "source_response_equivalent": True}
    if canonical(unordered_known(expected)) == canonical(unordered_known(actual)):
        return {"category": "known_collection_order_only",
                "source_response_equivalent": True,
                "note": "Record multiplicity preserved; says nothing about user intent."}
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return {"category": "response_key_schema_mismatch",
                    "source_response_equivalent": False,
                    "expected_keys": sorted(expected), "actual_keys": sorted(actual)}
        field = GENERATED_IDS.get(api)
        if (field and field in expected and expected[field] != actual[field]
                and isinstance(expected[field], str) and isinstance(actual[field], str)
                and canonical({k:v for k,v in expected.items() if k != field}) ==
                    canonical({k:v for k,v in actual.items() if k != field})):
            return {"category": "generated_identifier_only",
                    "source_response_equivalent": False,
                    "identifier_field": field,
                    "note": "Downstream binding not verified; do not turn into a label."}
        differences, extras = [], {}
        for key in expected:
            if key not in COLLECTION_FIELDS:
                if canonical(unordered_known(expected[key])) != canonical(unordered_known(actual[key])):
                    break
                continue
            if not isinstance(expected[key], list) or not isinstance(actual[key], list):
                break
            e = Counter(canonical(unordered_known(row)) for row in expected[key])
            a = Counter(canonical(unordered_known(row)) for row in actual[key])
            if e - a:
                break
            if a - e:
                differences.append(key)
                extras[key] = sum((a - e).values())
        else:
            if differences:
                return {"category": "actual_collection_strict_superset",
                        "source_response_equivalent": False,
                        "extra_record_counts": extras,
                        "note": "Intended filtering/source completeness requires independent review."}
    return {"category": "unresolved_payload_difference",
            "source_response_equivalent": False}
