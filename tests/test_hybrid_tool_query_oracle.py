import pytest
from scarx.evaluation.tool_query_oracle import query, multiset_equal

NOW = "2023-09-11 12:00:00"


def database():
    return {"Account": {"owner": {"username": "owner", "session_token": "owned"}},
            "Message": {"owner": {
                "a": {"message_id": "a", "timestamp": "2023-09-11 11:00:00", "sender": "x", "message": "red blue"},
                "b": {"message_id": "b", "timestamp": "2023-09-11 11:00:00", "sender": "y", "message": "red"},
                "c": {"message_id": "c", "timestamp": "2023-09-11 13:00:00", "sender": "x", "message": "red blue"},
            }},
            "Calendar": {"owner": {
                "c": {"event_id": "c", "start_time": "2023-09-11 08:00:00", "end_time": "2023-09-11 18:00:00"},
                "b": {"event_id": "b", "start_time": "2023-09-11 12:30:00", "end_time": "2023-09-11 13:00:00"},
                "a": {"event_id": "a", "start_time": "2023-09-11 13:00:01", "end_time": "2023-09-11 14:00:00"},
            }}}


def ask(db, api, args):
    return query(db, api, args, session_token="owned", now=NOW)


def test_any_all_sender_and_current_clock_exclude_future_records():
    a = ask(database(), "SearchMessages", {"query": "red blue"})
    assert [r["message_id"] for r in a["messages"]] == ["a", "b"]
    a = ask(database(), "SearchMessages", {"query": "red blue", "match_type": "all"})
    assert [r["message_id"] for r in a["messages"]] == ["a"]
    a = ask(database(), "SearchMessages", {"query": "red", "sender": "y"})
    assert [r["message_id"] for r in a["messages"]] == ["b"]


def test_calendar_containment_and_inclusive_boundary():
    a = ask(database(), "QueryCalendar", {"start_time": "2023-09-11 12:00:00", "end_time": "2023-09-11 13:00:00"})
    assert [r["event_id"] for r in a["events"]] == ["c", "b"]


def test_literal_keyword_wildcards_are_not_SQL_patterns():
    d = database()
    d["Message"]["owner"]["a"]["message"] = "a_b%"
    assert [r["message_id"] for r in ask(d, "SearchMessages", {"query": "_b%"})["messages"]] == ["a"]


def test_duplicate_multiset_records_and_wrong_field_do_not_match():
    assert not multiset_equal({"events": [{"id": "a"}]}, {"events": [{"id": "a"}, {"id": "a"}]})
    assert not multiset_equal({"events": [{"time": "12:00"}]}, {"events": [{"time": "13:00"}]})
    assert multiset_equal({"events": [{"id": "a"}, {"id": "b"}]}, {"events": [{"id": "b"}, {"id": "a"}]})


def test_unknown_principal_or_inverted_interval_is_rejected():
    with pytest.raises(ValueError):
        query(database(), "GetReminders", {}, session_token=None, now=NOW)
    with pytest.raises(ValueError):
        ask(database(), "QueryCalendar", {"start_time": "2023-09-11 14:00:00", "end_time": "2023-09-11 13:00:00"})


def test_five_most_recent_records_keep_tie_order():
    d = database()
    d["Message"]["owner"] = {str(i): {
        "message_id": str(i), "timestamp": "2023-09-11 11:00:00", "sender": "x", "message": "red"} for i in range(7)}
    assert [r["message_id"] for r in ask(d, "SearchMessages", {"query": "red"})["messages"]] == list("01234")
