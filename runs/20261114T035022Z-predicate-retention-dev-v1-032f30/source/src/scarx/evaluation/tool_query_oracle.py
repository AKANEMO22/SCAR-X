"""Independent SQL query-contract oracle, not natural-language intent scoring."""

from collections import Counter
from datetime import datetime
import json
import sqlite3

TOOLS = {
    "FindAlarms": ("Alarm", "alarms"), "GetReminders": ("Reminder", "reminders"),
    "QueryCalendar": ("Calendar", "events"), "SearchInbox": ("Email", "emails"),
    "SearchMessages": ("Message", "messages"),
}
FORMAT = "%Y-%m-%d %H:%M:%S"


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Expected ISO source datetime")
    return datetime.strptime(value, FORMAT).strftime(FORMAT)


def time_of_day(value):
    if not isinstance(value, str):
        raise ValueError("Expected source time")
    return datetime.strptime(value, "%H:%M:%S").strftime("%H:%M:%S")


def multiset_equal(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
        return False
    for key in left:
        if not isinstance(left[key], list) or not isinstance(right[key], list):
            return False
        dump = lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if Counter(map(dump, left[key])) != Counter(map(dump, right[key])):
            return False
    return True


def query(database, api, args, *, session_token, now):
    if api not in TOOLS or not isinstance(args, dict):
        raise ValueError("Unsupported read-query contract")
    actor = [row["username"] for row in database["Account"].values()
             if session_token is not None and row.get("session_token") == session_token]
    if len(actor) != 1:
        raise ValueError("Unauthenticated/ambiguous query principal")
    dbname, key = TOOLS[api]
    rows = list(database.get(dbname, {}).get(actor[0], {}).values())
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Invalid source records")
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE records(ord INTEGER PRIMARY KEY, raw TEXT, start TEXT, stop TEXT, sender TEXT, body TEXT, subject TEXT)")
        for index, row in enumerate(rows):
            if api == "FindAlarms":
                start = time_of_day(row["time"])
            elif api == "QueryCalendar":
                start = timestamp(row["start_time"])
            elif api == "SearchMessages":
                start = timestamp(row["timestamp"])
            elif api == "SearchInbox":
                start = timestamp(row["date"])
            else:
                start = None
            stop = timestamp(row["end_time"]) if api == "QueryCalendar" else None
            db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?)", (
                index, json.dumps(row, ensure_ascii=False), start, stop,
                row.get("sender"), row.get("message", row.get("body", "")),
                row.get("subject", "")))
        filters, params, order = [], [], " ORDER BY ord"
        if api == "QueryCalendar":
            begin, end = timestamp(args["start_time"]), timestamp(args["end_time"])
            if begin > end:
                raise ValueError("Inverted query interval")
            filters += ["start<=?", "stop>=?"]
            params += [end, begin]
        elif api == "FindAlarms":
            begin, end = args.get("start_range"), args.get("end_range")
            begin = time_of_day(begin) if begin is not None else None
            end = time_of_day(end) if end is not None else None
            if begin is not None and end is not None and begin > end:
                raise ValueError("Inverted query interval")
            if begin is not None:
                filters.append("start>=?")
                params.append(begin)
            if end is not None:
                filters.append("start<=?")
                params.append(end)
        elif api in ("SearchInbox", "SearchMessages"):
            now = timestamp(now)
            if not any(args.get(k) is not None for k in ("query", "sender", "start_date", "end_date")):
                raise ValueError("Missing search constraints")
            if args.get("match_type", "any") not in ("any", "all"):
                raise ValueError("Unknown search operator")
            filters.append("start<=?")
            params.append(now)
            if args.get("sender") is not None:
                filters.append("sender=?")
                params.append(args["sender"])
            begin, end = args.get("start_date"), args.get("end_date")
            begin = timestamp(begin) if begin is not None else None
            end = timestamp(end) if end is not None else None
            if begin is not None and end is not None and begin > end:
                raise ValueError("Inverted date filters")
            for value, operator in ((begin, ">="), (end, "<=")):
                if value is not None:
                    filters.append("start" + operator + "?")
                    params.append(value)
            keywords = args.get("query")
            if keywords is not None and not isinstance(keywords, str):
                raise ValueError("Invalid keyword query")
            words = keywords.lower().split() if keywords else []
            if words:
                clauses = []
                for word in words:
                    if api == "SearchInbox":
                        clauses.append("(instr(lower(body),?)>0 OR instr(lower(subject),?)>0)")
                        params.extend([word, word])
                    else:
                        clauses.append("instr(lower(body),?)>0")
                        params.append(word)
                connector = " AND " if args.get("match_type", "any") == "all" else " OR "
                filters.append("(" + connector.join(clauses) + ")")
            order = " ORDER BY start DESC,ord LIMIT 5"
        sql = "SELECT raw FROM records" + (" WHERE " + " AND ".join(filters) if filters else "") + order
        return {key: [json.loads(row[0]) for row in db.execute(sql, params)]}
