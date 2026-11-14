"""Query counterfactuals over fixed candidate records; never a serving encoder."""

import copy
import random
from .predicate_retention_dev import (
    FAMILIES, DBS, KEYS, ID_FIELDS, CLOCK, DAY, clock, independent_label)
from .tool_query_oracle import query

WORDS = ("red", "blue", "green", "yellow")


def seconds_integer(value):
    h, m, s = map(int, value.split(" ")[-1].split(":"))
    return h * 3600 + m * 60 + s


def positive_query(case):
    """Construct an actual query including this record, with no neural flag."""
    row, api = case["record"], case["api"]
    if api == "FindAlarms":
        return {"start_range": row["time"], "end_range": row["time"]}
    if api == "QueryCalendar":
        point = (seconds_integer(row["start_time"]) + seconds_integer(row["end_time"])) // 2
        when = DAY + " " + clock(point)
        return {"start_time": when, "end_time": when}
    if api == "SearchMessages":
        if row["timestamp"] > CLOCK:
            return None
        return {"query": row["message"].split()[0], "match_type": "any", "sender": row["sender"]}
    if api == "GetReminders":
        return {}
    raise ValueError("Unknown parent family")


def materialize(case, args, owner):
    result = copy.deepcopy(case)
    result["args"], result["owner"] = args.copy(), owner
    record_owner, api = result["record_owner"], result["api"]
    accounts = {owner: {"username": owner, "session_token": "owned-token"}}
    if owner != record_owner:
        accounts[record_owner] = {"username": record_owner, "session_token": "other-token"}
    result["source_state"] = {
        "Account": accounts,
        DBS[api]: {record_owner: {"record": copy.deepcopy(result["record"])}}}
    rows = query(result["source_state"], api, args, session_token="owned-token", now=CLOCK)[KEYS[api]]
    result["target"] = int(any(r[ID_FIELDS[api]] == "record" for r in rows))
    if result["target"] != independent_label(result):
        raise AssertionError("SQL/Python membership disagrees")
    return result


def build_pairs(rows):
    pairs, exclusions = [], []

    def add(case, kind, positive, negative):
        if positive["target"] != 1 or negative["target"] != 0:
            raise AssertionError("Counterfactual did not flip membership")
        if positive["record"] != negative["record"] or positive["record"] != case["record"]:
            raise AssertionError("Candidate record changed")
        pairs.append({"id": case["id"] + ":" + kind, "parent_id": case["id"],
                      "contrast": kind, "api": case["api"],
                      "include": positive, "exclude": negative})

    for case in rows:
        api, row, owner = case["api"], case["record"], case["record_owner"]
        if api not in FAMILIES:
            raise ValueError("Unexpected source family")
        args = positive_query(case)
        if args is None:
            exclusions.append({"id": case["id"], "contrast": "owner/keyword",
                               "reason": "candidate timestamp after fixed source clock"})
            continue
        included = materialize(case, args, owner)
        rng = random.Random("binding-v1/" + case["id"])
        other = owner
        while other == owner:
            other = "person" + "".join(rng.choices("abcdefghkmnprstuvxyz", k=7))
        add(case, "owner", included, materialize(case, args, other))
        if api in ("FindAlarms", "QueryCalendar"):
            if api == "FindAlarms":
                point = 0 if seconds_integer(row["time"]) else 86399
                excluded_args = {"start_range": clock(point), "end_range": clock(point)}
            else:
                point = seconds_integer(row["end_time"]) + 1
                if point > 86399:
                    point = seconds_integer(row["start_time"]) - 1
                if point < 0:
                    exclusions.append({"id": case["id"], "contrast": "time",
                                       "reason": "no point outside full-day interval"})
                    continue
                when = DAY + " " + clock(point)
                excluded_args = {"start_time": when, "end_time": when}
            add(case, "time", included, materialize(case, excluded_args, owner))
        if api == "SearchMessages":
            absent = [word for word in WORDS if word not in row["message"].split()]
            if not absent:
                exclusions.append({"id": case["id"], "contrast": "keyword",
                                   "reason": "source body contains all four colour words"})
                continue
            excluded_args = dict(args, query=rng.choice(absent))
            add(case, "keyword", included, materialize(case, excluded_args, owner))
    assert len({p["id"] for p in pairs}) == len(pairs)
    return pairs, exclusions


def metrics(records):
    if not records:
        raise ValueError("Empty contrast group")
    n = len(records)
    return {"pairs": n,
            "both_correct": sum(r["include_prediction"] == 1 and r["exclude_prediction"] == 0 for r in records) / n,
            "prediction_flip_rate": sum(r["include_prediction"] != r["exclude_prediction"] for r in records) / n,
            "mean_probability_contrast": sum(r["include_probability"] - r["exclude_probability"] for r in records) / n,
            "opposite_label_exact_aliases": sum(r["encoded_equal"] for r in records)}
