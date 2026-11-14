"""Owned query/record predicate tasks, two independent executable label paths."""

import hashlib
import random
from datetime import datetime
import numpy as np
from scarx.hybrid.learner import features
from .tool_query_oracle import query

FAMILIES = ("FindAlarms", "QueryCalendar", "SearchMessages", "GetReminders")
DBS = dict(zip(FAMILIES, ("Alarm", "Calendar", "Message", "Reminder")))
KEYS = dict(zip(FAMILIES, ("alarms", "events", "messages", "reminders")))
ID_FIELDS = dict(zip(FAMILIES, ("alarm_id", "event_id", "message_id", "reminder_id")))
DAY = "2023-09-11"
CLOCK = DAY + " 12:00:00"


def clock(seconds):
    return f"{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}"


def seconds(value):
    value = value.split(" ")[-1]
    t = datetime.strptime(value, "%H:%M:%S")
    return (t.hour * 3600 + t.minute * 60 + t.second) / 86400


def independent_label(case):
    api, args, row = case["api"], case["args"], case["record"]
    if case["record_owner"] != case["owner"]:
        return 0
    if api == "FindAlarms":
        return int(args["start_range"] <= row["time"] <= args["end_range"])
    if api == "QueryCalendar":
        return int(row["start_time"] <= args["end_time"] and row["end_time"] >= args["start_time"])
    if api == "GetReminders":
        return 1
    words = args["query"].lower().split()
    match = [word in row["message"].lower() for word in words]
    return int(row["timestamp"] <= CLOCK and row["sender"] == args["sender"]
               and (all(match) if args["match_type"] == "all" else any(match)))


def make_case(family, index, split, target):
    rng = random.Random(f"26011491/{family}/{index}/{split}")
    for attempt in range(1000):
        # Names belong to the sourcequery itself; hashes/indices/split are not
        # additional neural inputs. No same-owner comparison flag is provided.
        owner = "person" + "".join(rng.choices("abcdefghkmnprstuvxyz", k=7))
        other = "person" + "".join(rng.choices("abcdefghkmnprstuvxyz", k=7))
        candidate_owner = rng.choice((owner, owner, other))
        start, stop = sorted(rng.sample(range(0, 86400, 60), 2))
        a, b = sorted(rng.sample(range(0, 86400, 60), 2))
        sender = rng.choice(("senderred", "senderblue"))
        record_sender = rng.choice(("senderred", "senderblue"))
        body = " ".join(rng.sample(("red", "blue", "green", "yellow"), rng.randrange(1, 5)))
        args = ({"start_range": clock(start), "end_range": clock(stop)} if family == "FindAlarms"
                else {"start_time": DAY + " " + clock(start), "end_time": DAY + " " + clock(stop)}
                if family == "QueryCalendar"
                else {"query": " ".join(rng.sample(("red", "blue", "green", "yellow"), 2)),
                      "match_type": rng.choice(("any", "all")), "sender": sender}
                if family == "SearchMessages" else {})
        field = ID_FIELDS[family]
        row = ({field: "record", "time": clock(a)} if family == "FindAlarms"
               else {field: "record", "start_time": DAY + " " + clock(a), "end_time": DAY + " " + clock(b)}
               if family == "QueryCalendar"
               else {field: "record", "timestamp": DAY + " " + clock(a),
                     "sender": record_sender, "message": body}
               if family == "SearchMessages"
               else {field: "record", "task": body, "status": rng.choice(("pending", "completed"))})
        db = {
            "Account": {owner: {"username": owner, "session_token": "owned-token"},
                        other: {"username": other, "session_token": "other-token"}},
            DBS[family]: {candidate_owner: {"record": row}},
        }
        oracle = query(db, family, args, session_token="owned-token", now=CLOCK)
        actual_target = int(any(r[field] == "record" for r in oracle[KEYS[family]]))
        case = {"id": f"{split}:{family}:{index}", "api": family, "args": args,
                "owner": owner, "record_owner": candidate_owner, "record": row,
                "clock": CLOCK, "source_state": db, "target": actual_target}
        assert actual_target == independent_label(case)
        if actual_target == target:
            return case
    raise RuntimeError("Could not obtain pre-registered balanced query case")


def pair_sequence(case):
    api, args, row = case["api"], case["args"], case["record"]
    # Only actual query/source fields are encoded. Labels/specification identity
    # and source_state collection/outcome are not passed to the neuron core.
    q = [0.0] * 16
    e = [0.0] * 16
    q[15], e[15] = -1, 1
    if api == "FindAlarms":
        q[0], q[1], e[2] = seconds(args["start_range"]), seconds(args["end_range"]), seconds(row["time"])
    elif api == "QueryCalendar":
        q[0], q[1] = seconds(args["start_time"]), seconds(args["end_time"])
        e[2], e[3] = seconds(row["start_time"]), seconds(row["end_time"])
    elif api == "SearchMessages":
        q[0], e[2] = seconds(case["clock"]), seconds(row["timestamp"])
        q[4] = 1 if args["match_type"] == "all" else -1
    qtext = "query " + api + " owner " + case["owner"]
    etext = "record " + api + " owner " + case["record_owner"]
    if api == "SearchMessages":
        qtext += " sender " + args["sender"] + " keywords " + args["query"]
        etext += " sender " + row["sender"] + " body " + row["message"]
    return np.concatenate((features(qtext, q), features(etext, e))).astype(np.float32)


def dataset():
    stages = []
    counter = dict.fromkeys(FAMILIES, 0)
    for stage in range(20):
        focus = FAMILIES[stage % 4]
        counts = ({focus:16} if stage < 4 else
                  {family:(10 if family == focus else 2) for family in FAMILIES})
        rows = []
        for family, count in counts.items():
            for i in range(count):
                index = counter[family]
                counter[family] += 1
                rows.append(make_case(family, index, "train", i % 2))
        random.Random(26011491 + stage).shuffle(rows)
        stages.append(rows)
    probes = [make_case(family, index, "probe", index % 2)
              for family in FAMILIES for index in range(50)]
    final = [make_case(family, index, "final_DEV", index % 2)
             for family in FAMILIES for index in range(100)]
    return {"train_stages": stages, "probes": probes, "final_DEV": final}


def forgetting(curves):
    final = curves[-1]["by_family"]
    values = {}
    for introduction, family in enumerate(FAMILIES):
        peak = max(c["by_family"][family] for c in curves[introduction:])
        values[family] = max(0.0, peak - final[family]) * 100
    mean, worst = float(np.mean(list(values.values()))), max(values.values())
    return {"by_family_pp": values, "mean_pp": mean, "worst_pp": worst,
            "within_registered_limits": mean <= 3 + 1e-9 and worst <= 5 + 1e-9}
