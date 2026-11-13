"""Isolated DEV event-source ranking, current hybrid learner unchanged."""

import copy
import hashlib
import json
import random
import sqlite3
import numpy as np
from scarx.hybrid.learner import features

FAMILIES = ("latest", "before", "correction", "absent")
SEEDS = (260113, 260137, 260171)


def make_case(seed, family, index, split):
    rng = random.Random(f"{seed}/{family}/{index}/{split}")
    entity = "".join(rng.choice("abcdefhkmnprstuvxyz") for _ in range(8))
    v = rng.sample(range(100, 999), 5)
    phrasing = {
        "latest": {
            "train": "What is the latest approved value for {x}?",
            "test": "Cho biết giá trị được duyệt mới nhất của {x}.",
            "probe": "Find the current authorized setting of {x}.",
        },
        "before": {
            "train": "What approved value did {x} have before time15?",
            "test": "Giá trị được duyệt của {x} trước mốc15 là gì?",
            "probe": "For {x}, report the approved setting immediately before15.",
        },
        "correction": {
            "train": "After the withdrawn record, what is the corrected approved value of {x}?",
            "test": "Sau bản ghi bị thu hồi, giá trị đính chính được duyệt của {x} là gì?",
            "probe": "Which authorized replacement supersedes the withdrawn setting for {x}?",
        },
        "absent": {
            "train": "What approved value is recorded for {x}?",
            "test": "Có bằng chứng về giá trị được duyệt của {x} không?",
            "probe": "Report the authorized setting for {x}, or unknown if missing.",
        },
    }
    other = "".join(rng.choice("abcdefhkmnprstuvxyz") for _ in range(8))
    source_entity = other if family == "absent" else entity
    rows = []
    for j, (t, status, value) in enumerate(
        [
            (10, "approved", v[0]),
            (20, "withdrawn" if family == "correction" else "approved", v[1]),
            (30, "unverified", v[2]),
            (40, "approved", v[3]),
        ]
    ):
        words = f"Project {source_entity}. At time{t}, revision{j+1}, record status {status}, setting value {value}."
        if j == 0:
            words += " " + ("Project " + source_entity + " setting value. ") * 3
        rows.append(
            {
                "id": f"d{j}",
                "entity": source_entity,
                "time": t,
                "revision": j + 1,
                "status": status,
                "value": value,
                "text": words,
            }
        )
    rows.append(
        {
            "id": "d4",
            "entity": other + "q",
            "time": 50,
            "revision": 5,
            "status": "approved",
            "value": v[4],
            "text": f"Another project {other}q approved value {v[4]} at time50.",
        }
    )
    rng.shuffle(rows)
    gold = None if family == "absent" else "d0" if family == "before" else "d3"
    # Gold is fixed directly by the generative contract, independently of selectors.
    return {
        "id": f"{split}:{seed}:{family}:{index}",
        "family": family,
        "query": phrasing[family][split].format(x=entity),
        "entity": entity,
        "cutoff": 15 if family == "before" else None,
        "docs": rows,
        "gold_id": gold,
        "gold": None if gold is None else v[0] if gold == "d0" else v[3],
    }


def dataset():
    train = []
    for stage in range(20):
        cases = [
            make_case(44117 + stage, f, i, "train") for f in FAMILIES for i in range(3)
        ]
        positive = [
            {"case": c, "doc_id": c["gold_id"], "target": 1}
            for c in cases
            if c["gold_id"]
        ]
        positive += copy.deepcopy(positive[:3])
        negative = []
        for c in cases:
            wrong = next(d for d in c["docs"] if d["id"] != c["gold_id"])
            negative.append({"case": c, "doc_id": wrong["id"], "target": 0})
        train.append(positive + negative)
    tests = [make_case(88729, f, i, "test") for f in FAMILIES for i in range(12)]
    probes = [make_case(99661, f, i, "probe") for f in FAMILIES for i in range(2)]
    return {"train_stages": train, "test": tests, "probes": probes}


def pair_features(case, doc):
    return features(
        case["query"] + " [event] " + doc["text"], (min(len(doc["text"]) / 4000, 1.0),)
    )


def select_rule(case):
    rows = [
        d
        for d in case["docs"]
        if d["entity"] == case["entity"]
        and d["status"] == "approved"
        and (case["cutoff"] is None or d["time"] < case["cutoff"])
    ]
    return max(rows, key=lambda d: (d["time"], d["revision"]))["id"] if rows else None


def select_bm25(case):
    db = sqlite3.connect(":memory:")
    db.execute("create virtual table docs using fts5(id UNINDEXED,text)")
    db.executemany(
        "insert into docs values(?,?)", [(d["id"], d["text"]) for d in case["docs"]]
    )
    import re

    terms = re.findall(r"[^\W_]+", case["query"], re.UNICODE)[:16]
    result = db.execute(
        "select id from docs where docs match ? order by bm25(docs),rowid limit1".replace(
            "limit1", "limit 1"
        ),
        (" OR ".join('"' + t + '"' for t in terms),),
    ).fetchone()
    db.close()
    return result[0] if result else None


def select_neural(model, case):
    scores = [(model.relevance(case["query"], d), d["id"]) for d in case["docs"]]
    score, ident = max(scores, key=lambda r: r[0])
    return ident if score > 0.5 else None


class Episodic:
    def __init__(self):
        self.rows = []

    def observe(self, case, doc, target):
        vector = pair_features(case, doc)[0]
        self.rows.append((vector.copy(), target))
        self.rows = self.rows[-64:]

    def select(self, case):
        def score(doc):
            v = pair_features(case, doc)[0]
            nv = np.linalg.norm(v)
            near = sorted(
                (
                    (float(np.dot(v, x) / (nv * np.linalg.norm(x) + 1e-8)), y)
                    for x, y in self.rows
                ),
                reverse=True,
            )[:5]
            return sum(max(s, 0) * (1 if y else -1) for s, y in near) / (
                sum(max(s, 0) for s, y in near) + 1e-8
            )

        pairs = [(score(d), d["id"]) for d in case["docs"]]
        s, ident = max(pairs, key=lambda a: a[0])
        return ident if s > 0 else None


def exact_source(ident, case):
    return ident == case["gold_id"]


def forgetting(curves):
    # Positive backward loss on old source-selection probes, percentage points.
    result = {}
    for family in FAMILIES:
        scores = [x["families"][family] for x in curves]
        result[family] = max(scores[:-1]) - scores[-1] if len(scores) > 1 else 0
    result = {k: max(0.0, v) * 100 for k, v in result.items()}
    return {
        "families_pp": result,
        "mean_pp": float(np.mean(list(result.values()))),
        "within_registered_limits": float(np.mean(list(result.values()))) <= 3
        and max(result.values()) <= 5,
    }


def summary_metric(rows, key="source_correct"):
    return {
        "n": len(rows),
        "correct": sum(bool(r[key]) for r in rows),
        "accuracy": float(np.mean([r[key] for r in rows])) if rows else None,
        "families": {
            f: float(np.mean([r[key] for r in rows if r["family"] == f]))
            for f in FAMILIES
        },
    }
