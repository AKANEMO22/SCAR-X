"""Independent raw re-score and paired exploratory CI; no training or inference."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def gold(case):
    # Independently decode query identity/cutoff rather than trusting cached gold.
    entities = [d["entity"] for d in case["docs"]]
    candidates = [
        d
        for d in case["docs"]
        if re.search(r"\b" + re.escape(d["entity"]) + r"\b", case["query"])
        and d["status"] == "approved"
    ]
    before = bool(re.search(r"(before|trước|mốc)", case["query"], re.I))
    if before:
        candidates = [d for d in candidates if d["time"] < 15]
    if not candidates:
        return None, None
    row = sorted(candidates, key=lambda r: (r["time"], r["revision"]))[-1]
    return row["id"], row["value"]


def paired_bootstrap(rows, a, b, metric="grounded_correct"):
    seeds = sorted({r["seed"] for r in rows})
    families = sorted({r["family"] for r in rows})
    index = {(r["seed"], r["method"], r["id"]): float(r[metric]) for r in rows}
    ids = {f: sorted({r["id"] for r in rows if r["family"] == f}) for f in families}
    rng = np.random.default_rng(14112026)
    samples = []
    for _ in range(5000):
        differences = []
        for family in rng.choice(families, len(families), replace=True):
            caseids = rng.choice(ids[family], len(ids[family]), replace=True)
            for seed in rng.choice(seeds, len(seeds), replace=True):
                differences.extend(
                    index[seed, a, i] - index[seed, b, i] for i in caseids
                )
        samples.append(np.mean(differences))
    actual = np.mean(
        [
            index[s, a, i] - index[s, b, i]
            for s in seeds
            for f in families
            for i in ids[f]
        ]
    )
    return {
        "difference_pp": float(actual * 100),
        "CI95_pp": list(map(float, np.quantile(samples, [0.025, 0.975]) * 100)),
        "bootstrap_repetitions": 5000,
        "scope": "exploratory hierarchical paired resampling; only3independent neural seeds, same native responses shared for identical inputs; not a confirmatory claim",
    }


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    data = json.loads((run / "data.json").read_text(encoding="utf-8"))
    result_path = run / (
        "completed-inference.json"
        if (run / "completed-inference.json").exists()
        else "result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    problems = []
    if (
        hashlib.sha256((run / "data.json").read_bytes()).hexdigest()
        != manifest["data_sha256"]
    ):
        problems.append("data hash")
    for rel, expected in manifest["source_sha256"].items():
        if hashlib.sha256((run / "source" / rel).read_bytes()).hexdigest() != expected:
            problems.append("source " + rel)
    cases = {c["id"]: c for c in data["test"]}
    native = {}
    for native_file in ["native.jsonl", "native-completion.jsonl"]:
        if (run / native_file).exists():
            for raw in (run / native_file).read_text(encoding="utf-8").splitlines():
                row = json.loads(raw)
                if row["key"] in native:
                    problems.append("duplicate native response key")
                expected = hashlib.sha256(
                    json.dumps(
                        row["request"],
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                if row["key"] != expected:
                    problems.append("native request hash")
                native[row["key"]] = row
    for c in cases.values():
        gid, value = gold(c)
        if gid != c["gold_id"] or value != c["gold"]:
            problems.append("generative label mismatch " + c["id"])
    rows = []
    for old in result.get("predictions", []):
        c = cases[old["id"]]
        gid, value = gold(c)
        n = native.get(old.get("native_key"))
        if not n:
            problems.append("missing native answer")
            continue
        try:
            answer = json.loads(
                n["response"]["choices"][0]["message"].get("content", "")
            )
            answer_ok = (
                type(answer) is dict
                and set(answer) == {"value"}
                and answer["value"] == value
                and (
                    type(answer["value"]) is int
                    if value is not None
                    else answer["value"] is None
                )
            )
        except (ValueError, TypeError):
            answer_ok = False
        source = old["selected"] == gid
        grounded = bool(source and answer_ok)
        if old["source_correct"] != source or old["grounded_correct"] != grounded:
            problems.append("metric mismatch")
        rows.append(
            dict(
                old,
                source_correct=source,
                answer_correct=answer_ok,
                grounded_correct=grounded,
            )
        )
    methods = manifest["config"]["arms"]
    seeds = manifest["config"]["seeds"]
    metrics = {}
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        metrics[method] = {
            "n": len(selected),
            "grounded_QA": (
                float(np.mean([r["grounded_correct"] for r in selected]))
                if selected
                else None
            ),
            "source_selection": (
                float(np.mean([r["source_correct"] for r in selected]))
                if selected
                else None
            ),
            "per_seed": {
                str(s): {
                    "n": len([r for r in selected if r["seed"] == s]),
                    "grounded_QA": (
                        float(
                            np.mean(
                                [
                                    r["grounded_correct"]
                                    for r in selected
                                    if r["seed"] == s
                                ]
                            )
                        )
                        if selected
                        else None
                    ),
                }
                for s in seeds
            },
        }
    complete = (
        len(rows) == len(methods) * len(seeds) * len(cases) and not result["failures"]
    )
    contrasts = {}
    if complete:
        for b in [
            "bm25",
            "source_rule",
            "episodic",
            "frozen",
            "readout_only",
            "hidden_reset",
        ]:
            contrasts["hidden_minus_" + b] = paired_bootstrap(rows, "hidden", b)
    learning = {}
    retention = {}
    for seed in seeds:
        path = run / f"{seed}-learning.json"
        if path.exists():
            logs = json.loads(path.read_text(encoding="utf-8"))
            learning[str(seed)] = {}
            for method, items in logs.items():
                learning[str(seed)][method] = {
                    "attempts": len(items),
                    "accepted": sum(r.get("accepted", False) for r in items),
                    "prior_wrong": sum(r["prior_wrong"] for r in items),
                    "seconds": sum(r["seconds"] for r in items),
                    "hidden_weight_movement_accepted": sum(
                        r.get("accepted", False)
                        and any(
                            r.get("weight_delta", {}).get(k, 0) > 0
                            for k in ["w_in", "w_rec"]
                        )
                        for r in items
                    ),
                }
        stored = result["seeds"].get(str(seed), {})
        retention[str(seed)] = {}
        for method, curve in stored.get("curves", {}).items():
            loss = {
                f: max(
                    0.0,
                    max(p["families"][f] for p in curve[:-1])
                    - curve[-1]["families"][f],
                )
                * 100
                for f in curve[0]["families"]
            }
            retention[str(seed)][method] = {
                "first_accuracy": curve[0]["accuracy"],
                "final_accuracy": curve[-1]["accuracy"],
                "positive_forgetting_pp": loss,
                "mean_forgetting_pp": float(np.mean(list(loss.values()))),
                "retention_gate": np.mean(list(loss.values())) <= 3
                and max(loss.values()) <= 5,
                "note": "fixed8DEV probes; no claim of20real skills or formal retention",
            }
    qualified = complete and not problems
    if qualified:
        strongest = max(
            ["bm25", "source_rule", "episodic"], key=lambda m: metrics[m]["grounded_QA"]
        )
        main = contrasts["hidden_minus_" + strongest]
        qualified = (
            main["difference_pp"] >= 5
            and main["CI95_pp"][0] > 0
            and contrasts["hidden_minus_readout_only"]["CI95_pp"][0] > 0
            and contrasts["hidden_minus_hidden_reset"]["CI95_pp"][0] > 0
            and all(
                v.get("hidden", {}).get("retention_gate", False)
                for v in retention.values()
            )
        )
    return {
        "run_id": run.name,
        "complete": complete,
        "integrity_problems": problems,
        "raw_failures": result["failures"],
        "parent_failures": result.get("parent_failures", result["failures"]),
        "registered_parent_budget_pass": not bool(result.get("completion"))
        and not result["failures"],
        "completion": result.get("completion"),
        "metrics": metrics,
        "contrasts": contrasts,
        "learning": learning,
        "retention": retention,
        "native_unique_calls": len(native),
        "score_denominator_note": "same48questions x3neural seeds; identical VLM requests reused, not144independent question samples",
        "SNN_advantage_qualified_on_this_DEV": bool(qualified),
        "verdict": (
            "GO_DEV_SIGNAL_ONLY" if qualified else "PIVOT_NO_PROVEN_SNN_ADVANTAGE"
        ),
        "generalization_limit": "procedural provenance contract, small DEV; rule can hit ceiling, cannot establish broad language/vision gain",
        "current_scientific_gates_unchanged": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    result = audit(a.run)
    raw = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
        default=lambda x: bool(x) if isinstance(x, np.bool_) else str(x),
    )
    if a.out:
        with a.out.open("x", encoding="utf-8") as f:
            f.write(raw)
    print(raw)
