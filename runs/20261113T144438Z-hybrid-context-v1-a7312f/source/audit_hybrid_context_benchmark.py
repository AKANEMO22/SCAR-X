"""Recompute exact metrics/registered engineering decision from raw saved SSE."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def correct(text, expected):
    try:
        actual = json.loads(text, object_pairs_hook=unique_object)
        return type(actual) is dict and set(actual) == set(expected) and all(
            type(actual[k]) is type(v) and actual[k] == v
            for k, v in expected.items())
    except (ValueError, TypeError):
        return False


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    cases = json.loads((run / "cases.json").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in cases}
    common = {c["id"] for c in cases if c["common"]}
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    issues, arms = [], []
    for name, key in (("cases.json", "cases_sha256"),
                      ("source/hybrid_context_benchmark.py", "script_sha256"),
                      ("source/protocol.md", "protocol_sha256")):
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != manifest[key]:
            issues.append("Frozen hash mismatch: " + name)
    for arm in receipt["arms"]:
        ctx = arm["context"]
        scores, completed, long_correct, times, prefill, decode = {}, 0, 0, [], [], []
        for row in arm["records"]:
            case = by_id[row["case"]]
            if row["expected"] != case["expected"] or row["prompt_tokens"] != case["prompt_tokens"]:
                issues.append(f"Changed label/token count: {ctx}:{row['case']}")
            if row["admitted"] != (case["prompt_tokens"] + 96 < ctx):
                issues.append(f"Wrong admission: {ctx}:{row['case']}")
            if row["status"] != "COMPLETED":
                continue
            raw = json.loads((run / f"{ctx}-{row['case']}.raw.json").read_text(encoding="utf-8"))
            text = "".join(c.get("delta", {}).get("content") or "" for e in raw["chunks"] for c in e.get("choices", []))
            ok = raw["finish_reason"] == "stop" and correct(text, case["expected"])
            if text != row["answer"] or ok != row["correct"]:
                issues.append(f"Raw answer/score mismatch: {ctx}:{row['case']}")
            if raw["usage"].get("prompt_tokens") != case["prompt_tokens"]:
                issues.append(f"Input token count mismatch: {ctx}:{row['case']}")
            if raw["timings"].get("cache_n", -1) != 0:
                issues.append(f"Prefix reuse present/unknown: {ctx}:{row['case']}")
            completed += 1
            scores[row["case"]] = int(ok)
            if row["common"]:
                times.append(row["ttft_seconds"])
            else:
                long_correct += int(ok)
            prefill.append(raw["timings"].get("prompt_per_second", 0))
            decode.append(raw["timings"].get("predicted_per_second", 0))
        arms.append({"context": ctx, "status": arm["status"],
                     "common_complete": len(common.intersection(scores)),
                     "common_required": len(common),
                     "common_correct": sum(scores.get(k, 0) for k in common),
                     "extra_length_correct": long_correct, "completed": completed,
                     "peak_accounted_bytes": arm["peak_accounted_bytes"],
                     "median_common_ttft_seconds": statistics.median(times) if times else None,
                     "median_prefill_tokens_s": statistics.median(prefill) if prefill else None,
                     "median_decode_tokens_s": statistics.median(decode) if decode else None,
                     "scores": scores})
    baseline = next((a for a in arms if a["context"] == 16384), None)
    for arm in arms:
        conditions = {
            "all_common_complete": arm["common_complete"] == len(common),
            "baseline_common_complete": bool(baseline) and baseline["common_complete"] == len(common) and baseline["status"] == "COMPLETED",
            "resource_safe_with_150mb_reserve": arm["status"] == "COMPLETED" and arm["peak_accounted_bytes"] <= 3_850_000_000,
            "no_common_accuracy_regression": bool(baseline) and arm["common_correct"] >= baseline["common_correct"],
            "extra_length_benefit": arm["extra_length_correct"] >= 1,
            "common_latency_at_most_1_25x": bool(baseline and baseline["median_common_ttft_seconds"] is not None and arm["median_common_ttft_seconds"] is not None) and arm["median_common_ttft_seconds"] <= 1.25 * baseline["median_common_ttft_seconds"],
            "baseline_no_ram_breach": bool(baseline) and baseline["status"] != "RAM_FAIL",
            "integrity": not issues,
        }
        arm["registered_conditions"] = conditions
        arm["eligible_for_increase"] = arm["context"] > 16384 and all(conditions.values())
    return {"integrity_issues": issues, "arms": arms,
            "decision": "INCREASE_CANDIDATE" if any(a["eligible_for_increase"] for a in arms) else "DO_NOT_INCREASE",
            "scope": "Tiny synthetic engineering DEV pilot; no general-capability, video or retention inference. No training. Missing tasks not imputed as completed scores."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    result = audit(args.run.resolve())
    output = args.run / "independent-audit.json"
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
