"""Read-only independent rescoring of the locked native VLM engineering pair."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def strict_score(case, answer):
    answer = answer.strip()
    if "expected_json" in case:
        try:
            return json.loads(answer) == case["expected_json"]
        except (ValueError, TypeError):
            return False
    return answer.casefold() == case["expected"].casefold()


def audit(run):
    def read(name):
        return json.loads((run / name).read_text(encoding="utf-8"))

    lock = read("paired-lock.json")
    problems = []
    if hashlib.sha256((run / "paired-cases.json").read_bytes()).hexdigest() != lock["cases_sha256"]:
        problems.append("Locked case bytes changed")
    for rel, expected in lock["source_sha256"].items():
        raw = (run / "paired-source" / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            problems.append("Archived source mismatch: " + rel)
    cases = read("paired-cases.json")
    arms = {}
    for arm in ("baseline", "candidate"):
        data = read("paired-" + arm + ".json")
        if [r["id"] for r in data["cases"]] != [c["id"] for c in cases]:
            problems.append(arm + " missing/reordered case")
        records = []
        for case, row in zip(cases, data["cases"]):
            if row["question"] != case["question"]:
                problems.append(arm + " changed question: " + case["id"])
            response = row["response"]
            reconstructed = "".join(
                c.get("delta", {}).get("content") or ""
                for chunk in response["chunks"]
                for c in chunk.get("choices", [])
            )
            if reconstructed != response["answer"]:
                problems.append(arm + " edited response: " + case["id"])
            correct = strict_score(case, reconstructed)
            if correct != row["correct"]:
                problems.append(arm + " incorrect raw score: " + case["id"])
            records.append({
                "id": case["id"], "language": case["language"],
                "answer": reconstructed, "correct": correct,
                "ttft_seconds": response.get("ttft_seconds"),
                "native_decode_tokens_per_second": response["timings"]["predicted_per_second"],
            })
        with (run / ("paired-" + arm + "-gpu.csv")).open() as f:
            free = [int(r[3]) for r in csv.reader(f) if len(r) == 4]
        minimum = min(free + [data["loaded_gpu"]["free_MiB"]])
        if minimum != data["sampled_min_free_MiB"]:
            problems.append(arm + " minimum VRAM availability mismatch")
        total = sum(r["correct"] for r in records)
        vi = sum(r["correct"] for r in records if r["language"] == "vi")
        if (total, vi) != (data["score"], data["Vi_score"]):
            problems.append(arm + " aggregate score mismatch")
        parity = data["native_gate"]["parity"]
        parity_ok = (
            len(parity["cases"]) == 3
            and all(c["native_top1"] == c["reference_top1"]
                    and c["max_probability_difference"] <= .03 for c in parity["cases"])
        )
        ttfts = [r["ttft_seconds"] for r in records if r["ttft_seconds"] is not None]
        if len(ttfts) != len(records):
            problems.append(arm + " missing first-content latency")
        # All 12 TTFTs below 5s is stronger than a p95 gate on these 12 samples.
        resource_ok = (
            minimum >= lock["global_reserve_MiB"]
            and not data["failures"] and not data["resource_failures"]
            and len(ttfts) == 12 and max(ttfts) <= 5
            and len(data["cycles"]) == 3 and data["full_state_rollback_exact"]
            and data["gateway_stopped"] and data["tool_transport"]["tool_calls_emitted"]
        )
        arms[arm] = {
            "identity": data["identity"], "score": total, "Vi_score": vi,
            "sampled_min_free_MiB": minimum, "native_parity_pass": parity_ok,
            "engineering_resource_pass": resource_ok,
            "max_ttft_seconds": max(ttfts),
            "median_native_decode_tokens_per_second": statistics.median(
                r["native_decode_tokens_per_second"] for r in records),
            "asr_cycle_seconds": [c["seconds"] for c in data["cycles"]],
            "records": records,
        }
    ref = read("reference-keymap-v2.json")
    if ref["status"] != "REFERENCE_READY" or ref["mismatches"] or ref["unused_tensors"]:
        problems.append("Original reference coverage failed")
    if arms["candidate"]["identity"] != lock["identity"]:
        problems.append("Candidate model identity mismatch")
    quality_pass = (
        arms["candidate"]["score"] >= arms["baseline"]["score"]
        and arms["candidate"]["Vi_score"] >= arms["baseline"]["Vi_score"]
    )
    qualified = (
        not problems and quality_pass
        and all(a["native_parity_pass"] and a["engineering_resource_pass"] for a in arms.values())
    )
    return {
        "run_id": run.name, "integrity_problems": problems, "arms": arms,
        "locked_quality_gate_pass": quality_pass,
        "candidate_qualified_for_promotion": qualified,
        "verdict": "ENGINEERING_CANDIDATE_ONLY" if qualified else "NOT_QUALIFIED_KEEP_CURRENT_MODEL",
        "scope": "12 engineering DEV cases; no formal quality, 8h, retention or neural advantage claim",
        "original_failures_retained": True, "trains_or_changes_serving": False,
        "inputs_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in run.glob("*.json") if not f.name.startswith("audit-")
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = audit(a.run)
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    if a.output:
        with a.output.open("x", encoding="utf-8") as f:
            f.write(payload + "\n")
    print(json.dumps({k: report[k] for k in (
        "run_id", "integrity_problems", "verdict", "locked_quality_gate_pass"
    )}, ensure_ascii=False))
