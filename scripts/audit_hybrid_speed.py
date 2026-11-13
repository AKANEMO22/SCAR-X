"""Recalculate deployment speedup from all raw SSE; never rerun inference."""

import argparse, hashlib, json, statistics
from pathlib import Path


def audit(run):
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    issues = []
    result = {
        "arms": {},
        "speedups": {},
        "limitations": [
            "3 engineering repeats, no p95/CI claim",
            "native VLM runtime only; not complete Hermes agent/tool latency",
            "CPU/GPU floating-point accumulation may change text",
            "both128token text answers are truncated, not quality passes",
        ],
    }
    if (
        hashlib.sha256((run / "inputs.json").read_bytes()).hexdigest()
        != manifest["inputs_sha256"]
    ):
        issues.append("Input hash mismatch")
    for name, value in manifest["source_sha256"].items():
        if hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() != value:
            issues.append("Source mismatch: " + name)
    by_mode = {a["mode"]: a for a in receipt["arms"]}
    answers = {}
    for mode, arm in by_mode.items():
        if arm["status"] != "COMPLETED" or len(arm["records"]) != 9:
            issues.append("Incomplete " + mode)
        metrics = {}
        for case in ("vi-generation", "long-log", "vision"):
            rows = [r for r in arm["records"] if r["case"] == case]
            for row in rows:
                raw = json.loads(
                    (run / f'{mode}-{case}-{row["repeat"]-1}.raw.json').read_text(
                        encoding="utf-8"
                    )
                )
                actual = "".join(
                    c.get("delta", {}).get("content") or ""
                    for e in raw["chunks"]
                    for c in e.get("choices", [])
                )
                if actual != row["answer"]:
                    issues.append("SSE answer mismatch")
                if raw["timings"].get("cache_n") != 0:
                    issues.append("Unexpected text prefix cache")
                n = raw["timings"]["predicted_n"]
                ms = raw["timings"]["predicted_ms"]
                calculated = max(0, n - 1) * 1000 / ms if ms else 0
                if abs(calculated - raw["timings"]["predicted_per_second"]) > 1e-6:
                    issues.append("Decode TPS formula mismatch")
            metrics[case] = {
                "median_seconds": statistics.median(r["seconds"] for r in rows),
                "median_TTFT": statistics.median(r["ttft_seconds"] for r in rows),
                "median_decode_tokens_s": statistics.median(
                    r["timings"]["predicted_per_second"] for r in rows
                ),
                "median_prefill_tokens_s": statistics.median(
                    r["timings"]["prompt_per_second"] for r in rows
                ),
                "input_tokens": [r["usage"]["prompt_tokens"] for r in rows],
                "output_tokens": [r["usage"]["completion_tokens"] for r in rows],
                "finish_reasons": [r["finish_reason"] for r in rows],
                "characters_per_second": [r["characters"] / r["seconds"] for r in rows],
            }
            answers[(mode, case)] = [r["answer"] for r in rows]
        result["arms"][mode] = {
            "metrics": metrics,
            "resources": arm["resources"],
            "startup_seconds": arm["load_seconds"],
            "native_parity": arm["parity"]["status"],
        }
    for case in ("vi-generation", "long-log", "vision"):
        cpu = result["arms"]["cpu"]["metrics"][case]
        gpu = result["arms"]["gpu"]["metrics"][case]
        result["speedups"][case] = {
            "wall_speedup": cpu["median_seconds"] / gpu["median_seconds"],
            "TTFT_speedup": cpu["median_TTFT"] / gpu["median_TTFT"],
            "decode_speedup": gpu["median_decode_tokens_s"]
            / cpu["median_decode_tokens_s"],
            "same_input_tokens": cpu["input_tokens"] == gpu["input_tokens"],
            "exact_answers_equal": answers[("cpu", case)] == answers[("gpu", case)],
        }
    result["integrity_issues"] = issues
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    result = audit(args.run.resolve())
    with (args.run / "speed-audit.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {"speedups": result["speedups"], "issues": result["integrity_issues"]},
            indent=2,
        )
    )
