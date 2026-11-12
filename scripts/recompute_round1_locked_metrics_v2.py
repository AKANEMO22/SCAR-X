"""Recompute published trial summaries; never run or update a neural model."""
from pathlib import Path
import argparse
import json

import numpy as np

from review_round1_handoffs_v2 import ROOT, pinned_json


def bootstrap(values):
    a = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(2101)
    means = a[rng.integers(0, len(a), (10000, len(a)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return {"mean": float(a.mean()), "lower95": float(lo), "upper95": float(hi), "n": len(a)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = (ROOT / args.run_dir).resolve()
    if run.parent != ROOT / "runs" or not (run / "result.json").is_file():
        raise ValueError("Existing owned review run required")
    result = {"scope": "Published-row arithmetic only; not a replacement statistical gate",
              "diagnostic_bootstrap": {"seed": 2101, "resamples": 10000}, "grok": {}, "gemini": {}}
    for name in ("grok-slow-trace-v1", "grok-slow-trace-v1a-confirmation-71005"):
        d, sha = pinned_json("grok", f"runs/{name}/result.json")
        gains = [r["slow-correct"] - r["slow-frozen"] for r in d["confirmation"]]
        stats = bootstrap(gains)
        stats.update(result_sha256=sha, gains=gains,
                     lower_bound_reaches_registered_0_05=stats["lower95"] >= .05)
        result["grok"][name] = stats
    d, _ = pinned_json("antigravity_gemini", "artifacts/collab/scarx-neural-round1-20261111/antigravity/dc-alif-round1-latest.json")
    rows = [r for r in d["p1_trials"] if r["phase"] == "confirmation"]
    for task in ("order", "xor", "strict_binding"):
        by = {method: sorted([r for r in rows if r["task"] == task and r["method"] == method], key=lambda r:r["seed"])
              for method in ("correct", "frozen", "zero", "shuffled", "no_gate", "bptt_control")}
        means = {m: float(np.mean([r["direct"] for r in rs])) for m, rs in by.items()}
        summary = d["p1_confirmation"]["by_task"][task]
        result["gemini"][task] = {
            "direct_means": means,
            "direct_mean_matches_stored": {m: means[m] == summary[m + "_direct"]["mean"] for m in means if m != "zero"},
            "correct_minus_frozen_direct": bootstrap([a["direct"] - b["direct"] for a,b in zip(by["correct"],by["frozen"])]),
            "correct_minus_frozen_probe": bootstrap([a["probe"] - b["probe"] for a,b in zip(by["correct"],by["frozen"])]),
            "zero_frozen_direct_equal_all_seeds": all(a["direct"] == b["direct"] for a,b in zip(by["zero"],by["frozen"])),
            "initial_to_trained_probe_gain": bootstrap([r["probe_gain"] for r in by["correct"]]),
        }
    target = run / "locked-metrics-audit.json"
    with target.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
