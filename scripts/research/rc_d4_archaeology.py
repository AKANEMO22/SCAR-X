"""Bounded data-only D4 audit. Never imports SCAR-X or executes a neural model.

Inputs are the immutable P1b source tree, ACR DEV predictions and saved Hybrid
training-case direction vectors. Historical TEST/confirmation arrays are excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import psutil

EXPECTED_SHA = "1ea065be194d44000ba795138c5917f3d13adda0"
ACR = "20261112T121634Z-acr-dev-v1-resume-59ced4"
CREDIT = "20261114T064319Z-credit-direction-diagnostic-v1-fc84eb"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--max-rss-mb", type=float, default=300)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    research = Path(__file__).resolve().parents[2]
    if not output.is_relative_to(research / "runs") or not output.name.startswith("rc-d4-"):
        raise ValueError("Output must be a new rc-d4-* directory under this research worktree/runs")
    if source == research or output.exists():
        raise ValueError("Immutable source must be separate; output must not already exist")
    if args.seconds > 60 or args.max_rss_mb > 300:
        raise ValueError("D4 locked budget cannot be increased")
    started = time.monotonic()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    inputs = {}

    def tick():
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        if time.monotonic() - started > args.seconds or peak_rss > args.max_rss_mb * 1024**2:
            raise RuntimeError("D4 STOP: registered time/RSS budget exceeded")

    def read_json(path):
        tick()
        inputs[str(path.relative_to(source)).replace("\\", "/")] = sha(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def git(*command):
        return subprocess.check_output(["git", "-C", str(source), *command], text=True, encoding="utf-8").strip()

    if git("rev-parse", "HEAD") != EXPECTED_SHA or git("status", "--porcelain"):
        raise ValueError("Source is not the pinned clean immutable input")
    output.mkdir(parents=False)
    result = {"completed": False, "audit_level": "data-only recomputation by same Codex system",
              "source_sha": EXPECTED_SHA, "source_root": str(source),
              "source_branch": git("branch", "--show-current"),
              "script_sha256": sha(Path(__file__)), "new_model_calls": 0,
              "new_forward_calls": 0, "new_backward_calls": 0, "new_training_updates": 0,
              "new_readout_fits": 0, "opened_TEST_or_confirmation_raw": False,
              "seed_selection": "all recorded ACR DEV rows and all 48 Hybrid diagnostic cases; no selection by scores"}
    try:
        catalog = read_json(source / "docs/research/experiment-catalog.json")
        refs = 0
        for entry in catalog["entries"]:
            for path, expected in entry["report_sha256"].items():
                tick()
                data = (source / path).read_bytes().replace(b"\r\n", b"\n")
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("Catalog report hash mismatch: " + path)
                refs += 1
        archive = read_json(source / "docs/research/archive/source-index.json")
        for item in archive["files"]:
            tick()
            if sha(source / item["local_path"]) != item["selected_source"]["sha256"]:
                raise ValueError("Archive hash mismatch")
        result["catalog_integrity"] = {"entries": len(catalog["entries"]), "report_references": refs,
                                       "immutable_copies": len(archive["files"]),
                                       "scope": "file integrity only, not scientific verification"}
        result["local_inventory"] = {"worktrees": git("worktree", "list", "--porcelain"),
            "refs": git("for-each-ref", "--format=%(refname) %(objectname)", "refs/heads", "refs/tags", "refs/remotes")}

        acr = source / "runs" / ACR
        manifest = read_json(acr / "manifest.json")
        if manifest.get("confirmation"):
            raise ValueError("ACR input must be DEV")
        table = acr / "per-seed-results-v2.csv"
        inputs[str(table.relative_to(source)).replace("\\", "/")] = sha(table)
        with table.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        contrasts = {}
        scored = 0
        max_error = 0.0
        seeds = set()
        for row in rows:
            record = read_json(acr / row["path"] / ("result-" + row["model_index"] + ".json"))
            seeds.add(int(row["seed"]))
            for split in ("id", "ood"):
                pred = record["predictions"][split]
                target = np.asarray(pred["target"])
                for kind in ("direct", "probe"):
                    values = np.asarray(pred[kind])
                    if len(values) != len(target):
                        raise ValueError("Prediction/target shape mismatch")
                    accuracy = float(np.mean(values == target))
                    expected = float(row[split + "_" + kind])
                    max_error = max(max_error, abs(accuracy - expected))
                    scored += len(values)
                    if row["budget"] == "equal_updates" and row["condition"] == "learned_mtsg" and split == "id":
                        key = row["profile"] + "/" + row["task"] + "/" + kind
                        contrasts.setdefault(key, []).append(accuracy)
        if max_error > 1e-12:
            raise ValueError("ACR CSV differs from raw prediction equality counts")
        result["ACR_DEV"] = {"classification": "VERIFIED", "scope": "saved prediction arithmetic only; no regenerated labels/checkpoint inference",
            "source_commit": manifest["source_commit"], "seeds": sorted(seeds), "model_records": len(rows),
            "prediction_entries_recounted": scored, "max_accuracy_error": max_error,
            "equal_updates_learned_id": {k: {"mean": statistics.mean(v), "rows": len(v)} for k, v in sorted(contrasts.items())},
            "retained_verdict": "ACR_SPECIALIZED_GAIN; no universal credit or continual PASS"}

        credit = source / "runs" / CREDIT
        control = read_json(credit / "control-result.json")
        groups = {}
        worst_error = 0.0
        for record in control["records"]:
            tick()
            path = credit / record["case_file"]
            reference_path = path.with_name(path.stem + "-surrogate.npz")
            for p in (path, reference_path):
                inputs[str(p.relative_to(source)).replace("\\", "/")] = sha(p)
            with np.load(path, allow_pickle=False) as case, np.load(reference_path, allow_pickle=False) as reference:
                ref = reference["hidden_direction"].astype(np.float64).ravel()
                for vector, field in (("local_direction", "local_vs_surrogate"), ("prospective_delta", "prospective_delta_vs_surrogate")):
                    candidate = case[vector].astype(np.float64).ravel()
                    cosine = float(candidate @ ref / (np.linalg.norm(candidate) * np.linalg.norm(ref)))
                    worst_error = max(worst_error, abs(cosine - record[field]["cosine"]))
                    groups.setdefault(record["state"] + "/" + vector, []).append(cosine)
        if worst_error > 1e-12:
            raise ValueError("Saved vector cosine mismatch")
        result["Hybrid_credit_vectors"] = {"classification": "VERIFIED", "scope": "saved-vector arithmetic; not independent eligibility/backward reproduction or causal quality effect",
            "cases": len(control["records"]), "seeds": sorted({r["seed"] for r in control["records"]}),
            "max_cosine_error": worst_error,
            "groups": {k: {"n": len(v), "median_cosine": statistics.median(v), "negative": sum(x < 0 for x in v)} for k, v in sorted(groups.items())}}
        result["excluded_raw"] = ["all P1b TEST", "all historical confirmation/TEST arrays", "R13.2 raw predictions (reuse documented prior recount only)",
            "semantic feature archive labels.npy includes 600 TRAIN/DEV/TEST labels: archive not loaded by this audit"]
        result["completed"] = True
    except Exception as exc:
        result["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        result["input_sha256"] = inputs
        result["resource_receipt"] = {"elapsed_seconds": time.monotonic() - started, "sampled_peak_rss_bytes": peak_rss,
            "rss_sampling": "before each input/vector record; short peaks may be missed", "processes": 1, "GPU_used": False,
            "seconds_cap": args.seconds, "rss_cap_mb": args.max_rss_mb, "threads": 1}
        temp = output / "result.json.tmp"
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(output / "result.json")
    print(json.dumps({"completed": result["completed"], "output": str(output), "resources": result["resource_receipt"]}))


if __name__ == "__main__":
    main()
