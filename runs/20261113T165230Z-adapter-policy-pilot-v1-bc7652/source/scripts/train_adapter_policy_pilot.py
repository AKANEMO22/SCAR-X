"""Registered small ALIF policy study; never loads artifacts into live serving."""

import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import hashlib
import json
import sys
import time
import uuid
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.adapter_policy import (
    ACTIONS,
    LifecycleALIF,
    generated_scenarios,
    score_rows,
)
from scarx.hybrid.store import Store, encode


def balanced(predictions):
    return float(
        np.mean(
            [
                np.mean([r["correct"] for r in predictions if r["target"] == a])
                for a in ACTIONS
            ]
        )
    )


def family_ci(records, difference=False):
    families = sorted({x["family"] for x in records})
    rng = np.random.default_rng(880013)
    values = []
    for family in families:
        rows = [r for r in records if r["family"] == family]
        values.append(
            np.mean([r["delta"] if difference else r["correct"] for r in rows])
        )
    sample = np.asarray(values)
    boot = np.mean(sample[rng.integers(0, len(sample), (2000, len(sample)))], axis=1)
    return {
        "mean": float(sample.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "family_count": len(families),
        "scope": "synthetic scenario-family bootstrap, not real LoRA outcome or independent people",
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    source = (
        ROOT / "data/candidates/20261113T155741Z-adapter-data-v1-614f34/wordings.json"
    )
    words = json.loads(source.read_text(encoding="utf-8"))
    if len(words) != 96:
        raise ValueError("Need all reviewed wordings")
    rows = generated_scenarios(words)
    rid = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-adapter-policy-pilot-v1-"
        + uuid.uuid4().hex[:6]
    )
    run = ROOT / "runs" / rid
    run.mkdir()
    protocol = ROOT / "experiments/adapter-lifecycle-v1.md"
    (run / "cases.json").write_text(encode(rows), encoding="utf-8")
    files = [
        protocol,
        Path(__file__),
        ROOT / "src/scarx/evaluation/adapter_policy.py",
        source,
    ]
    manifest = {
        "run": rid,
        "created": time.time(),
        "seeds": [270113, 270127, 270131],
        "updates_per_arm": 512,
        "modes": ["learned", "readout_only", "frozen"],
        "data_sha256": hashlib.sha256((run / "cases.json").read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "synthetic_decisions_only": True,
        "live_model_changed": False,
    }
    (run / "manifest.json").write_text(encode(manifest), encoding="utf-8")
    store = Store(ROOT / "local/hybrid-v1")

    def event(kind, payload):
        store.event("adapter_policy." + kind, {"run": rid, **payload}, "orchestrator")
        print(encode({"kind": kind, **payload}), flush=True)

    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    if set(r["family"] for r in train) & set(r["family"] for r in test):
        raise ValueError("Family split overlap")
    event(
        "started",
        {
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
            "arms": manifest["modes"],
            "seeds": manifest["seeds"],
            "registered_updates": 512,
            "scope": "Synthetic local-policy teaching, no VLM LoRA training",
        },
    )
    summary = []
    test_records = {}
    parity = []
    start = time.monotonic()
    process = psutil.Process()
    peak = 0
    for seed in manifest["seeds"]:
        rng = np.random.default_rng(seed + 19)
        by_target = {a: [r for r in train if r["target"] == a] for a in ACTIONS}
        order = [
            by_target[ACTIONS[i % 6]][int(rng.integers(len(by_target[ACTIONS[i % 6]])))]
            for i in range(512)
        ]
        for mode in manifest["modes"]:
            model = LifecycleALIF(seed, mode)
            initial = model.model.core.w_in.copy()
            logs = []
            for i, row in enumerate(order):
                result = model.learn(row)
                peak = max(peak, process.memory_info().rss)
                logs.append(
                    {
                        "step": i + 1,
                        "id": row["id"],
                        "loss": result["loss"],
                        "updates": result["updates"],
                        "hidden_updates": result["hidden_updates"],
                        "delta": result["weight_delta"],
                    }
                )
                if (i + 1) % 128 == 0:
                    event(
                        "learning",
                        {
                            "seed": seed,
                            "mode": mode,
                            "step": i + 1,
                            "loss": result["loss"],
                            "synapse_updates": result["updates"],
                            "hidden_updates": result["hidden_updates"],
                            "weight_delta": result["weight_delta"],
                            "process_rss_bytes": process.memory_info().rss,
                        },
                    )
                if time.monotonic() - start > 180:
                    raise TimeoutError("Pilot total 180s cap")
            development = score_rows(model, dev)
            predictions = score_rows(model, test)
            record = {
                "seed": seed,
                "mode": mode,
                "dev_balanced_accuracy": balanced(development),
                "test_balanced_accuracy": balanced(predictions),
                "test_correct": sum(r["correct"] for r in predictions),
                "test_n": len(predictions),
                "unsafe_count": sum(r["unsafe"] for r in predictions),
                "hidden_weight_delta": float(
                    np.linalg.norm(model.model.core.w_in - initial)
                ),
                "model_bytes": model.model.memory_bytes(),
                "bootstrap": family_ci(predictions),
            }
            summary.append(record)
            test_records[(seed, mode)] = predictions
            # Capture actual outputs and labels as a separate timestamped trace.
            event(
                "answers",
                {
                    "seed": seed,
                    "mode": mode,
                    "examples": [
                        {k: v for k, v in p.items() if k != "probabilities"}
                        for p in predictions[:6]
                    ],
                    "all_predictions": str(
                        (run / f"{seed}-{mode}.predictions.json").relative_to(ROOT)
                    ),
                },
            )
            event("arm_finished", record)
            data = model.dumps()
            (run / f"{seed}-{mode}.model.npz").write_bytes(data)
            restored = LifecycleALIF.loads(data)
            expected = model.learn(order[0])
            actual = restored.learn(order[0])
            same = expected == actual and model.dumps() == restored.dumps()
            parity.append(same)
            (run / f"{seed}-{mode}.learning.json").write_text(
                encode(logs), encoding="utf-8"
            )
            (run / f"{seed}-{mode}.predictions.json").write_text(
                encode(predictions), encoding="utf-8"
            )
    contrasts = []
    for seed in manifest["seeds"]:
        own = test_records[(seed, "learned")]
        other = test_records[(seed, "readout_only")]
        differences = [
            {"family": a["family"], "delta": int(a["correct"]) - int(b["correct"])}
            for a, b in zip(own, other)
        ]
        contrasts.append(
            {"seed": seed, "hidden_minus_readout": family_ci(differences, True)}
        )
    frozen = {r["seed"]: r for r in summary if r["mode"] == "frozen"}
    eligible = all(
        r["test_balanced_accuracy"] >= 0.85
        and r["unsafe_count"] == 0
        and r["test_balanced_accuracy"] - frozen[r["seed"]]["test_balanced_accuracy"]
        >= 0.1
        for r in summary
        if r["mode"] == "learned"
    ) and all(parity)
    report = {
        "manifest": manifest,
        "arms": summary,
        "contrasts": contrasts,
        "cold_next_update_exact": all(parity),
        "rule_baseline_accuracy": 1.0,
        "rule_baseline_scope": "labels are defined by explicit synthetic contract, so this is an oracle upper reference, not a competitive learned result",
        "eligible_for_advisory": eligible,
        "decision": "ADVISORY_RESEARCH_ONLY" if eligible else "NOT_QUALIFIED",
        "live_weights_changed": False,
        "VLM_LoRA_trained": False,
        "seconds": time.monotonic() - start,
        "peak_process_rss_bytes": peak,
        "limits": [
            "no actual LoRA artifact or measured downstream benefit",
            "no real retention20stages",
            "three seeds and synthetic known contract only",
            "no production policy replacement",
        ],
    }
    (run / "receipt.json").write_text(encode(report), encoding="utf-8")
    event("finished", report)
    store.close()


if __name__ == "__main__":
    main()
