"""Independent CPU linear algebra on archived CUDA representations, no learning."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("FindAlarms", "QueryCalendar", "SearchMessages", "GetReminders")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(run, output):
    run = run.resolve()
    assert ROOT / "runs" in run.parents
    manifest = json.loads((run / "manifest.json").read_text())
    result = json.loads((run / "result.json").read_text())
    assert not result["failures"] and len(result["cells"]) == 18
    for rel, digest in manifest["source_sha256"].items():
        assert sha((run / "source" / rel).read_bytes()) == digest
    parent = ROOT / "runs" / manifest["parent_run"]
    data_raw = (parent / "locked-owned-data.json").read_bytes()
    assert sha(data_raw) == manifest["parent_data_sha256"]
    data = json.loads(data_raw)
    rows = sum(data["train_stages"], []) + data["final_DEV"]
    targets = np.array([r["target"] for r in rows])
    features_by_seed = {}
    checked, sensitivity = [], []
    for cell in result["cells"]:
        seed, arm, reset = cell["seed"], cell["arm"], cell["reset_hidden"]
        key = f"{seed}-{arm}" + ("-reset_hidden" if reset else "")
        with np.load(run / (key + ".npz"), allow_pickle=False) as arrays:
            f = arrays["features"]
            assert f.shape == (720, 128) and f.dtype == np.float32
            assert sha(f.tobytes()) == cell["feature_sha256"]
            x = f[:320].astype(np.float64)
            mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-3)
            np.testing.assert_allclose(mean, arrays["mean"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(scale, arrays["scale"], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(arrays["train_targets"], targets[:320])
            design = np.column_stack(((x - mean) / scale, np.ones(320)))
            weights = np.linalg.solve(design.T @ design + np.eye(129), design.T @ (targets[:320] * 2 - 1))
            np.testing.assert_allclose(weights, arrays["weights"], rtol=1e-6, atol=1e-7)
            all_design = np.column_stack(((f.astype(np.float64) - mean) / scale, np.ones(720)))
            scores = all_design @ weights
            np.testing.assert_allclose(scores, arrays["scores"], rtol=1e-6, atol=1e-6)
            predicted = (scores > 0).astype(int)
            np.testing.assert_array_equal(predicted, arrays["predictions"])
            for split, start, stop in (("train",0,320),("final_DEV",320,720)):
                assert float(np.mean(predicted[start:stop] == targets[start:stop])) == cell[split]["accuracy"]
                for api in FAMILIES:
                    mask = np.array([r["api"] == api for r in rows[start:stop]])
                    accuracy = float(np.mean(predicted[start:stop][mask] == targets[start:stop][mask]))
                    assert accuracy == cell[split]["by_family"][api]
            if arm == "frozen":
                features_by_seed[seed] = f.copy()
            elif reset or arm == "readout_only":
                np.testing.assert_array_equal(f, features_by_seed[seed])
            alias_ids = {"final_DEV:GetReminders:79", "final_DEV:GetReminders:85"}
            keep = np.array([r["id"] not in alias_ids for r in rows[320:]])
            sensitivity.append({"key":key,"original_accuracy":cell["final_DEV"]["accuracy"],
                "exclude_two_original_encoded_aliases_accuracy":float(np.mean(predicted[320:][keep] == targets[320:][keep])),
                "does_not_replace_original_scores":True})
        checkpoint = f"{seed}-{arm}/final-full-state.bin"
        assert sha((parent / checkpoint).read_bytes()) == manifest["checkpoint_sha256"][checkpoint]
        checked.append({"key":key,"train":cell["train"]["accuracy"],"final_DEV":cell["final_DEV"]["accuracy"]})
    contrasts = {}
    for arm in ("current_hidden", "protected_hidden"):
        differences = np.array([[next(c for c in result["cells"]
            if c["seed"] == seed and c["arm"] == arm and not c["reset_hidden"])["final_DEV"]["by_family"][api] -
            next(c for c in result["cells"] if c["seed"] == seed and c["arm"] == "frozen")["final_DEV"]["by_family"][api]
            for api in FAMILIES] for seed in (260113,260137,260171)])
        rng, distribution = np.random.default_rng(26011499), []
        for _ in range(10000):
            chosen = rng.integers(0,3,3)
            distribution.append(np.mean([np.mean(differences[s,rng.integers(0,4,4)]) for s in chosen]))
        interval = np.percentile(distribution,[2.5,97.5])*100
        actual = result[arm + "_minus_frozen_common_probe"]
        assert abs(actual["mean_pp"] - differences.mean()*100) < 1e-10
        np.testing.assert_allclose(actual["paired_seed_family_CI95_pp"],interval)
        contrasts[arm] = actual
    assert result["forwards"] == manifest["max_neural_forwards"] == 12960
    assert result["seconds"] < manifest["max_seconds"]
    report = {"run_id":run.name,"checked_cells":checked,
        "parent_checkpoints_and_source_hashes_match":True,
        "raw_features_ignore_trained_readout_and_online_normalization_verified":True,
        "reset_hidden_feature_matrices_exactly_equal_frozen":True,
        "CPU_closed_form_rederives_GPU_ridge_scores_and_predictions":True,
        "contrasts":contrasts,"alias_sensitivity_only":sensitivity,
        "new_learning_or_promotion":False,
        "verdict":"ARCHIVED_COMMON_PROBE_VERIFIED_ORIGINAL_CANDIDATE_REMAINS_PIVOT",
        "scope":"Same-orchestrator data-only audit, not outside peer review or fresh confirmation"}
    with (output or run / "independent-common-probe-audit-v1.json").open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ("checked_cells","alias_sensitivity_only")},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    main(a.run,a.output)
