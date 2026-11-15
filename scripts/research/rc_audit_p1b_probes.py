"""Independent-path arithmetic audit of saved P1b D1/D2 artifacts.

No SCAR-X/model imports, neural forward/backward, optimizer or TEST loading.
Ridge coefficients use augmented weighted least squares, not normal equations.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "runs/20261115T174514Z-scope-p1b-v1-b30f0b"
ORIGINAL_SHA = "1ea065be194d44000ba795138c5917f3d13adda0"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binary_metrics(scores, labels):
    score = np.asarray(scores, np.float64).copy()
    score[np.abs(score) <= 1e-12] = 0
    y = np.asarray(labels, np.int64)
    prediction = (score > 0).astype(np.int64)
    confusion = np.bincount(2 * y + prediction, minlength=4).reshape(2, 2)
    recall = np.diag(confusion) / np.sum(confusion, axis=1)
    # Average tied ranks and Mann-Whitney U, independently of pairwise AUC.
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    positives = int(np.sum(y == 1))
    negatives = len(y) - positives
    auc = (np.sum(ranks[y == 1]) - positives * (positives + 1) / 2) / (positives * negatives)
    margin = np.where(y == 1, score, -score)
    return {"accuracy": float(np.trace(confusion) / len(y)),
            "balanced_accuracy": float(np.sum(recall) / 2),
            "negative_recall": float(recall[0]), "positive_recall": float(recall[1]),
            "AUC": float(auc), "mean_signed_margin": float(np.mean(margin)),
            "min_signed_margin": float(np.min(margin)), "confusion": confusion.tolist()}


def augmented_ridge(train, labels, dev):
    x = np.asarray(train, np.float64)
    dx = np.asarray(dev, np.float64)
    y = np.asarray(labels, np.int64)
    mean = np.sum(x, axis=0) / len(x)
    scale = np.maximum(np.sqrt(np.sum((x - mean) ** 2, axis=0) / len(x)), 1e-6)
    train_design = np.concatenate(((x - mean) / scale, np.ones((len(x), 1))), axis=1)
    dev_design = np.concatenate(((dx - mean) / scale, np.ones((len(dx), 1))), axis=1)
    counts = np.bincount(y, minlength=2)
    weights = np.sqrt(len(y) / (2 * counts[y]))
    dimensions = x.shape[1]
    penalty_rows = np.concatenate((np.eye(dimensions), np.zeros((dimensions, 1))), axis=1)
    lhs = np.concatenate((train_design * weights[:, None], penalty_rows), axis=0)
    rhs = np.concatenate((np.where(y == 1, 1.0, -1.0) * weights, np.zeros(dimensions)))
    coefficient = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return {"coefficient": coefficient, "mean": mean, "scale": scale,
            "train_scores": train_design @ coefficient, "dev_scores": dev_design @ coefficient,
            "augmented_residual_L2": float(np.linalg.norm(lhs @ coefficient - rhs))}


def information(features, labels, spikes, saturation):
    f = np.asarray(features)
    y = np.asarray(labels)
    x = np.asarray(f, np.float64)
    centered = x - np.sum(x, axis=0) / len(x)
    singular = np.linalg.svd(centered, compute_uv=False)
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular[0]
    groups = {}
    for row, label in zip(f, y):
        identity = row.tobytes()
        groups[identity] = groups.get(identity, 0) | (1 << int(label))
    ticks = len(x) * 32
    if spikes.shape != (64,) or not np.issubdtype(spikes.dtype, np.integer):
        raise ValueError("Expected integer all-tick count for each of 64 units")
    if np.any(spikes < 0) or np.any(spikes > ticks) or not 0 <= saturation <= ticks * 64:
        raise ValueError("Stored all-tick count exceeds its registered denominator")
    return {"centered_rank": int(np.count_nonzero(singular > tolerance)),
            "singular_values": singular.tolist(),
            "variance_sum": float(np.sum(centered * centered) / len(x)),
            "observed_never_spiked_units": int(64 - np.count_nonzero(spikes)),
            "observed_never_spiked_fraction": float((64 - np.count_nonzero(spikes)) / 64),
            "saturated_q_fraction_all_ticks": float(saturation / (ticks * 64)),
            "unique_endpoint_vectors": len(groups),
            "opposite_class_exact_collision_groups": sum(value == 3 for value in groups.values()),
            "class_centroid_L2": float(np.linalg.norm(np.mean(x[y == 1], axis=0) - np.mean(x[y == 0], axis=0))),
            "stored_total_spikes": int(np.sum(spikes)), "stored_saturated_tick_unit_count": int(saturation),
            "tick_unit_denominator": ticks * 64}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--max-rss-mb", type=float, default=300)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if not run.is_relative_to(ROOT / "runs") or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Both run and output must be in this research worktree/runs")
    if output.exists() or not output.name.startswith("rc-audit-p1b-") or output == run:
        raise ValueError("Audit output must be a new unique rc-audit-p1b-* directory")
    if args.seconds > 60 or args.max_rss_mb > 300:
        raise ValueError("Registered resource budget cannot be raised")
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = 0
    input_hashes = {}
    maximum_errors = {"coefficient": 0.0, "scores": 0.0, "normalization": 0.0,
                      "metrics": 0.0, "information": 0.0, "q_endpoint": 0.0, "direct_probability": 0.0}
    solves = 0

    def guard():
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        if time.perf_counter() - started > args.seconds or peak_rss > args.max_rss_mb * 1_000_000:
            raise RuntimeError("STOP: audit time/RSS budget exceeded")

    def track(path):
        guard()
        if "sealed-test" in path.name.lower():
            raise ValueError("TEST paths are forbidden")
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        actual = digest(path)
        if rel in input_hashes and actual != input_hashes[rel]:
            raise ValueError("An audit input changed while being read")
        input_hashes[rel] = actual
        return actual

    def read_json(path):
        track(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def close(a, b, group, atol=1e-9, rtol=1e-9):
        aa, bb = np.asarray(a), np.asarray(b)
        if aa.shape != bb.shape or not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
            raise ValueError("Shape/nonfinite comparison failure: " + group)
        difference = float(np.max(np.abs(aa - bb))) if aa.size else 0.0
        maximum_errors[group] = max(maximum_errors[group], difference)
        if not np.allclose(aa, bb, atol=atol, rtol=rtol):
            raise ValueError("Arithmetic mismatch in " + group + ": " + str(difference))

    def compare_metrics(actual, expected):
        for name, value in actual.items():
            close(value, expected[name], "metrics", atol=1e-7, rtol=1e-7)

    def hidden_identity(path):
        track(path)
        with np.load(path, allow_pickle=False) as archive:
            config = json.loads(archive["metadata"].tobytes())["config"]
            h = hashlib.sha256()
            for name in ("core_w_in", "core_w_rec", "core_indices"):
                h.update(archive[name].tobytes())
            h.update(json.dumps(config, sort_keys=True).encode())
            return h.hexdigest()

    output.mkdir()
    result = {"completed": False, "audit_level": "same-system independent implementation, data-only refit algebra verification",
              "run": str(run), "audit_script_sha256": digest(Path(__file__)),
              "neural_model_imports": 0, "neural_forward_calls": 0, "backward_calls": 0,
              "hidden_training_updates": 0, "candidate_head_fits_added": 0,
              "data_only_fixed_problem_lstsq_solves": 0, "TEST_opened": False, "train_H_opened": False,
              "intervals_recomputed": False, "original_G2": "FAIL_UNCHANGED"}
    try:
        original_manifest = OLD / "closure/FINAL_MANIFEST.json"
        track(original_manifest)
        original_blob = subprocess.check_output(["git", "show", ORIGINAL_SHA + ":" + str(original_manifest.relative_to(ROOT)).replace("\\", "/")], cwd=ROOT)
        if hashlib.sha256(original_blob).hexdigest() != digest(original_manifest):
            raise ValueError("Historical custody manifest differs from pinned original commit")
        custody = read_json(original_manifest)["files"]

        def historical(path):
            actual = track(path)
            if actual != custody[str(path.relative_to(OLD)).replace("\\", "/")]:
                raise ValueError("Historical custody mismatch: " + str(path))

        for file in ("learning/G2-decision.json", "learning/result.json", "learning/completed.json"):
            historical(OLD / file)
        if read_json(OLD / "learning/G2-decision.json")["pilot_gate_pass"]:
            raise ValueError("Original G2 must remain FAIL")
        original_result = read_json(OLD / "learning/result.json")
        if original_result["G2"]["pilot_gate_pass"] or original_result["TEST_model_forwarded"]:
            raise ValueError("Unexpected historical G2/TEST state")
        status = read_json(run / "result.json")
        if status["status"] != "DONE" or status["G2"] != "FAIL_UNCHANGED" or status["hidden_training_updates"] or status["TEST_opened"]:
            raise ValueError("D1/D2 is incomplete or exceeds the diagnostic scope")
        registration = read_json(run / "registration.json")
        result["extraction_source_commit"] = registration["source_commit"]
        for name, expected in registration["source_hashes"].items():
            blob = subprocess.check_output(["git", "show", registration["source_commit"] + ":" + name], cwd=ROOT)
            committed = registration.get("committed_blob_hashes", registration["source_hashes"])[name]
            if hashlib.sha256(blob).hexdigest() != committed:
                raise ValueError("Extraction source blob hash differs from registration")
            snapshot = run / "source" / name
            if track(snapshot) != expected:
                raise ValueError("Executed source snapshot raw hash differs from registration")
            if snapshot.read_bytes().replace(b"\r\n", b"\n") != blob.replace(b"\r\n", b"\n"):
                raise ValueError("Executed and committed source differ beyond CRLF/LF recovery")
        for relative, expected in registration["input_hashes"].items():
            path = OLD / relative
            historical(path)
            if digest(path) != expected:
                raise ValueError("Extraction input differs from its registration")

        fit_labels = {}
        for seed in (261161, 261167, 261173):
            path = OLD / "learning" / ("fit-" + str(seed) + ".npz")
            historical(path)
            # Only these two array keys are materialized; never train_H or TEST.
            with np.load(path, allow_pickle=False) as archive:
                train_y = archive["train_y"].copy()
                dev_y = archive["dev_y"].copy()
            if set(np.unique(train_y)) != {0, 1} or set(np.unique(dev_y)) != {0, 1}:
                raise ValueError("Class labels are not binary and nonempty")
            fit_labels[seed] = (train_y, dev_y)

        summaries = read_json(run / "D1-results.json")
        published_information = read_json(run / "D2-information.json")
        published_probes = read_json(run / "probe-metrics-by-hidden.json")
        extractions = read_json(run / "extraction-results.json")
        extracted = {entry["key"]: entry for entry in extractions}
        if len(extracted) != len(extractions) or len(extracted) != status["unique_hidden_states"]:
            raise ValueError("Extraction key/count mismatch")
        original_rows = read_json(OLD / "learning/completed.json")
        original_by_arm = {(r["seed"], r["sign"], r["arm"]): r for r in original_rows}
        if len(summaries) != 24 or len({(r["seed"], r["sign"], r["arm"]) for r in summaries}) != 24:
            raise ValueError("Expected every original arm/seed/sign exactly once")
        key_seeds = {}
        checked_hidden = {}
        key_checkpoint_hashes = {}
        for summary in summaries:
            identity = (summary["seed"], summary["sign"], summary["arm"])
            original = original_by_arm[identity]
            for kind in ("initial", "final"):
                checkpoint = OLD / original["relative_directory"] / (kind + ".npz")
                key = summary["keys"][kind]
                if hidden_identity(checkpoint) != key:
                    raise ValueError("Hidden dedup identity differs from actual checkpoint arrays/config")
                if key in key_seeds and key_seeds[key] != summary["seed"]:
                    raise ValueError("One feature key reused across different seed datasets")
                key_seeds[key] = summary["seed"]
                checked_hidden[key] = checkpoint
                key_checkpoint_hashes.setdefault(key, set()).add(digest(checkpoint))
        if set(key_seeds) != set(extracted):
            raise ValueError("Missing or unregistered hidden extraction")

        recomputed = {}
        all_information = {}
        for key, entry in extracted.items():
            guard()
            seed = key_seeds[key]
            directory = run / "features" / key
            feature_path = directory / "features.npz"
            checkpoint = checked_hidden[key]
            receipt = read_json(directory / "result.json")
            data_path = OLD / "learning" / ("fit-" + str(seed) + ".npz")
            if receipt != entry or entry["status"] != "DONE" or entry["hidden_updates"] != 0:
                raise ValueError("Extraction per-job receipt disagrees with parent ledger")
            if track(feature_path) != entry["artifact_sha256"] or digest(data_path) != entry["data_sha256"]:
                raise ValueError("Feature/fit artifact hash differs from extraction receipt")
            # Deduplicated hidden state can have a different online head. The
            # receipt checkpoint must match one original checkpoint sharing key.
            if entry["checkpoint_sha256"] not in key_checkpoint_hashes[key]:
                raise ValueError("Extraction checkpoint hash is not an original checkpoint sharing this hidden key")
            train_y, dev_y = fit_labels[seed]
            if entry["forwards"] != len(train_y) + len(dev_y):
                raise ValueError("Unexpected extraction forward count")
            with np.load(feature_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in
                    ("train_f", "train_q", "dev_f", "dev_q", "train_spikes", "dev_spikes", "saturation")}
            if arrays["train_f"].shape != (len(train_y), 64) or arrays["dev_f"].shape != (len(dev_y), 64):
                raise ValueError("Unexpected feature shape")
            for split in ("train", "dev"):
                close(arrays[split + "_q"], np.clip(arrays[split + "_f"] / .1, -5, 5), "q_endpoint", atol=1e-6)
            recomputed[key] = {}
            all_information[key] = {}
            for variant in ("f", "q"):
                solution = augmented_ridge(arrays["train_" + variant], train_y, arrays["dev_" + variant])
                solves += 1
                guard()
                probe_path = directory / ("probe-" + variant + ".npz")
                track(probe_path)
                with np.load(probe_path, allow_pickle=False) as saved:
                    for name in ("coefficient", "mean", "scale", "train_scores", "dev_scores"):
                        group = "coefficient" if name == "coefficient" else ("normalization" if name in ("mean", "scale") else "scores")
                        close(solution[name], saved[name], group, atol=2e-7, rtol=2e-7)
                record = {"TRAIN": binary_metrics(solution["train_scores"], train_y),
                          "DEV": binary_metrics(solution["dev_scores"], dev_y),
                          "augmented_residual_L2": solution["augmented_residual_L2"]}
                for split in ("TRAIN", "DEV"):
                    compare_metrics(record[split], published_probes[key][variant][split])
                recomputed[key][variant] = record
            for split, labels, saturation in (("TRAIN", train_y, arrays["saturation"][0]), ("DEV", dev_y, arrays["saturation"][1])):
                prefix = split.lower()
                record = information(arrays[prefix + "_f"], labels, arrays[prefix + "_spikes"], int(saturation))
                for name, value in published_information[key][split].items():
                    close(record[name], value, "information", atol=1e-6, rtol=1e-6)
                all_information[key][split] = record

        adverse = []
        for summary in summaries:
            seed, sign, arm = summary["seed"], summary["sign"], summary["arm"]
            final_key, initial_key = summary["keys"]["final"], summary["keys"]["initial"]
            for split in ("TRAIN", "DEV"):
                compare_metrics(recomputed[final_key]["f"][split], summary["primary_probe"][split])
                compare_metrics(recomputed[final_key]["q"][split], summary["secondary_probe_q"][split])
                compare_metrics(recomputed[initial_key]["f"][split], summary["initial_hidden_probe"][split])
            original = original_by_arm[(seed, sign, arm)]
            checkpoint = OLD / original["relative_directory"] / "final.npz"
            with np.load(run / "features" / final_key / "features.npz", allow_pickle=False) as saved:
                q = saved["dev_q"]
            with np.load(checkpoint, allow_pickle=False) as saved:
                logits = q @ saved["core_w_out"].T + saved["core_bias"]
            probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            prediction_path = OLD / original["relative_directory"] / "predictions.npz"
            historical(prediction_path)
            with np.load(prediction_path, allow_pickle=False) as saved:
                close(probabilities, saved["dev_p_u192"], "direct_probability", atol=1e-6)
                close(saved["dev_y"], fit_labels[seed][1], "metrics", atol=0, rtol=0)
            direct = binary_metrics(probabilities[:, 1] - probabilities[:, 0], fit_labels[seed][1])
            compare_metrics(direct, summary["direct"])
            if arm == "teacher" and sign == -1:
                probe = recomputed[final_key]["f"]["DEV"]
                adverse.append({"seed": seed, "probe": probe, "direct": direct})
        diagnostic = all(r["probe"]["balanced_accuracy"] >= .75 and
            min(r["probe"]["negative_recall"], r["probe"]["positive_recall"]) >= .65 and
            r["probe"]["balanced_accuracy"] - r["direct"]["balanced_accuracy"] >= .10 for r in adverse)
        if len(adverse) != 3 or diagnostic != status["adverse_teacher_meets_preregistered_probe_diagnostic"]:
            raise ValueError("Preregistered adverse-teacher diagnostic disagrees")
        if solves != status["diagnostic_head_fits"]:
            raise ValueError("Diagnostic fit count disagrees with unique hidden variants")
        # Re-hash all historical/feature/probe/receipt inputs after analysis.
        for path, expected in list(input_hashes.items()):
            guard()
            if digest(ROOT / path) != expected:
                raise ValueError("Read-only audit input changed: " + path)
        result.update(completed=True, original_G2_unchanged=True,
            verified_subjects=len(summaries), verified_unique_hidden_states=len(extracted),
            data_only_fixed_problem_lstsq_solves=solves, candidate_head_fits_added=0,
            adverse_teacher_diagnostic=diagnostic, adverse_teacher_by_seed=adverse,
            max_errors=maximum_errors, recomputed_probes=recomputed, recomputed_information=all_information,
            extraction_provenance_scope="source/input/artifact/receipt hashes and hidden dedup keys; no independent neural replay",
            all_tick_scope="stored aggregate counts and denominators; per-tick tapes not present, not re-extracted",
            excludes="all TEST arrays, train_H, teacher identity/weights, source training, conditional bootstrap verification")
    except Exception as exc:
        result["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        result["data_only_fixed_problem_lstsq_solves"] = solves
        result["input_sha256"] = input_hashes
        result["resources"] = {"elapsed_seconds": time.perf_counter() - started,
            "sampled_peak_rss_bytes": peak_rss, "RSS_cap_bytes": args.max_rss_mb * 1_000_000,
            "seconds_cap": args.seconds, "processes": 1, "BLAS_threads": 1, "GPU": False,
            "cap_method": "cooperative sampling; brief peaks may be missed"}
        temporary = output / "result.json.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output / "result.json")
    print(json.dumps({"completed": result["completed"], "fixed_problem_solves": solves,
                      "output": str(output), "resources": result["resources"]}))


if __name__ == "__main__":
    main()
