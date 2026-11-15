"""RC-D3: saved-array arithmetic only; no neural imports, forward, fit or training."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
import psutil

SOURCE_SHA = "1ea065be194d44000ba795138c5917f3d13adda0"
RUN = "runs/20261115T174514Z-scope-p1b-v1-b30f0b"
EPS = 1e-12
TOL = 1e-5
ROWS = slice(8, 64)
COLS = [0, 1, 3, 4]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in (archive.files if keys is None else keys)}


def norm(x):
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return float(np.sum(a * b) / denominator) if denominator > EPS else None


def relative(a, b):
    denominator = (norm(a) + norm(b)) / 2
    return norm(a - b) / denominator if denominator > EPS else None


def disagreement(a, b):
    active = (np.abs(a) > EPS) & (np.abs(b) > EPS)
    return float(np.mean(np.sign(a[active]) != np.sign(b[active]))) if active.any() else None


def fraction_active(x):
    return float(np.mean(np.abs(x) > EPS))


def clipped_fraction(raw, threshold):
    active = np.abs(raw) > EPS
    return float(np.mean(np.abs(raw[active]) > threshold)) if active.any() else None


def stats(values):
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max())}


def summarize(records):
    excluded = {"seed", "sign", "arm", "update", "label", "window"}
    keys = sorted(set().union(*(set(row) for row in records)) - excluded)
    return {key: stats([row.get(key) for row in records]) for key in keys}


def step(direction, previous_m, previous_v, weights):
    m = .9 * previous_m + .1 * direction
    v = .99 * previous_v + .01 * direction ** 2
    raw = .008 * m / (np.sqrt(v) + 1e-4)
    clipped = np.clip(raw, -.01, .01)
    delta = np.clip(weights + clipped, -3, 3) - weights
    delta[:8] = 0
    return m, v, raw, clipped, delta


def main():
    started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    source = args.source_root.resolve()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    peak_rss = 0
    peak_os = 0
    failures = []

    def check_budget():
        nonlocal peak_rss, peak_os
        info = process.memory_info()
        peak_rss = max(peak_rss, info.rss)
        peak_os = max(peak_os, getattr(info, "peak_wset", 0))
        if time.perf_counter() - started >= 60:
            raise RuntimeError("Registered 60-second budget exceeded")
        if info.rss >= 300_000_000:
            raise RuntimeError("Registered sampled RSS budget exceeded")

    def need(condition, why):
        if not condition:
            failures.append(why)

    def close_error(a, b, why):
        error = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
        need(error < TOL, why)
        return error

    final_status = {"status": "FAILED", "neural_forwards": 0, "hidden_training_updates": 0,
                    "probe_fits": 0, "TEST_opened": False, "G2": "FAIL_UNCHANGED"}
    try:
        check_budget()
        source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        authoring_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        need(source_sha == SOURCE_SHA, "immutable source checkout SHA")
        phase = source / RUN / "learning"
        registration = json.loads((phase / "registration.json").read_text(encoding="utf-8"))
        completed = json.loads((phase / "completed.json").read_text(encoding="utf-8"))
        locks = json.loads((phase / "data-lock.json").read_text(encoding="utf-8"))
        need(len(completed) == 24, "24 registered cells")
        need({(r["seed"], r["sign"], r["arm"]) for r in completed} ==
             {(s, sign, a) for s in (261161, 261167, 261173) for sign in (1, -1)
              for a in ("label", "teacher", "shuffled", "readout")}, "exact matrix")
        input_paths = [phase / n for n in ("registration.json", "completed.json", "data-lock.json")]
        input_paths += [phase / "source" / name for name in registration["source_hashes"]]
        input_paths += [phase / f"fit-{seed}.npz" for seed in (261161, 261167, 261173)]
        for record in completed:
            directory = (source / RUN / record["relative_directory"]).resolve()
            need(directory.is_relative_to(phase / "trials"), "trial path within allowlisted directory")
            input_paths += [directory / name for name in ("initial.npz", "final.npz", "learning.csv",
                                                       "learning-diagnostics.npz", "predictions.npz")]
        manifest = {str(p.relative_to(source)).replace("\\", "/"): digest(p) for p in input_paths}
        config = {"protocol": "rc-d3-signal-forensic-v1", "source_checkout_sha": source_sha,
                  "historical_registered_source_sha": registration["source_commit"],
                  "authoring_commit": authoring_sha, "epsilon": EPS, "audit_tolerance": TOL,
                  "wall_budget_seconds": 60, "sampled_rss_limit_bytes": 300_000_000,
                  "threads": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                  "script_sha256": digest(Path(__file__)),
                  "protocol_sha256": digest(root / "experiments/rc-d3-signal-forensic-v1.md"),
                  "input_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  "started_UTC": datetime.now(timezone.utc).isoformat()}
        config["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        atomic_json(output / "registration.json", config)
        atomic_json(output / "input-hashes.json", manifest)
        for name, expected in registration["source_hashes"].items():
            need(manifest[str((phase / "source" / name).relative_to(source)).replace("\\", "/")] == expected,
                 "registered source hash " + name)
        for item in locks:
            key = str((phase / f"fit-{item['seed']}.npz").relative_to(source)).replace("\\", "/")
            need(manifest[key] == item["fit_sha256"], "registered fit archive hash")
        if failures:
            raise AssertionError("Pre-metric provenance checks failed: " + repr(failures))
        all_rows = []
        trial_summaries = []
        pair_rows = []
        max_errors = {}
        fit_labels = {seed: load_npz(phase / f"fit-{seed}.npz", ["train_y"])["train_y"]
                      for seed in (261161, 261167, 261173)}
        csv_handle = (output / "updates.csv").open("w", encoding="utf-8", newline="")
        writer = None
        for record in completed:
            check_budget()
            seed, sign, arm = record["seed"], record["sign"], record["arm"]
            identity = f"{seed}/{sign}/{arm}"
            directory = source / RUN / record["relative_directory"]
            d = load_npz(directory / "learning-diagnostics.npz")
            ini = load_npz(directory / "initial.npz")
            fin = load_npz(directory / "final.npz")
            log = list(csv.DictReader((directory / "learning.csv").open(encoding="utf-8")))
            need(len(log) == 192 and len(d["directions"]) == 192, identity + " 192 updates")
            need(digest(directory / "initial.npz") == record["initial_checkpoint_sha256"], identity + " initial hash")
            need(digest(directory / "final.npz") == record["final_checkpoint_sha256"], identity + " final hash")
            frozen = np.ones((64, 64), dtype=bool)
            frozen[:, COLS] = False
            frozen[:8] = True
            need(np.array_equal(ini["core_w_in"][frozen], fin["core_w_in"][frozen]), identity + " frozen weights")
            need(np.array_equal(ini["core_w_rec"], fin["core_w_rec"]), identity + " recurrent frozen")
            need(np.array_equal(d["weights_before"][0], ini["core_w_in"][:, COLS]), identity + " initial registered coordinates")
            need(np.array_equal(d["mom_after"][-1], fin["mom"]), identity + " final momentum")
            need(np.array_equal(d["energy_after"][-1], fin["energy"]), identity + " final energy")
            rows = []
            errors = {key: 0.0 for key in ("direction", "moment", "energy", "physical_delta", "next_weights", "head", "bias", "CSV", "probability")}
            for i in range(192):
                check_budget()
                label = int(fit_labels[seed][i])
                p = d["p_before"][i].astype(float)
                error = -p
                error[label] += 1
                head = d["head_before"][i].astype(float)
                q = d["q_endpoint"][i].astype(float)
                E = d["eligibility"][i].astype(float)
                L = ((head.T @ error) * (np.abs(q) < 5) / .1)[:, None] * E
                L[:8] = 0
                H = d["hint_banks"][i].astype(float)
                H[:8] = 0
                D = L + H if arm in ("teacher", "shuffled") else L.copy()
                weights = d["weights_before"][i].astype(float)
                actual = d["actual_hidden_deltas"][i].astype(float)
                previous_m = d["mom_after"][i - 1].astype(float) if i else ini["mom"].astype(float)
                previous_v = d["energy_after"][i - 1].astype(float) if i else ini["energy"].astype(float)
                m, v, raw, clipped, delta = step(D, previous_m, previous_v, weights)
                omission_delta = step(L, previous_m, previous_v, weights)[-1] if arm in ("teacher", "shuffled") else None
                expected_m, expected_v = (previous_m, previous_v) if arm == "readout" else (m, v)
                expected_delta = np.zeros_like(delta) if arm == "readout" else delta
                next_weights = d["weights_before"][i + 1] if i < 191 else fin["core_w_in"][:, COLS]
                logits = head @ q + d["bias_before"][i].astype(float)
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                head_raw = .04 * error[:, None] * q[None, :]
                head_clipped = np.clip(head_raw, -.025, .025)
                head_clipped[:, :8] = 0
                next_head = d["head_before"][i + 1] if i < 191 else fin["core_w_out"]
                next_bias = d["bias_before"][i + 1] if i < 191 else fin["core_bias"]
                current_errors = {"direction": close_error(D, d["directions"][i], identity + " direction"),
                    "moment": close_error(expected_m, d["mom_after"][i], identity + " moment"),
                    "energy": close_error(expected_v, d["energy_after"][i], identity + " energy"),
                    "physical_delta": close_error(expected_delta, actual, identity + " physical delta"),
                    "next_weights": close_error(weights + actual, next_weights, identity + " next weights"),
                    "head": close_error(np.clip(head + head_clipped, -3, 3), next_head, identity + " head"),
                    "bias": close_error(np.clip(d["bias_before"][i] + np.clip(.04 * error, -.025, .025), -3, 3), next_bias, identity + " bias"),
                    "probability": close_error(probabilities, p, identity + " saved probability")}
                csv_errors = [abs(norm(actual) - float(log[i]["actual_hidden_delta_L2"])),
                              abs(norm(D) - float(log[i]["direction_L2"])),
                              abs(norm(H) - float(log[i]["hint_bank_L2"])),
                              abs(norm(E) - float(log[i]["eligibility_L2"])),
                              abs(-float(np.log(max(p[label], 1e-12))) - float(log[i]["loss"]))]
                current_errors["CSV"] = max(csv_errors)
                need(max(csv_errors) < TOL, identity + " CSV norms/loss")
                for key, value in current_errors.items():
                    errors[key] = max(errors[key], value)
                l, h, direction, a, e, momentum = (x[ROWS] for x in (L, H, D, actual, E, m))
                row = {"seed": seed, "sign": sign, "arm": arm, "update": i + 1,
                       "label": label, "window": i // 48 + 1, "label_L2": norm(l), "hint_L2": norm(h),
                       "direction_L2": norm(direction), "eligibility_L2": norm(e), "physical_delta_L2": norm(a),
                       "hint_over_label_norm": norm(h) / norm(l) if norm(l) > EPS else None,
                       "cos_label_hint": cosine(l, h), "sign_conflict_label_hint": disagreement(l, h),
                       "cos_direction_delta": cosine(direction, a), "cos_label_delta": cosine(l, a),
                       "cos_hint_delta": cosine(h, a), "cos_direction_momentum": cosine(direction, momentum),
                       "sign_conflict_direction_delta": disagreement(direction, a),
                       "sign_conflict_direction_momentum": disagreement(direction, momentum),
                       "label_active_fraction": fraction_active(l), "hint_active_fraction": fraction_active(h),
                       "eligibility_active_fraction": fraction_active(e), "direction_active_fraction": fraction_active(direction),
                       "physical_active_fraction": fraction_active(a),
                       "normalized_proposal_L2": norm(raw[ROWS]), "clipped_proposal_L2": norm(clipped[ROWS]),
                       "proposal_scalar_clip_fraction_active": clipped_fraction(raw[ROWS], .01),
                       "hidden_bound_intervention_fraction": float(np.mean(np.abs(weights[ROWS] + clipped[ROWS]) > 3)) if arm != "readout" else 0.0,
                       "hidden_bound_occupancy_fraction": float(np.mean(np.abs(next_weights[ROWS]) >= 3)),
                       "hidden_weight_maxabs": float(np.max(np.abs(next_weights[ROWS]))),
                       "endpoint_q_clip_fraction": float(np.mean(np.abs(q[ROWS]) >= 5)),
                       "head_scalar_clip_fraction_active": clipped_fraction(head_raw[:, 8:], .025),
                       "head_bound_intervention_fraction": float(np.mean(np.abs((head + head_clipped)[:, 8:]) > 3)),
                       "head_bound_occupancy_fraction": float(np.mean(np.abs(next_head[:, 8:]) >= 3)),
                       "head_detector_contrast_mean": float(np.mean((head[1] - head[0])[8:36])),
                       "positive_probability_before": float(p[1]), "loss": float(log[i]["loss"]),
                       "omit_hint_delta_relative_difference": relative(delta[ROWS], omission_delta[ROWS]) if omission_delta is not None else None,
                       "omit_hint_delta_cosine": cosine(delta[ROWS], omission_delta[ROWS]) if omission_delta is not None else None}
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(csv_handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
            csv_handle.flush()
            predictions = load_npz(directory / "predictions.npz", ["dev_p_u192", "dev_y"])
            y = predictions["dev_y"]
            predicted = predictions["dev_p_u192"].argmax(axis=1)
            confusion = [[int(np.sum((y == a) & (predicted == b))) for b in (0, 1)] for a in (0, 1)]
            accuracy = float(np.mean(predicted == y))
            positive_recall = float(np.mean(predicted[y == 1] == 1))
            need(abs(accuracy - record["final_dev"]["accuracy"]) < 1e-12, identity + " final accuracy")
            item = {"seed": seed, "sign": sign, "arm": arm, "final_dev_accuracy": accuracy,
                    "final_positive_recall": positive_recall, "confusion_y_by_prediction": confusion,
                    "final_detector_contrast_mean": float(np.mean((fin["core_w_out"][1] - fin["core_w_out"][0])[8:36])),
                    "hidden_final_drift_L2": norm(fin["core_w_in"] - ini["core_w_in"]),
                    "audit_max_errors": errors, "all_updates": summarize(rows),
                    "windows": {str(w): summarize([x for x in rows if x["window"] == w]) for w in (1, 2, 3, 4)},
                    "classes": {str(c): summarize([x for x in rows if x["label"] == c]) for c in (0, 1)}}
            trial_summaries.append(item)
            all_rows.extend(rows)
            for key, value in errors.items():
                max_errors[key] = max(max_errors.get(key, 0), value)
            print(json.dumps({"completed_cell": identity, "accuracy": accuracy,
                              "median_label_hint_cos": item["all_updates"]["cos_label_hint"]["median"]}), flush=True)
        csv_handle.close()
        for seed in (261161, 261167, 261173):
            for sign in (1, -1):
                check_budget()
                arrays = {}
                for arm in ("teacher", "shuffled"):
                    r = next(x for x in completed if (x["seed"], x["sign"], x["arm"]) == (seed, sign, arm))
                    arrays[arm] = load_npz(source / RUN / r["relative_directory"] / "learning-diagnostics.npz")
                for i in range(192):
                    row = {"seed": seed, "sign": sign, "update": i + 1, "window": i // 48 + 1}
                    for key, short in (("hint_banks", "hint"), ("eligibility", "eligibility"), ("directions", "direction"),
                                       ("actual_hidden_deltas", "physical"), ("weights_before", "weights_before")):
                        a, b = (arrays[arm][key][i, ROWS].astype(float) for arm in ("teacher", "shuffled"))
                        row[short + "_cosine"] = cosine(a, b)
                        row[short + "_relative_difference"] = relative(a, b)
                    a, b = (arrays[arm]["directions"][i, ROWS].astype(float) - arrays[arm]["hint_banks"][i, ROWS].astype(float)
                            for arm in ("teacher", "shuffled"))
                    row["label_component_relative_difference"] = relative(a, b)
                    pair_rows.append(row)
        with (output / "teacher-shuffled-paired-updates.csv").open("w", encoding="utf-8", newline="") as handle:
            pair_writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
            pair_writer.writeheader()
            pair_writer.writerows(pair_rows)
        groups = {}
        for sign in (1, -1):
            for arm in ("label", "teacher", "shuffled", "readout"):
                selected = [x for x in all_rows if x["sign"] == sign and x["arm"] == arm]
                groups[f"{sign}/{arm}"] = {"all_updates": summarize(selected),
                    "windows": {str(w): summarize([x for x in selected if x["window"] == w]) for w in (1, 2, 3, 4)},
                    "classes": {str(c): summarize([x for x in selected if x["label"] == c]) for c in (0, 1)}}
        paired = {str(sign): {"all_updates": summarize([x for x in pair_rows if x["sign"] == sign]),
                              "step1": summarize([x for x in pair_rows if x["sign"] == sign and x["update"] == 1]),
                              "later": summarize([x for x in pair_rows if x["sign"] == sign and x["update"] > 1])}
                  for sign in (1, -1)}
        result = {"audit_level": "same-Codex independent data-only arithmetic implementation",
                  "source_checkout_sha": source_sha, "trials": trial_summaries, "by_sign_arm": groups,
                  "teacher_shuffled_pairs": paired, "audit_max_errors": max_errors,
                  "audited_updates": len(all_rows), "seed_count": 3, "arm_run_count": 24,
                  "neural_forwards": 0, "hidden_training_updates": 0, "probe_fits": 0, "TEST_opened": False,
                  "G2": "FAIL_UNCHANGED", "failures": failures,
                  "not_measured": ["student full-episode saturation", "per-tick eligibility/modulation decomposition",
                                   "causal failure attribution", "seed-population confidence interval"]}
        atomic_json(output / "metrics.json", result)
        final_status["status"] = "COMPLETED" if not failures else "AUDIT_FAILED"
        final_status["audit_failures"] = failures
        final_status["audited_updates"] = len(all_rows)
    except Exception as exc:
        final_status["error"] = repr(exc)
        final_status["audit_failures"] = failures
    finally:
        info = process.memory_info()
        peak_rss = max(peak_rss, info.rss)
        peak_os = max(peak_os, getattr(info, "peak_wset", 0))
        resource = {"elapsed_seconds": time.perf_counter() - started, "sampled_peak_RSS_bytes": peak_rss,
                    "OS_peak_working_set_bytes": peak_os or None, "RSS_limit_bytes": 300_000_000,
                    "wall_limit_seconds": 60, "sampling": "at startup, per trial and per update",
                    "GPU_used": False, "pid": os.getpid(), "python": sys.executable,
                    "BLAS_OpenMP_threads": 1, "hard_RSS_cap_enforced": False}
        if resource["elapsed_seconds"] >= 60 or peak_rss >= 300_000_000:
            final_status["status"] = "RESOURCE_LIMIT_FAILED"
        atomic_json(output / "resource.json", resource)
        output_hashes = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "status.json"}
        atomic_json(output / "output-hashes.json", output_hashes)
        final_status["resource_receipt"] = "resource.json"
        final_status["finished_UTC"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "status.json", final_status)
        print(json.dumps(final_status), flush=True)
    return 0 if final_status["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
