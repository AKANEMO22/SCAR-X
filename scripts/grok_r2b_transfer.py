"""Round 2b audit and one locked transfer screen.

Protocol: experiments/collab/scarx-neural-round1-20261111/grok/protocol-r2b-v1.md
Stage 2 does not start if the surrogate finite-difference check fails.
"""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import psutil

from scarx.evaluation.binding_audit21 import strict_binding
from scarx.evaluation.dc_alif_eligibility import (
    DCALIFEligibility,
    compute_coact,
    dc_alif_finite_difference_check,
)
from scarx.evaluation.phase21 import interval
from scarx.evaluation.readout_eligibility import ReadoutEligibility
from scarx.evaluation.tasks21 import representation, ridge_probe
from scarx.neural.core import Config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "grok-r2b-transfer-v1"
CKPT = Path(
    r"C:\Users\hachimi\Downloads\model agent\scar-x-antigravity-round1"
    r"\artifacts\collab\scarx-neural-round1-20261111\antigravity\checkpoints"
)
CHECKPOINT_SEED = 35201
STAGE1_TASKS = ("order", "xor", "strict_binding")
STAGE2_TASKS = ("order", "xor", "strict_binding", "pair_binding")
STAGE2_SEEDS = (93011, 93017, 93023)
LESIONS = ("", "drop_coincidence", "drop_quadratic", "zero_plateau", "open_gate", "shuffle_gate")
ARMS = ("correct", "frozen", "shuffled", "zero", "no_gate")


def executed_config(seed):
    config = Config(
        inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0
    )
    if config.leak != 0.85 or config.feature_decay != 0.92 or config.neuron != "alif":
        raise ValueError("Executed Config defaults changed")
    return config


def pair_binding(seed, count):
    rng = np.random.default_rng(seed)
    labels = np.arange(count) % 2
    values = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    for i in range(count):
        stored = {0: int(labels[i]), 1: 1 - int(labels[i])}
        positions = (int(rng.integers(1, 3)), int(rng.integers(4, 7)))
        for position, key in zip(positions, rng.permutation(2)):
            key = int(key)
            values[i, position, key] += 1.5
            values[i, position, 2 + stored[key]] += 1.5
    order = rng.permutation(count)
    return values[order], labels[order].astype(np.int32)


def generate(task, seed, count):
    if task == "strict_binding":
        return strict_binding(seed, count)
    if task == "pair_binding":
        return pair_binding(seed, count)
    if task in ("order", "xor"):
        return representation(task, seed, count)
    raise ValueError(task)


def code_q(model, weights, sample):
    hat = weights @ model.P
    current = hat * sample
    total = float(np.sum(current))
    pairs = current.reshape(-1, 2).sum(axis=1)
    return 0.5 * (total * total - float(np.sum(pairs * pairs)))


def documented_q(weights, sample):
    terms = weights * sample
    total = float(np.sum(terms))
    return 0.5 * (total * total - float(np.sum(terms * terms)))


def formula_gap(model):
    rng = np.random.default_rng(93000)
    gaps = []
    for _ in range(32):
        sample = rng.normal(0, 0.4, model.core.config.inputs).astype(np.float64)
        sample[rng.integers(0, sample.size)] += 1.5
        weights = model.core.w_in[0].astype(np.float64)
        gaps.append(abs(code_q(model, weights, sample) - documented_q(weights, sample)))
    return {
        "mean_abs_gap": float(np.mean(gaps)),
        "max_abs_gap": float(np.max(gaps)),
        "coact_depends_on_weights": False,
        "coact_example": compute_coact(rng.normal(0, 0.4, 8)),
    }


def stage0():
    config = executed_config(94001)
    model = DCALIFEligibility(config, "correct")
    sequences, _labels = strict_binding(94011, 4)
    finite = dc_alif_finite_difference_check(model, sequences[0])
    constants = {
        "config_leak": config.leak,
        "protocol_theorem_leak": 0.88,
        "config_feature_decay": config.feature_decay,
        "config_threshold": config.threshold,
        "config_adaptation_strength": config.adaptation_strength,
        "alpha_s": DCALIFEligibility.alpha_s,
        "alpha_d": DCALIFEligibility.alpha_d,
        "gamma_q": DCALIFEligibility.gamma_q,
        "beta_g": DCALIFEligibility.beta_g,
        "beta_d": DCALIFEligibility.beta_d,
        "d_decay": DCALIFEligibility.d_decay,
        "sampler": "hard threshold u >= theta",
        "finite_difference_scope": finite["oracle_scope"],
        "max_absolute_error_e_f": finite["max_absolute_error_e_f"],
        "max_absolute_error_e_d": finite["max_absolute_error_e_d"],
        "formula_gap": formula_gap(model),
    }
    constants["surrogate_check_passes"] = (
        finite["max_absolute_error_e_f"] < 1e-5 and finite["max_absolute_error_e_d"] < 1e-5
    )
    return constants


def load_checkpoint(task, arm):
    path = CKPT / f"confirmation-{task}-{CHECKPOINT_SEED}-{arm}.dc_alif.zip"
    payload = path.read_bytes()
    model = DCALIFEligibility.loads(payload)
    model.diagnostic_lesion = ""
    model.diagnostic_record = False
    return model, hashlib.sha256(payload).hexdigest()


def score_model(model, sequences, labels):
    predicted = np.asarray([model.predict(sequence) for sequence in sequences])
    return float(np.mean(predicted == labels)), predicted.astype(int).tolist()


def plateau_probe(model, sequences, labels):
    model.diagnostic_record = True
    model.diagnostic_lesion = ""
    early = []
    late = []
    final = []
    for sequence in sequences:
        model.diagnostic_trace = []
        model.begin(False)
        for row in sequence:
            model.step(row)
        model.segment_active = False
        trace = model.diagnostic_trace
        early.append(trace[9][0])
        late.append(trace[min(10, len(trace) - 1)][0])
        final.append(model.core.features.copy())
    model.diagnostic_record = False
    model.diagnostic_trace = []
    fit = slice(0, 64)
    test = slice(64, None)
    return {
        "plateau_tick9": ridge_probe(np.stack(early)[fit], labels[fit], np.stack(early)[test], labels[test]),
        "plateau_tick10": ridge_probe(np.stack(late)[fit], labels[fit], np.stack(late)[test], labels[test]),
        "end_features": ridge_probe(np.stack(final)[fit], labels[fit], np.stack(final)[test], labels[test]),
    }


def stage1():
    report = {}
    for task in STAGE1_TASKS:
        sequences, labels = generate(task, 94004, 96)
        task_report = {"checkpoint_sha256": {}, "arms": {}, "lesions_on_correct": {}, "lesions_on_frozen": {}}
        loaded = {}
        for arm in ARMS:
            model, digest = load_checkpoint(task, arm)
            loaded[arm] = model
            task_report["checkpoint_sha256"][arm] = digest
            accuracy, _predictions = score_model(model, sequences, labels)
            task_report["arms"][arm] = accuracy
        for arm in ("correct", "frozen"):
            task_report["probes_" + arm] = plateau_probe(loaded[arm], sequences, labels)
            for lesion in LESIONS:
                loaded[arm].diagnostic_lesion = lesion
                loaded[arm].diagnostic_permutation = None
                accuracy, _predictions = score_model(loaded[arm], sequences, labels)
                key = "intact" if lesion == "" else lesion
                task_report["lesions_on_" + arm][key] = accuracy
            loaded[arm].diagnostic_lesion = ""
        report[task] = task_report
    return report


def run_arm(seed, arm, task, calibration, train, train_y, dev, dev_y):
    if arm == "baseline_elig":
        model = ReadoutEligibility(executed_config(seed), "correct")
    else:
        model = DCALIFEligibility(executed_config(seed), arm)
    model.calibrate(list(calibration))
    before = model.core.w_in.copy()
    for sequence, label in zip(train, train_y):
        model.learn(sequence, int(label))
    accuracy, predictions = score_model(model, dev, dev_y)
    return {
        "accuracy": accuracy,
        "hidden_changed": bool(np.any(model.core.w_in != before)),
        "predictions": predictions,
    }


def stage2(clock):
    projected_unit = time.perf_counter()
    probe = DCALIFEligibility(executed_config(93001), "frozen")
    sample, sample_y = generate("order", 93008, 16)
    probe.calibrate(list(generate("order", 93007, 128)[0]))
    for sequence, label in zip(sample, sample_y):
        probe.learn(sequence, int(label))
    elapsed_16 = time.perf_counter() - projected_unit
    projected = elapsed_16 / 16.0 * (128 * len(STAGE2_SEEDS) * len(STAGE2_TASKS) * 6)
    if projected > 540.0 or time.perf_counter() - clock > 540.0:
        return {"status": "BLOCKED", "projected_seconds": projected, "timing_16_learns_seconds": elapsed_16}, projected
    rows = {task: [] for task in STAGE2_TASKS}
    for task in STAGE2_TASKS:
        calibration, _unused = generate(task, 93003, 128)
        train, train_y = generate(task, 93002, 128)
        dev, dev_y = generate(task, 93004, 128)
        for seed in STAGE2_SEEDS:
            if time.perf_counter() - clock > 540.0:
                return {"status": "BLOCKED", "projected_seconds": projected, "partial": rows}, projected
            row = {"seed": seed}
            for arm in ("correct", "frozen", "shuffled", "zero", "no_gate", "baseline_elig"):
                scored = run_arm(seed, arm, task, calibration, train, train_y, dev, dev_y)
                row[arm] = scored["accuracy"]
                row[arm + "_hidden_changed"] = scored["hidden_changed"]
                row[arm + "_predictions"] = scored["predictions"]
            rows[task].append(row)
    summary = {}
    learn_tasks = []
    specific_tasks = []
    direct_tasks = []
    regressions = []
    for task, task_rows in rows.items():
        correct = [row["correct"] for row in task_rows]
        frozen = [row["frozen"] for row in task_rows]
        shuffled = [row["shuffled"] for row in task_rows]
        baseline = [row["baseline_elig"] for row in task_rows]
        no_gate = [row["no_gate"] for row in task_rows]
        gain_frozen = interval([a - b for a, b in zip(correct, frozen)])
        gain_shuffled = interval([a - b for a, b in zip(correct, shuffled)])
        gain_gate = interval([a - b for a, b in zip(correct, no_gate)])
        gain_base = interval([a - b for a, b in zip(correct, baseline)])
        if float(np.mean(correct)) >= 0.65:
            direct_tasks.append(task)
        if gain_frozen["mean"] >= 0.05 and gain_frozen["lower95"] > 0:
            learn_tasks.append(task)
        if gain_shuffled["lower95"] > 0:
            specific_tasks.append(task)
        if float(np.mean(correct)) < float(np.mean(baseline)) - 0.03:
            regressions.append(task)
        summary[task] = {
            "seeds": task_rows,
            "correct_mean": float(np.mean(correct)),
            "frozen_mean": float(np.mean(frozen)),
            "shuffled_mean": float(np.mean(shuffled)),
            "zero_mean": float(np.mean([row["zero"] for row in task_rows])),
            "no_gate_mean": float(np.mean(no_gate)),
            "baseline_elig_mean": float(np.mean(baseline)),
            "correct_minus_frozen": gain_frozen,
            "correct_minus_shuffled": gain_shuffled,
            "correct_minus_no_gate": gain_gate,
            "correct_minus_baseline_elig": gain_base,
        }
    shared = [task for task in learn_tasks if task in specific_tasks]
    passed = len(direct_tasks) >= 2 and len(learn_tasks) >= 2 and len(shared) >= 2 and not regressions
    return {
        "status": "GO" if passed else "STOP",
        "projected_seconds": projected,
        "timing_16_learns_seconds": elapsed_16,
        "direct_tasks": direct_tasks,
        "learning_tasks": learn_tasks,
        "specific_tasks": specific_tasks,
        "regressions_versus_baseline": regressions,
        "tasks": summary,
        "confirmation_opened": False,
    }, projected


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    process = psutil.Process()
    peak = process.memory_info().rss
    clock = time.perf_counter()
    audit = stage0()
    peak = max(peak, process.memory_info().rss)
    diagnosis = None
    screen = None
    if audit["surrogate_check_passes"]:
        diagnosis = stage1()
        peak = max(peak, process.memory_info().rss)
        screen, _projected = stage2(clock)
        peak = max(peak, process.memory_info().rss)
    else:
        screen = {"status": "STOP", "reason": "surrogate finite-difference check failed"}
    info = process.memory_info()
    report = {
        "protocol": "experiments/collab/scarx-neural-round1-20261111/grok/protocol-r2b-v1.md",
        "base_commit": "81641f773145f5c441b5ad3b0f5771f645a5cbd6",
        "stage0": audit,
        "stage1_checkpoint_seed": CHECKPOINT_SEED,
        "stage1": diagnosis,
        "stage2": screen,
        "seconds": time.perf_counter() - clock,
        "peak_rss_bytes_sampled": int(peak),
        "peak_wset_bytes": int(getattr(info, "peak_wset", 0)),
        "child_processes": len(process.children(recursive=True)),
    }
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "surrogate_passes": audit["surrogate_check_passes"],
        "leak": audit["config_leak"],
        "formula_gap_max": audit["formula_gap"]["max_abs_gap"],
        "stage2_status": screen.get("status"),
        "seconds": report["seconds"],
        "peak_rss_bytes_sampled": report["peak_rss_bytes_sampled"],
    }, indent=2))


if __name__ == "__main__":
    main()
