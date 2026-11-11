"""Locked temporal-stress screen. Protocol r3b-v1. Hidden weights stay frozen."""
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

from scarx.evaluation.phase21 import interval
from scarx.evaluation.temporal_stress import (
    NormHead,
    accuracy,
    core_features,
    delay_set,
    fit_reader,
    length_set,
    make_core,
    noise_gap,
    order_pairs,
    order_with_distractors,
    paired_flip,
    query_set,
    ridge_on,
    rollout,
    sim_xor,
    stream_episodes,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "grok-r3b-temporal-stress-v1"
STAGE0_SEEDS = (81001, 81007, 81013)
SCREEN_SEEDS = (94011, 94017, 94023)
SUBSTRATES = ("decay025", "spikesum", "bank5")
READERS = ("raw", "norm")


def digest(values):
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def hidden_token(core):
    return hashlib.sha256(core.w_in.tobytes() + core.w_rec.tobytes()).hexdigest()


def features_of(core, sequences, substrate):
    built, _state = rollout(core, sequences)
    return built[substrate]


def train_eval(core, substrate, reader, train, train_y, test, test_y, seed):
    trained = features_of(core, train, substrate)
    held = features_of(core, test, substrate)
    head = fit_reader(reader, trained, train_y, seed)
    score, predicted = accuracy(head, held, test_y)
    ridge = ridge_on(trained, train_y, held, test_y) if reader == "norm" else None
    return head, score, predicted, ridge, trained, held


def stage0():
    rows = []
    for seed in STAGE0_SEEDS:
        core = make_core(seed)
        token = hidden_token(core)
        calibration, _unused = sim_xor(seed + 100_000, 128)
        train, train_y = sim_xor(seed + 200_000, 1024)
        test, test_y = sim_xor(seed + 300_000, 512)
        for substrate in ("decay025", "spikesum"):
            head, score, predicted, ridge, trained, held = train_eval(
                core, substrate, "norm", train, train_y, test, test_y, seed)
            rows.append({
                "seed": seed,
                "substrate": substrate,
                "ridge": ridge,
                "norm_head": score,
                "prediction_sha256": digest(predicted),
                "feature_sha256": digest(held[:4]),
                "calibration_count": 128,
                "calibration_sha256": digest(features_of(core, calibration, substrate)[:8]),
                "norm_count_after_eval": head.count,
            })
        if hidden_token(core) != token:
            raise RuntimeError("Hidden weights changed")
    ridge_sum = np.mean([row["ridge"] for row in rows if row["substrate"] == "spikesum"])
    ridge_decay = np.mean([row["ridge"] for row in rows if row["substrate"] == "decay025"])
    head_decay = np.mean([row["norm_head"] for row in rows if row["substrate"] == "decay025"])
    passed = bool(ridge_sum >= 0.80 and ridge_sum - ridge_decay >= 0.15 and head_decay < 0.65)
    return {
        "rows": rows,
        "ridge_spikesum_mean": float(ridge_sum),
        "ridge_decay025_mean": float(ridge_decay),
        "norm_decay025_mean": float(head_decay),
        "contrast_reproduced": passed,
    }


def mean_interval(values):
    return interval(values)


def screen_order(core, seed):
    train, train_y, _other, _other_y = order_pairs(94102, 128)
    twin_a, labels_a, twin_b, labels_b = order_pairs(94104, 128)
    mixed_x = np.concatenate((train, _other))
    mixed_y = np.concatenate((train_y, _other_y))
    order = np.random.default_rng(seed + 3).permutation(len(mixed_y))
    results = {}
    for substrate in SUBSTRATES:
        trained = features_of(core, mixed_x[order], substrate)
        head = fit_reader("norm", trained, mixed_y[order], seed)
        first = features_of(core, twin_a, substrate)
        second = features_of(core, twin_b, substrate)
        results[substrate] = {
            "correct_flip": paired_flip(head, first, labels_a, second, labels_b),
            "feature_l1": float(np.mean(np.abs(first - second))),
        }
    return results


def screen_delay(core, seed):
    parts = [delay_set(94102, 64, tick) for tick in (3, 6, 11)]
    train = np.concatenate([part[0] for part in parts])
    train_y = np.concatenate([part[1] for part in parts])
    out = {}
    for substrate in SUBSTRATES:
        head = fit_reader("norm", features_of(core, train, substrate), train_y, seed)
        out[substrate] = {}
        for tick in (3, 6, 11):
            test, test_y = delay_set(94104, 128, tick)
            score, _predicted = accuracy(head, features_of(core, test, substrate), test_y)
            out[substrate][str(tick)] = score
    return out


def screen_distractor(core, seed):
    train, train_y = order_with_distractors(94102, 256, 0)
    out = {}
    for substrate in SUBSTRATES:
        head = fit_reader("norm", features_of(core, train, substrate), train_y, seed)
        out[substrate] = {}
        for extras in (0, 1, 2):
            test, test_y = order_with_distractors(94104, 128, extras)
            score, _predicted = accuracy(head, features_of(core, test, substrate), test_y)
            out[substrate][str(extras)] = score
    return out


def screen_query(core, seed):
    train, train_y, _first = query_set(94102, 256)
    test, test_y, first = query_set(94104, 128)
    out = {"first_key_policy": float(np.mean(first == test_y))}
    for substrate in SUBSTRATES:
        head = fit_reader("norm", features_of(core, train, substrate), train_y, seed)
        score, _predicted = accuracy(head, features_of(core, test, substrate), test_y)
        out[substrate] = score
    return out


def screen_length(core, seed):
    train, train_y = length_set(94102, 256, 12, False)
    out = {}
    for substrate in SUBSTRATES:
        head = fit_reader("norm", features_of(core, train, substrate), train_y, seed)
        out[substrate] = {}
        for length in (8, 12, 16):
            for pad_start in (False, True):
                test, test_y = length_set(94104, 128, length, pad_start)
                score, _predicted = accuracy(head, features_of(core, test, substrate), test_y)
                out[substrate][f"len{length}_{'start' if pad_start else 'end'}"] = score
    return out


def stream_score(core, head, substrate, episodes, mode):
    voltage = adaptation = spikes = None
    correct = {name: [] for name in ("sim_xor", "order", "delay")}
    for index, (name, sequence, label) in enumerate(episodes):
        if mode == "A":
            voltage = adaptation = spikes = None
        elif mode == "C":
            _ignored, (voltage, adaptation, spikes) = rollout(
                core, noise_gap(94180, index), None if voltage is None else (voltage, adaptation, spikes))
        built, (voltage, adaptation, spikes) = rollout(
            core, sequence, None if voltage is None else (voltage, adaptation, spikes))
        correct[name].append(int(head[name].predict(built[substrate]) == label))
    return {name: float(np.mean(values)) for name, values in correct.items()}


def screen_stream(cores, process):
    episodes = stream_episodes(94180, 1000)
    makers = {
        "sim_xor": lambda: sim_xor(94102, 256),
        "order": _stack_order,
        "delay": lambda: delay_set(94102, 256, 6),
    }
    collected = {substrate: {mode: [] for mode in "ABC"} for substrate in ("decay025", "spikesum")}
    before = process.memory_info().rss
    for _seed, core in cores.items():
        for substrate in ("decay025", "spikesum"):
            heads = {}
            for name, maker in makers.items():
                data, labels = maker()
                heads[name] = fit_reader("norm", features_of(core, data, substrate), labels, _seed)
            for mode in "ABC":
                collected[substrate][mode].append(stream_score(core, heads, substrate, episodes, mode))
    growth = int(process.memory_info().rss - before)
    report = {
        "episodes": 1000,
        "feature_width": {"decay025": 48, "spikesum": 48, "bank5": 240},
        "rss_growth_bytes": growth,
    }
    for substrate, modes in collected.items():
        report[substrate] = {}
        for mode, rows in modes.items():
            report[substrate][mode] = {
                name: float(np.mean([row[name] for row in rows]))
                for name in ("sim_xor", "order", "delay")
            }
            report[substrate][mode + "_seeds"] = rows
    return report


def _stack_order():
    early, early_y, late, late_y = order_pairs(94102, 128)
    return np.concatenate((early, late)), np.concatenate((early_y, late_y))


def leakage_controls(core):
    train, train_y = sim_xor(94102, 256)
    test, test_y = sim_xor(94104, 256)
    features = features_of(core, train, "spikesum")
    held = features_of(core, test, "spikesum")
    head = NormHead(features.shape[1], 94011)
    for row in features:
        head.observe(row)
    head.freeze()
    before = head.mean.copy()
    count = head.count
    for row in held:
        head.transform(row)
    clean = bool(np.array_equal(before, head.mean) and head.count == count)
    shuffled = train_y.copy()
    rng = np.random.default_rng(7)
    rng.shuffle(shuffled)
    leaky = fit_reader("norm", features, shuffled, 94011)
    random_score, _predicted = accuracy(leaky, held, test_y)
    silent = np.zeros((256, 12, 8), np.float32)
    silent_labels = np.arange(256) % 2
    silent_train, silent_test = silent[:128], silent[128:]
    silent_y, silent_test_y = silent_labels[:128], silent_labels[128:]
    blank = fit_reader(
        "norm", features_of(core, silent_train, "spikesum"), silent_y, 94011)
    blank_score, _predicted = accuracy(
        blank, features_of(core, silent_test, "spikesum"), silent_test_y)
    return {
        "predict_does_not_change_calibration": clean,
        "random_label_head": float(random_score),
        "no_input_head": float(blank_score),
    }


def parity():
    core = make_core(94001)
    rng = np.random.default_rng(4)
    sequences = rng.normal(0, 0.2, (4, 12, 8)).astype(np.float32)
    reference = np.stack([core_features(core, sequence) for sequence in sequences])
    mine, _state = rollout(core, sequences)
    gap = float(np.max(np.abs(reference - mine["decay025"])))
    state_a, (voltage, adaptation, spikes) = rollout(core, sequences[:, :6])
    continued, _final = rollout(core, sequences[:, 6:], (voltage, adaptation, spikes))
    whole, _whole_state = rollout(core, sequences)
    # Continued decay features are not equal to the suffix of a whole-window feature.
    # Resume parity is on the carried voltage after the same 12 steps.
    resumed, final = rollout(core, sequences[:, :6])
    full_state_steps = None
    voltage = adaptation = spikes = None
    for step in range(12):
        _built, (voltage, adaptation, spikes) = rollout(
            core, sequences[:, step:step + 1],
            None if voltage is None else (voltage, adaptation, spikes))
    _built, (once_v, once_a, once_z) = rollout(core, sequences)
    resume_gap = float(max(
        np.max(np.abs(voltage - once_v)),
        np.max(np.abs(adaptation - once_a)),
        np.max(np.abs(spikes - once_z)),
    ))
    return {"feature_gap": gap, "resume_gap": resume_gap, "unused": float(np.max(np.abs(state_a["decay025"])))}


def evaluate(stage0, order_rows, delay_rows, query_rows, stream, controls, parity_report, rss_growth):
    order_values = [row["spikesum"]["correct_flip"] for row in order_rows]
    query_values = [row["spikesum"] for row in query_rows]
    order_interval = mean_interval(order_values)
    query_interval = mean_interval(query_values)
    short = [row["spikesum"]["3"] for row in delay_rows]
    long = [row["spikesum"]["11"] for row in delay_rows]
    reset_xor = stream["spikesum"]["A"]["sim_xor"]
    persistent_xor = stream["spikesum"]["B"]["sim_xor"]
    distract_xor = stream["spikesum"]["C"]["sim_xor"]
    reset_order = stream["spikesum"]["A"]["order"]
    persistent_order = stream["spikesum"]["B"]["order"]
    distract_order = stream["spikesum"]["C"]["order"]
    gates = {
        "G1_stage0_contrast": bool(stage0["contrast_reproduced"]),
        "G2_calibration": bool(controls["predict_does_not_change_calibration"]),
        "G3_order_flip": bool(order_interval["mean"] >= 0.65 and order_interval["lower95"] > 0.55),
        "G4_query": bool(
            query_interval["mean"] >= 0.65 and query_interval["lower95"] > 0.55
            and all(row["first_key_policy"] <= 0.55 for row in query_rows)),
        "G5_delay": bool(float(np.mean(short)) >= 0.65 and float(np.mean(long)) >= float(np.mean(short)) - 0.05),
        "G6_streaming": bool(
            persistent_xor >= reset_xor - 0.05 and distract_xor >= reset_xor - 0.05
            and persistent_order >= reset_order - 0.05 and distract_order >= reset_order - 0.05),
        "G7_memory": bool(rss_growth <= 50_000_000),
        "G8_parity": bool(parity_report["feature_gap"] < 1e-4 and parity_report["resume_gap"] < 1e-4),
        "leakage_random_label": bool(controls["random_label_head"] < 0.60),
        "leakage_no_input": bool(controls["no_input_head"] < 0.60),
    }
    passed = all(gates.values())
    return gates, "GO" if passed else "STOP", {
        "order_flip": order_interval,
        "query": query_interval,
        "delay_short_mean": float(np.mean(short)),
        "delay_long_mean": float(np.mean(long)),
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    process = psutil.Process()
    rss_start = process.memory_info().rss
    peak = rss_start
    clock = time.perf_counter()
    reproduced = stage0()
    peak = max(peak, process.memory_info().rss)
    status = "STOP"
    payload = {"stage0": reproduced}
    if reproduced["contrast_reproduced"] and time.perf_counter() - clock < 540:
        order_rows = []
        delay_rows = []
        distract_rows = []
        query_rows = []
        length_rows = []
        cores = {}
        for seed in SCREEN_SEEDS:
            core = make_core(seed)
            cores[seed] = core
            order_rows.append(screen_order(core, seed))
            delay_rows.append(screen_delay(core, seed))
            distract_rows.append(screen_distractor(core, seed))
            query_rows.append(screen_query(core, seed))
            length_rows.append(screen_length(core, seed))
            peak = max(peak, process.memory_info().rss)
        stream = screen_stream(cores, process)
        controls = leakage_controls(cores[94011])
        parity_report = parity()
        peak = max(peak, process.memory_info().rss)
        gates, status, summaries = evaluate(
            reproduced, order_rows, delay_rows, query_rows, stream, controls, parity_report,
            stream["rss_growth_bytes"])
        payload.update({
            "order": order_rows,
            "delay": delay_rows,
            "distractor": distract_rows,
            "query": query_rows,
            "length": length_rows,
            "stream": stream,
            "controls": controls,
            "parity": parity_report,
            "gates": gates,
            "summaries": summaries,
        })
    elif not reproduced["contrast_reproduced"]:
        status = "STOP"
        payload["stopped_after"] = "stage0"
    else:
        status = "BLOCKED"
        payload["stopped_after"] = "time"
    info = process.memory_info()
    payload.update({
        "status": status,
        "confirmation_opened": False,
        "seconds": time.perf_counter() - clock,
        "peak_rss_bytes_sampled": int(peak),
        "rss_growth_bytes": int(process.memory_info().rss - rss_start),
        "peak_wset_bytes": int(getattr(info, "peak_wset", 0)),
        "protocol": "experiments/collab/scarx-neural-round1-20261111/grok/protocol-r3b-v1.md",
    })
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "stage0": {
            "ridge_spikesum_mean": reproduced["ridge_spikesum_mean"],
            "ridge_decay025_mean": reproduced["ridge_decay025_mean"],
            "norm_decay025_mean": reproduced["norm_decay025_mean"],
            "contrast_reproduced": reproduced["contrast_reproduced"],
        },
        "seconds": payload["seconds"],
        "peak_rss_bytes_sampled": payload["peak_rss_bytes_sampled"],
    }, indent=2))


if __name__ == "__main__":
    main()
