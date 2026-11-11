"""CORRECTED_REPLAY of the R4A shift, then the locked safe-normalization screen."""
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

from scarx.evaluation import r4a_probe as probe
from scarx.evaluation import r4a_relational as rel
from scarx.evaluation.phase21 import interval
from scarx.evaluation.r5_audit import CONF_SEEDS, cue_at, fit_safe, historical_shift, pure_shift, r5_features
from scarx.evaluation.readout_mechanism import (
    delay_set, evaluate, fit_arm, make_core, order_index, order_pairs, paired_flip,
    query_set, ridge_score, rollout, separated_xor, sim_xor,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "grok-r5-audit-v1"
HISTORICAL = {
    96001: 0.61328125, 96007: 0.705078125, 96013: 0.677734375, 96019: 0.59375,
    96023: 0.65234375, 96029: 0.62890625, 96031: 0.677734375, 96037: 0.56640625,
}
SCREEN = (98011, 98017, 98023)


def r5_cell(seed, train_x, train_y, test_x, test_y, capture_tick, cue_tick):
    train_f, _one = r5_features(seed, train_x, capture_tick, cue_tick)
    test_f, _two = r5_features(seed, test_x, capture_tick, cue_tick)
    head = probe.train_head(train_f, train_y, seed=seed)
    predicted = head.predict(test_f)
    labels = np.asarray(test_y)
    pairs, pair_y = predicted.reshape(-1, 2), labels.reshape(-1, 2)
    valid = pair_y[:, 0] != pair_y[:, 1]
    consistency = float(np.mean(pairs[valid, 0] != pairs[valid, 1])) if np.any(valid) else 0.0
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "train_accuracy": float(head.acc(train_f, train_y)),
        "consistency": consistency,
        "cue_key1_fraction": float(np.mean(cue_at(test_x, cue_tick)[:, 1])),
        "zero_tick_fraction": float(np.mean(np.all(np.asarray(test_x) == 0, axis=2))),
    }


def replay(seed):
    train_x, train_y, _pid, _flag = rel.paired_counterfactual(seed + 200_000, 512)
    test_x, test_y, _pid, _flag = rel.paired_counterfactual(seed + 300_000, 256)
    published_train, published_test = historical_shift(train_x), historical_shift(test_x)
    pure_train, pure_test = pure_shift(train_x), pure_shift(test_x)
    clean_f, _cap = r5_features(seed, train_x, 10, 10)
    head = probe.train_head(clean_f, train_y, seed=seed)
    pure_f, _pure = r5_features(seed, pure_test, 11, 11)
    return {
        "classification": "CORRECTED_REPLAY",
        "historical_published_r5": HISTORICAL[seed],
        "F1_published_cue": r5_cell(seed, published_train, train_y, published_test, test_y, 11, 10),
        "F2_corrected_cue": r5_cell(seed, published_train, train_y, published_test, test_y, 11, 11),
        "F3_pure_shift": r5_cell(seed, pure_train, train_y, pure_test, test_y, 11, 11),
        "F4_train_clean_test_pure": {
            "accuracy": float(np.mean(head.predict(pure_f) == test_y)),
            "train_accuracy": float(head.acc(clean_f, train_y)),
            "cue_key1_fraction": float(np.mean(cue_at(pure_test, 11)[:, 1])),
            "zero_tick_fraction": float(np.mean(np.all(pure_test == 0, axis=2))),
        },
    }


def order_row(core, seed, mode, order_kind, train_seed, test_seed):
    packed, test = order_pairs(train_seed, 128), order_pairs(test_seed, 128)
    train_x = np.concatenate((packed[0], packed[2]))
    train_y = np.concatenate((packed[1], packed[3]))
    trained = rollout(core, train_x)["decay025"]
    left, right = rollout(core, test[0])["decay025"], rollout(core, test[2])["decay025"]
    held, labels = np.concatenate((left, right)), np.concatenate((test[1], test[3]))
    ridge = ridge_score(trained, train_y, held, labels)
    if mode == "safe":
        head = fit_safe(trained[order_index(train_y, order_kind)], train_y[order_index(train_y, order_kind)], seed)
        flip = sum(int(head.predict_many([a])[0] == int(ya) and head.predict_many([b])[0] == int(yb))
                   for a, ya, b, yb in zip(left, test[1], right, test[3])) / len(test[1])
        return {"paired_flip": flip, "accuracy": float(np.mean(head.predict_many(held) == labels)),
                "ridge": ridge, "finite": bool(np.isfinite(head.weights).all())}
    head, stats = fit_arm(trained, train_y, seed, mode, order_kind)
    return {"paired_flip": paired_flip(head, left, test[1], right, test[3]),
            "accuracy": evaluate(head, held, labels)["accuracy"], "train_accuracy": stats["train_accuracy"],
            "clip_fraction": stats["clip_fraction"], "ridge": ridge, "finite": True}


def table(core, seed, train_x, train_y, test_x, test_y, order_kind):
    trained, held = rollout(core, train_x)["decay025"], rollout(core, test_x)["decay025"]
    order = order_index(train_y, order_kind)
    norm, _stats = fit_arm(trained, train_y, seed, "norm", order_kind)
    safe = fit_safe(trained[order], train_y[order], seed)
    return {"norm": evaluate(norm, held, test_y)["accuracy"],
            "safe": float(np.mean(safe.predict_many(held) == np.asarray(test_y))),
            "ridge": ridge_score(trained, train_y, held, test_y),
            "finite": bool(np.isfinite(safe.weights).all())}


def screen(seed, order_kind):
    core = make_core(seed)
    out = {"order": {"norm": order_row(core, seed, "norm", order_kind, 96102, 96104),
                     "safe": order_row(core, seed, "safe", order_kind, 96102, 96104)}}
    specs = {
        "sim_xor": (sim_xor(96102, 256), sim_xor(96104, 128)),
        "separated_xor": (separated_xor(96102, 256), separated_xor(96104, 128)),
        "delay": (delay_set(96102, 256, 6), delay_set(96104, 128, 11)),
        "query": (query_set(96102, 256)[:2], query_set(96104, 128)[:2]),
    }
    for name, (train, test) in specs.items():
        out[name] = table(core, seed, train[0], train[1], test[0], test[1], order_kind)
    base, labels, _late, _late_y = order_pairs(96104, 128)
    pulsed = base.copy()
    pulsed[:, 4, 4] += 1.5
    train = order_pairs(96102, 128)
    out["distractor"] = table(
        core, seed, np.concatenate((train[0], train[2])), np.concatenate((train[1], train[3])),
        pulsed, labels, order_kind)
    return out


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    process = psutil.Process()
    peak = process.memory_info().rss
    clock = time.perf_counter()
    forensic = [{"seed": seed, **replay(seed)} for seed in CONF_SEEDS]
    peak = max(peak, process.memory_info().rss)
    historical_rows = []
    for seed in (94011, 94017, 94023):
        core = make_core(seed)
        historical_rows.append({
            "seed": seed,
            "norm": order_row(core, seed, "norm", "original", 94102, 94104),
            "safe": order_row(core, seed, "safe", "original", 94102, 94104),
            "raw": order_row(core, seed, "raw", "original", 94102, 94104),
        })
    fresh, permuted = [], []
    for seed in SCREEN:
        fresh.append({"seed": seed, **screen(seed, "original")})
        permuted.append({"seed": seed, "order": screen(seed, "perm")["order"]})
        peak = max(peak, process.memory_info().rss)
    gain = interval([row["order"]["safe"]["paired_flip"] - row["order"]["norm"]["paired_flip"] for row in fresh])
    gain_perm = interval([row["order"]["safe"]["paired_flip"] - row["order"]["norm"]["paired_flip"] for row in permuted])
    other, qualified, regressions = {}, [], []
    for task in ("sim_xor", "separated_xor", "delay", "query", "distractor"):
        base = [row[task]["norm"] for row in fresh]
        cand = [row[task]["safe"] for row in fresh]
        ridge = [row[task]["ridge"] for row in fresh]
        paired = interval([a - b for a, b in zip(cand, base)])
        other[task] = {"baseline": float(np.mean(base)), "candidate": float(np.mean(cand)),
                       "ridge": float(np.mean(ridge)), "gain": paired}
        if float(np.mean(ridge)) >= 0.80 and paired["mean"] >= 0.10 and paired["lower95"] > 0:
            qualified.append(task)
        if float(np.mean(cand)) < float(np.mean(base)) - 0.03:
            regressions.append(task)
    finite = all(row["order"]["safe"]["finite"] and all(row[task]["finite"] for task in other) for row in fresh)
    status = "GO" if (
        gain["mean"] >= 0.10 and gain["lower95"] > 0 and len(qualified) >= 1 and not regressions
        and finite and gain_perm["mean"] >= 0.10 and gain_perm["lower95"] > 0
    ) else "STOP"
    payload = {
        "forensic_class": "CORRECTED_REPLAY",
        "historical_shift_mean": 0.639404296875,
        "forensic": forensic,
        "forensic_means": {name: float(np.mean([row[name]["accuracy"] for row in forensic]))
                           for name in ("F1_published_cue", "F2_corrected_cue", "F3_pure_shift", "F4_train_clean_test_pure")},
        "historical_order": historical_rows,
        "screen": fresh,
        "screen_perm_order": permuted,
        "order_gain": gain,
        "perm_order_gain": gain_perm,
        "other_tasks": other,
        "qualified_other_tasks": qualified,
        "regressions": regressions,
        "status": status,
        "confirmation_opened": False,
        "seconds": time.perf_counter() - clock,
        "peak_rss_bytes_sampled": int(max(peak, process.memory_info().rss)),
        "peak_wset_bytes": int(getattr(process.memory_info(), "peak_wset", 0)),
    }
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({
        "protocol": "experiments/collab/scarx-neural-round1-20261111/grok-r5/protocol-v1.md",
        "r4a_source_commit": "10e1c7df9a585566db04e754af9ba7b419d8bab1",
        "forensic_class": "CORRECTED_REPLAY",
        "status": status,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "means": payload["forensic_means"],
        "order_gain": gain,
        "qualified": qualified,
        "regressions": regressions,
        "seconds": payload["seconds"],
        "peak_rss_bytes_sampled": payload["peak_rss_bytes_sampled"],
    }, indent=2))


if __name__ == "__main__":
    main()
