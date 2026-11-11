"""Registered readout-mechanism run. Protocol v1 decides the candidate."""
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
from scarx.evaluation.readout_mechanism import (
    ARMS,
    delay_set,
    evaluate,
    fit_arm,
    geometry,
    make_core,
    order_pairs,
    order_with_distractors,
    paired_flip,
    query_set,
    query_swap,
    ridge_score,
    rollout,
    separated_xor,
    sim_xor,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "grok-r4-readout-v1"
REPRO_SEEDS = (94011, 94017, 94023)
XOR_SEEDS = (81001, 81007, 81013)
SCREEN_SEEDS = (96011, 96017, 96023)
SELECTABLE = ("noclip", "nofloor", "nonorm", "causal")


def take(core, sequences, name):
    return rollout(core, sequences)[name]


def hidden(core):
    return core.w_in.tobytes() + core.w_rec.tobytes()


def pack_order(train_seed, test_seed, count):
    early, early_y, late, late_y = order_pairs(train_seed, count)
    train_x = np.concatenate((early, late))
    train_y = np.concatenate((early_y, late_y))
    first, first_y, second, second_y = order_pairs(test_seed, count)
    return train_x, train_y, first, first_y, second, second_y


def score_order(core, seed, mode, order_kind, train_seed, test_seed, count, substrate):
    train_x, train_y, first, first_y, second, second_y = pack_order(train_seed, test_seed, count)
    trained = take(core, train_x, substrate)
    left = take(core, first, substrate)
    right = take(core, second, substrate)
    head, train = fit_arm(trained, train_y, seed, mode, order_kind)
    held = np.concatenate((left, right))
    labels = np.concatenate((first_y, second_y))
    test = evaluate(head, held, labels)
    test["paired_flip"] = paired_flip(head, left, first_y, right, second_y)
    test["ridge"] = ridge_score(trained, train_y, held, labels)
    test["train"] = train
    return test


def score_table(core, seed, mode, order_kind, train_x, train_y, test_x, test_y, substrate):
    trained = take(core, train_x, substrate)
    held = take(core, test_x, substrate)
    head, train = fit_arm(trained, train_y, seed, mode, order_kind)
    test = evaluate(head, held, test_y)
    test["ridge"] = ridge_score(trained, train_y, held, test_y)
    test["train"] = train
    return test


def reproduce():
    rows = []
    for seed in REPRO_SEEDS:
        core = make_core(seed)
        token = hidden(core)
        for substrate in ("decay025", "bank5"):
            scored = score_order(core, seed, "norm", "original", 94102, 94104, 128, substrate)
            rows.append({"seed": seed, "substrate": substrate, **{key: scored[key] for key in ("paired_flip", "ridge", "accuracy", "loss")}, "train_accuracy": scored["train"]["train_accuracy"], "clip_fraction": scored["train"]["clip_fraction"]})
        if hidden(core) != token:
            raise RuntimeError("Hidden weights changed")
    decay_ridge = [row["ridge"] for row in rows if row["substrate"] == "decay025"]
    decay_head = [row["paired_flip"] for row in rows if row["substrate"] == "decay025"]
    bank_head = [row["paired_flip"] for row in rows if row["substrate"] == "bank5"]
    passed = float(np.mean(decay_ridge)) >= 0.80 and float(np.mean(decay_head)) <= 0.40 and float(np.mean(bank_head)) >= 0.90
    return {"rows": rows, "passed": passed}


def diagnose(core):
    train_x, train_y, first, first_y, second, second_y = pack_order(94102, 94104, 128)
    trained = take(core, train_x, "decay025")
    report = {"geometry": geometry(trained), "orders": {}}
    for mode in ("norm", "causal", "nonorm"):
        report["orders"][mode] = {}
        for kind in ("original", "perm", "balanced"):
            scored = score_order(core, 94011, mode, kind, 94102, 94104, 128, "decay025")
            report["orders"][mode][kind] = {
                "paired_flip": scored["paired_flip"],
                "train_accuracy": scored["train"]["train_accuracy"],
                "heldout_accuracy": scored["accuracy"],
                "clip_fraction": scored["train"]["clip_fraction"],
            }
    return report


def ablate():
    order_rows = []
    xor_rows = []
    for seed in REPRO_SEEDS:
        core = make_core(seed)
        per_arm = {}
        for mode in ARMS:
            scored = score_order(core, seed, mode, "original", 94102, 94104, 128, "decay025")
            per_arm[mode] = {"paired_flip": scored["paired_flip"], "train_accuracy": scored["train"]["train_accuracy"], "heldout_accuracy": scored["accuracy"], "clip_fraction": scored["train"]["clip_fraction"]}
        order_rows.append({"seed": seed, "arms": per_arm})
    for seed in XOR_SEEDS:
        core = make_core(seed)
        train_x, train_y = sim_xor(seed + 200_000, 256)
        test_x, test_y = sim_xor(seed + 300_000, 256)
        per_arm = {}
        for mode in ARMS:
            scored = score_table(core, seed, mode, "original", train_x, train_y, test_x, test_y, "decay025")
            per_arm[mode] = {"accuracy": scored["accuracy"], "train_accuracy": scored["train"]["train_accuracy"], "ridge": scored["ridge"]}
        xor_rows.append({"seed": seed, "arms": per_arm})
    return {"order": order_rows, "sim_xor": xor_rows}


def choose(ablation, reproduction):
    if not reproduction["passed"]:
        return None, "reproduction bar failed"
    base_train = np.mean([row["arms"]["norm"]["train_accuracy"] for row in ablation["order"]])
    if base_train >= 0.65:
        return None, "original head already fits the training rows"
    gains = {}
    for mode in SELECTABLE:
        order_gain = np.mean([row["arms"][mode]["paired_flip"] - row["arms"]["norm"]["paired_flip"] for row in ablation["order"]])
        xor_drop = np.mean([row["arms"]["norm"]["accuracy"] - row["arms"][mode]["accuracy"] for row in ablation["sim_xor"]])
        gains[mode] = {"order_gain": float(order_gain), "xor_drop": float(xor_drop)}
    eligible = [mode for mode, item in gains.items() if item["order_gain"] >= 0.10 and item["xor_drop"] <= 0.03]
    if not eligible:
        return None, {"reason": "no single factor gained 0.10 without a sim_xor regression", "gains": gains}
    winner = max(eligible, key=lambda mode: gains[mode]["order_gain"])
    return winner, gains


def screen_tasks(seed, mode, order_kind):
    core = make_core(seed)
    token = hidden(core)
    out = {}
    train_x, train_y, first, first_y, second, second_y = pack_order(96102, 96104, 128)
    scored = score_order(core, seed, mode, order_kind, 96102, 96104, 128, "decay025")
    out["order"] = scored
    builders = {
        "sim_xor": lambda s, n: sim_xor(s, n),
        "separated_xor": lambda s, n: separated_xor(s, n),
        "delay": lambda s, n: delay_set(s, n, 6 if s == 96102 else 11),
    }
    for name, builder in builders.items():
        train_x, train_y = builder(96102, 256)
        test_x, test_y = builder(96104, 128)
        out[name] = score_table(core, seed, mode, order_kind, train_x, train_y, test_x, test_y, "decay025")
    clean_x, clean_y = order_with_distractors(96102, 256, 0)
    probe_x, probe_y = order_with_distractors(96104, 128, 0)
    probe_x = probe_x.copy()
    probe_x[:, 4, 4] += 1.5
    out["distractor"] = score_table(core, seed, mode, order_kind, clean_x, clean_y, probe_x, probe_y, "decay025")
    query_x, query_y, _first = query_set(96102, 256)
    query_test, query_test_y, _key = query_set(96104, 128)
    out["query"] = score_table(core, seed, mode, order_kind, query_x, query_y, query_test, query_test_y, "decay025")
    swapped_x, swapped_y = query_swap(query_test, query_test_y)
    trained = take(core, query_x, "decay025")
    head, _train = fit_arm(trained, query_y, seed, mode, order_kind)
    swap_eval = evaluate(head, take(core, swapped_x, "decay025"), swapped_y)
    out["query_swap"] = swap_eval
    if hidden(core) != token:
        raise RuntimeError("Hidden weights changed")
    return out


def gate(rows, mode):
    tasks = ("order", "sim_xor", "separated_xor", "delay", "distractor", "query")
    summary = {}
    qualified = []
    regressions = []
    for task in tasks:
        if task == "order":
            base = [row["norm"]["order"]["paired_flip"] for row in rows]
            cand = [row[mode]["order"]["paired_flip"] for row in rows]
            ridge = [row["norm"]["order"]["ridge"] for row in rows]
        else:
            base = [row["norm"][task]["accuracy"] for row in rows]
            cand = [row[mode][task]["accuracy"] for row in rows]
            ridge = [row["norm"][task]["ridge"] for row in rows]
        gain = interval([a - b for a, b in zip(cand, base)])
        summary[task] = {"baseline": float(np.mean(base)), "candidate": float(np.mean(cand)), "ridge": float(np.mean(ridge)), "gain": gain}
        if float(np.mean(ridge)) >= 0.80 and gain["mean"] >= 0.10 and gain["lower95"] > 0:
            qualified.append(task)
        if float(np.mean(cand)) < float(np.mean(base)) - 0.03:
            regressions.append(task)
    return summary, qualified, regressions


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    process = psutil.Process()
    peak = process.memory_info().rss
    clock = time.perf_counter()
    reproduced = reproduce()
    peak = max(peak, process.memory_info().rss)
    payload = {"reproduction": reproduced, "confirmation_opened": False}
    status = "STOP"
    if reproduced["passed"] and time.perf_counter() - clock < 540:
        core = make_core(94011)
        payload["diagnosis"] = diagnose(core)
        payload["ablation"] = ablate()
        candidate, detail = choose(payload["ablation"], reproduced)
        payload["candidate"] = candidate
        payload["selection"] = detail
        peak = max(peak, process.memory_info().rss)
        if candidate is not None and time.perf_counter() - clock < 540:
            rows = []
            perm_rows = []
            for seed in SCREEN_SEEDS:
                item = {"seed": seed, "norm": screen_tasks(seed, "norm", "original"), candidate: screen_tasks(seed, candidate, "original")}
                rows.append(item)
                perm_rows.append({"seed": seed, "norm": screen_tasks(seed, "norm", "perm"), candidate: screen_tasks(seed, candidate, "perm")})
                peak = max(peak, process.memory_info().rss)
            original, qualified, regressions = gate(rows, candidate)
            permuted, qualified_perm, regressions_perm = gate(perm_rows, candidate)
            passed = (
                len(qualified) >= 2
                and "order" in qualified
                and not regressions
                and len(qualified_perm) >= 2
                and not regressions_perm
            )
            payload["screen"] = {"original": original, "permuted": permuted, "qualified": qualified, "qualified_perm": qualified_perm, "regressions": regressions, "rows": rows, "perm_rows": perm_rows}
            status = "GO" if passed else "STOP"
        elif candidate is None:
            status = "STOP"
        else:
            status = "BLOCKED"
    elif not reproduced["passed"]:
        payload["stopped_after"] = "reproduction"
    else:
        status = "BLOCKED"
    info = process.memory_info()
    payload.update({
        "status": status,
        "seconds": time.perf_counter() - clock,
        "peak_rss_bytes_sampled": int(peak),
        "peak_wset_bytes": int(getattr(info, "peak_wset", 0)),
    })
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "passed": reproduced["passed"], "candidate": payload.get("candidate"), "seconds": payload["seconds"], "peak_rss_bytes_sampled": payload["peak_rss_bytes_sampled"]}, indent=2))


if __name__ == "__main__":
    main()
