"""Reproduce R5B case A, map the errors, then screen the unclipped head if the rule allows."""
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

from scarx.evaluation import r5b_independent as gen
from scarx.evaluation import r5b_probe as probe
from scarx.evaluation.phase21 import interval
from scarx.evaluation.r6_candidate import NoClip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "grok-r6-causal-v1"
CONF = (98001, 98007, 98013, 98019, 98023, 98029, 98031, 98037)
SCREEN = (120011, 120017, 120023)
PROJECTIONS = (1234, 7, 99)


def features(seed, sequences, tick, projection=1234):
    captured = probe.capture(seed, sequences, query_tick=tick)
    state = probe.state_vector(captured, "post")
    return probe.bilinear_query_map(state, gen.query_cue(sequences, tick=tick), seed=projection)


def pair_stats(predicted, labels, flips):
    left, right = predicted[0::2], predicted[1::2]
    ya, yb = labels[0::2], labels[1::2]
    both = (left == ya) & (right == yb)
    differ = np.asarray(flips, bool)
    return {
        "direct": float(np.mean(predicted == labels)),
        "both_A": float(np.mean(both[differ])) if np.any(differ) else None,
        "both_B": float(np.mean(both[~differ])) if np.any(~differ) else None,
        "correct_A_when_differ": float(np.mean(left[differ] == ya[differ])) if np.any(differ) else None,
        "correct_B_when_differ": float(np.mean(right[differ] == yb[differ])) if np.any(differ) else None,
        "sibling_error_correlation": float(np.mean((left[differ] != ya[differ]) == (right[differ] != yb[differ]))) if np.any(differ) else None,
    }


def fit_predict(cls, train_f, train_y, test_f, seed):
    head = cls(train_f.shape[1], seed)
    for row in train_f:
        head.observe(row)
    for row, label in zip(train_f, train_y):
        head.learn(row, int(label))
    return head.predict(test_f), head.acc(train_f, train_y)


def one_seed(seed, cls=probe.Head, projection=1234):
    train_x, train_y, _va, _vb, _q, _pid = gen.independent_query_set_v1(seed + 200_000, 1024)
    _x, _y, _a, _b, _qq, _pp = gen.independent_query_set_v1(seed + 300_000, 512)
    pair_x, pair_y, _pva, _pvb, _pq, _pid, flips = gen.independent_paired_set_v1(seed + 400_000, 256)
    before = probe.weights(seed)[2].tobytes()
    tick = gen.locate_query_tick(train_x[0])
    train_f = features(seed, train_x, tick, projection)
    pair_f = features(seed, pair_x, gen.locate_query_tick(pair_x[0]), projection)
    predicted, train_acc = fit_predict(cls, train_f, train_y, pair_f, seed)
    stats = pair_stats(predicted, pair_y, flips)
    stats["train_accuracy"] = train_acc
    ridge_hat = probe.ridge_predict(train_f, train_y, pair_f)
    stats["ridge"] = pair_stats(ridge_hat, pair_y, flips)
    stats["query_tick"] = tick
    stats["oracle_matches_labels"] = bool(np.array_equal(gen.oracle_all(pair_x), pair_y))
    stats["hidden_unchanged"] = probe.weights(seed)[2].tobytes() == before
    stats["siblings_share_prefix"] = bool(np.max(np.abs(pair_x[0::2, :10] - pair_x[1::2, :10])) == 0.0)
    return stats


def reproduce():
    rows = [dict(seed=seed, **one_seed(seed)) for seed in CONF]
    return rows


def projections(seed=98001):
    rows = []
    for value in PROJECTIONS:
        stats = one_seed(seed, projection=value)
        rows.append({"projection": value, "both_A": stats["both_A"], "direct_on_pairs": stats["direct_on_pairs"]})
    return rows


def screen_seed(seed, cls, order_kind="original"):
    train_x, train_y, _va, _vb, q, _pid = gen.independent_query_set_v1(seed + 200_000, 1024)
    pair_x, pair_y, _a, _b, _q, _pid, flips = gen.independent_paired_set_v1(seed + 400_000, 256)
    order = np.arange(len(train_y))
    if order_kind == "perm":
        order = np.random.default_rng(96001).permutation(len(train_y))
    tick = gen.locate_query_tick(train_x[0])
    train_f = features(seed, train_x, tick)[order]
    train_y = train_y[order]
    pair_f = features(seed, pair_x, gen.locate_query_tick(pair_x[0]))
    predicted, train_acc = fit_predict(cls, train_f, train_y, pair_f, seed)
    stats = pair_stats(predicted, pair_y, flips)
    stats["train_accuracy"] = train_acc
    stats["direct_iid"] = float(np.mean(predicted[:0] == pair_y[:0])) if False else None
    held_x, held_y, *_ = gen.independent_query_set_v1(seed + 300_000, 512)
    held_f = features(seed, held_x, gen.locate_query_tick(held_x[0]))
    # Refit is wasteful; predict held from the same head by repeating fit. The head was not returned.
    predicted_held, _train = fit_predict(cls, train_f, train_y, held_f, seed)
    stats["direct"] = float(np.mean(predicted_held == held_y))
    stats["finite"] = True
    return stats


def protected(seed, cls):
    from scarx.evaluation.readout_mechanism import fit_arm, make_core, rollout, sim_xor, order_pairs, paired_flip, evaluate
    core = make_core(seed)
    out = {}
    for name, train, test in (
        ("xor", sim_xor(96102, 256), sim_xor(96104, 128)),
        ("order", None, None),
    ):
        if name == "xor":
            trained = rollout(core, train[0])["decay025"]
            held = rollout(core, test[0])["decay025"]
            mode = "noclip" if cls is NoClip else "norm"
            head, _stats = fit_arm(trained, train[1], seed, mode, "original")
            out[name] = evaluate(head, held, test[1])["accuracy"]
        else:
            packed = order_pairs(96102, 128)
            train_x = np.concatenate((packed[0], packed[2]))
            train_y = np.concatenate((packed[1], packed[3]))
            test = order_pairs(96104, 128)
            trained = rollout(core, train_x)["decay025"]
            left = rollout(core, test[0])["decay025"]
            right = rollout(core, test[2])["decay025"]
            mode = "noclip" if cls is NoClip else "norm"
            head, _stats = fit_arm(trained, train_y, seed, mode, "original")
            out[name] = paired_flip(head, left, test[1], right, test[3])
    return out


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    process = psutil.Process()
    peak = process.memory_info().rss
    clock = time.perf_counter()
    rows = []
    for seed in CONF:
        rows.append({"seed": seed, **one_seed(seed)})
        peak = max(peak, process.memory_info().rss)
    head_both = [row["both_A"] for row in rows]
    ridge_both = [row["ridge"]["both_A"] for row in rows]
    candidate = "noclip" if float(np.mean(ridge_both)) >= 0.75 and float(np.mean(head_both)) < 0.70 else None
    payload = {
        "reproduction_class": "REPRODUCTION",
        "reproduction": rows,
        "head_both_A_mean": float(np.mean(head_both)),
        "ridge_both_A_mean": float(np.mean(ridge_both)),
        "query_gap": float(np.mean([row["correct_A_when_differ"] - row["correct_B_when_differ"] for row in rows])),
        "projection_distribution_seed_98001": projections(),
        "candidate": candidate,
        "confirmation_opened": False,
    }
    status = "STOP"
    if candidate is not None and time.perf_counter() - clock < 540:
        fresh = [{"seed": seed, "norm": screen_seed(seed, probe.Head), "noclip": screen_seed(seed, NoClip)} for seed in SCREEN]
        perm = [{"seed": seed, "norm": screen_seed(seed, probe.Head, "perm"), "noclip": screen_seed(seed, NoClip, "perm")} for seed in SCREEN]
        gain = interval([row["noclip"]["both_A"] - row["norm"]["both_A"] for row in fresh])
        gain_perm = interval([row["noclip"]["both_A"] - row["norm"]["both_A"] for row in perm])
        protected_rows = [{"seed": seed, "norm": protected(seed, probe.Head), "noclip": protected(seed, NoClip)} for seed in SCREEN]
        xor_drop = float(np.mean([row["norm"]["xor"] - row["noclip"]["xor"] for row in protected_rows]))
        order_drop = float(np.mean([row["norm"]["order"] - row["noclip"]["order"] for row in protected_rows]))
        direct = float(np.mean([row["noclip"]["direct"] for row in fresh]))
        both_b = float(np.mean([row["noclip"]["both_B"] for row in fresh]))
        passed = (
            float(np.mean([row["noclip"]["both_A"] for row in fresh])) >= 0.75
            and gain["mean"] >= 0.05 and gain["lower95"] > 0
            and direct >= 0.85 and both_b >= 0.70
            and xor_drop <= 0.03 and order_drop <= 0.03
            and gain_perm["mean"] >= 0.05 and gain_perm["lower95"] > 0
        )
        payload.update({"screen": fresh, "screen_perm": perm, "protected": protected_rows,
                        "gain": gain, "perm_gain": gain_perm, "xor_drop": xor_drop, "order_drop": order_drop})
        status = "GO" if passed else "STOP"
        peak = max(peak, process.memory_info().rss)
    payload.update({
        "status": status,
        "seconds": time.perf_counter() - clock,
        "peak_rss_bytes_sampled": int(max(peak, process.memory_info().rss)),
        "peak_wset_bytes": int(getattr(process.memory_info(), "peak_wset", 0)),
    })
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "head_both_A": payload["head_both_A_mean"],
        "ridge_both_A": payload["ridge_both_A_mean"],
        "query_gap": payload["query_gap"],
        "candidate": candidate,
        "seconds": payload["seconds"],
    }, indent=2))


def screen_seed(seed, cls, order_kind="original"):
    train_x, train_y, _va, _vb, _q, _pid = gen.independent_query_set_v1(seed + 200_000, 1024)
    pair_x, pair_y, _a, _b, _q, _pid, flips = gen.independent_paired_set_v1(seed + 400_000, 256)
    order = np.arange(len(train_y)) if order_kind == "original" else np.random.default_rng(96001).permutation(len(train_y))
    tick = gen.locate_query_tick(train_x[0])
    train_f = features_of(seed, train_x, tick)[order]
    predicted, train_acc = fit_predict(cls, train_f, train_y[order], features_of(seed, pair_x, gen.locate_query_tick(pair_x[0])), seed)
    stats = pair_stats(predicted, pair_y, flips)
    held_x, held_y, *_rest = gen.independent_query_set_v1(seed + 300_000, 512)
    held_hat, _ignored = fit_predict(cls, train_f, train_y[order], features_of(seed, held_x, gen.locate_query_tick(held_x[0])), seed)
    stats["direct"] = float(np.mean(held_hat == held_y))
    stats["train_accuracy"] = train_acc
    return stats


def features_of(seed, sequences, tick):
    return features(seed, sequences, tick)


def pair_stats(predicted, labels, flips):
    return pair_stats_impl(predicted, labels, flips)


def pair_stats_impl(predicted, labels, flips):
    left, right = predicted[0::2], predicted[1::2]
    ya, yb = labels[0::2], labels[1::2]
    both = (left == ya) & (right == yb)
    differ = np.asarray(flips, bool)
    return {
        "direct_on_pairs": float(np.mean(predicted == labels)),
        "both_A": float(np.mean(both[differ])),
        "both_B": float(np.mean(both[~differ])),
        "correct_A_when_differ": float(np.mean(left[differ] == ya[differ])),
        "correct_B_when_differ": float(np.mean(right[differ] == yb[differ])),
    }


def fit_predict(cls, train_f, train_y, test_f, seed):
    head = cls(train_f.shape[1], seed)
    for row in train_f:
        head.observe(row)
    for row, label in zip(train_f, train_y):
        head.learn(row, int(label))
    return head.predict(test_f), head.acc(train_f, train_y)


if __name__ == "__main__":
    main()
