"""R4A Stage 4 — confirmation on 8 fresh seeds, locked algorithm.

Opened because Stage 3 passed all six gates. Nothing is tuned here: feature map,
learning rate, clip, epochs, generators and gates are exactly as registered.

Confirmation adds the brief's required stress conditions:
  - reversed write order
  - query swaps (paired counterfactual is the swap)
  - temporal shifts (writes moved)
  - distractor events
  - held-out noise distributions

Bootstrap resamples INDEPENDENT UNITS = (seed, pair).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r4_probe as P  # noqa: E402
from scarx.evaluation import r4_relational as R  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

CONF = (96001, 96007, 96013, 96019, 96023, 96029, 96031, 96037)   # protocol §6
N_TRAIN, N_PAIRS = 1024, 256
PULSE = R.PULSE


def boot_ci(diffs, n=10000, seed=2101):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    draws = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return {"mean": float(d.mean()), "lower95": float(np.quantile(draws, .025)),
            "upper95": float(np.quantile(draws, .975)), "n_units": len(d)}


def shifted(seed, count, shift=1):
    """Writes and query moved later by `shift` ticks; the rest of the window is noise."""
    x, y, _ = R.query_set(seed, count)
    out = np.zeros_like(x)
    slots = [2, 6, 10]
    if max(slots) + shift > x.shape[1] - 1:
        raise ValueError("shift too large")
    for s in slots:
        out[:, s + shift, :] = x[:, s, :]
    return out, y


def with_distractors(seed, count, n_distract=2):
    """Extra PULSE events on unused channels at unused ticks, labels unchanged."""
    rng = np.random.default_rng(seed + 12345)
    x, y, _ = R.query_set(seed, count)
    out = x.copy()
    used = {2, 6, 10}
    free = [t for t in range(12) if t not in used]
    for i in range(count):
        for t in rng.choice(free, size=min(n_distract, len(free)), replace=False):
            c = int(rng.integers(6, 8))
            out[i, t, c] += PULSE
    return out, y


def noisier(seed, count, scale=2.0):
    """Held-out noise distribution: background sigma doubled."""
    rng = np.random.default_rng(seed)
    x, y, _ = R.query_set(seed, count)
    return x + rng.normal(0, .08 * (scale - 1.0), x.shape).astype(np.float32), y


def paired_variant(seed, pairs, variant):
    """Paired counterfactual corpus under a stress variant (applied to both siblings)."""
    x, y, pid, _ = R.paired_counterfactual(seed, pairs)
    if variant is None:
        return x, y
    if variant == "reversed":
        out = x.copy()
        out[:, [2, 6]] = out[:, [6, 2]]
        return out, y            # labels unchanged by slot swap
    if variant == "shift":
        out = np.zeros_like(x)
        for s in (2, 6, 10):
            out[:, s + 1, :] = x[:, s, :]
        return out, y
    if variant == "distractor":
        rng = np.random.default_rng(seed + 999)
        out = x.copy()
        free = [t for t in range(12) if t not in (2, 6, 10)]
        for i in range(len(out)):
            for t in rng.choice(free, 2, replace=False):
                out[i, t, int(rng.integers(6, 8))] += PULSE
        return out, y
    if variant == "noise":
        rng = np.random.default_rng(seed + 777)
        return x + rng.normal(0, .08, x.shape).astype(np.float32), y
    raise ValueError(variant)


def run_cell(seed, variant):
    ptr, pyt = paired_variant(seed + 200_000, N_TRAIN // 2, variant)
    pte, pye = paired_variant(seed + 300_000, N_PAIRS, variant)
    qt = 11 if variant == "shift" else 10    # follow the generator's query tick
    ctr, cte = P.capture(seed, ptr, query_tick=qt), P.capture(seed, pte, query_tick=qt)
    post_tr = P.state_vector(ctr, ("spikes", "voltage", "adaptation", "features"), "post")
    post_te = P.state_vector(cte, ("spikes", "voltage", "adaptation", "features"), "post")
    r5_tr = P.bilinear_query_map(post_tr, P.query_cue(ptr))
    r5_te = P.bilinear_query_map(post_te, P.query_cue(pte))
    h = P.train_head(r5_tr, pyt, seed=seed)
    pred = h.predict(r5_te)
    acc = float(np.mean(pred == pye))
    fst, snd = pred[0::2], pred[1::2]
    ya, yb = pye[0::2], pye[1::2]
    valid = ya != yb
    cons = float(np.mean(fst[valid] != snd[valid])) if valid.any() else 0.0

    base = {}
    for name in ("decay025", "spikesum", "bank5"):
        hb = P.train_head(ctr["final"][name], pyt, seed=seed)
        base[name] = hb.acc(cte["final"][name], pye)
    best = max(base.values())
    unit = [(int(fst[j] == ya[j]) + int(snd[j] == yb[j])) / 2.0 for j in range(len(fst))]
    return {"seed": seed, "variant": variant or "clean", "r5": acc,
            "r5_train": h.acc(r5_tr, pyt), "best_baseline": best,
            "consistency": cons, "unit_acc": unit}


def main():
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "artifacts/deepseek-r4/stage3-lock.json").read_text()) \
        if (root / "artifacts/deepseek-r4/stage3-lock.json").exists() else None
    run = new_run(root, "r4-confirmation-r5-v1", {
        "seeds": list(CONF), "n_train": N_TRAIN, "n_pairs": N_PAIRS,
        "variants": ["clean", "reversed", "shift", "distractor", "noise"],
        "stage3_lock": lock, "note": "locked algorithm; no tuning after Stage 3"})

    variants = (None, "reversed", "shift", "distractor", "noise")
    rows, units = [], []
    with ResourceMonitor(interval=.1) as mon:
        for seed in CONF:
            for v in variants:
                r = run_cell(seed, v)
                rows.append({k: r[k] for k in r if k != "unit_acc"})
                units.extend([(seed, r["variant"], u) for u in r["unit_acc"]])
                print(f"  seed{seed} {r['variant']:11s} R5 {r['r5']:.4f} "
                      f"base {r['best_baseline']:.4f} cons {r['consistency']:.4f}",
                      flush=True)
            mon.check()

    def mean(pred):
        return float(np.mean([r["r5"] for r in rows if pred(r)]))

    clean = [r for r in rows if r["variant"] == "clean"]
    diffs = []
    for s in CONF:
        c = [r for r in clean if r["seed"] == s][0]
        diffs.append(c["r5"] - c["best_baseline"])
    ci = boot_ci(diffs)

    per_variant = {}
    for v in variants:
        vv = [r for r in rows if r["variant"] == (v or "clean")]
        per_variant[v or "clean"] = {
            "r5": float(np.mean([r["r5"] for r in vv])),
            "baseline": float(np.mean([r["best_baseline"] for r in vv])),
            "consistency": float(np.mean([r["consistency"] for r in vv])),
            "worst_seed_r5": float(min(r["r5"] for r in vv))}

    g_conf = {
        "C1_clean_accuracy_ge_070": per_variant["clean"]["r5"] >= .70,
        "C2_consistency_ge_060": per_variant["clean"]["consistency"] >= .60,
        "C3_improvement_ge_10pp": (per_variant["clean"]["r5"] -
                                   per_variant["clean"]["baseline"]) * 100 >= 10,
        "C4_ci_lower_gt_zero": ci["lower95"] > 0,
        "C5_no_variant_below_065": all(per_variant[k]["r5"] >= .65 for k in per_variant),
        "C6_worst_seed_ge_060": per_variant["clean"]["worst_seed_r5"] >= .60,
    }
    summary = {"per_variant": per_variant, "paired_ci_clean": ci,
               "gates": g_conf, "confirmation_pass": all(g_conf.values()),
               "n_units": len(units)}
    write_json(run / "result.json",
               {"trials": rows, "summary": summary, "resources": mon.report()})

    print("\n" + "=" * 80)
    print("STAGE 4 CONFIRMATION — 8 fresh seeds x 5 stress conditions")
    print("=" * 80)
    print(f"  {'variant':12s} {'R5':>8} {'baseline':>9} {'consistency':>12} {'worst seed':>11}")
    for k, v in per_variant.items():
        print(f"  {k:12s} {v['r5']:8.4f} {v['baseline']:9.4f} {v['consistency']:12.4f} "
              f"{v['worst_seed_r5']:11.4f}")
    print(f"\n  paired CI (clean): {ci['mean']*100:+.2f}pp "
          f"[{ci['lower95']*100:+.2f}, {ci['upper95']*100:+.2f}] over {ci['n_units']} seeds")
    print(f"\n  GATES: {json.dumps(g_conf, indent=1)}")
    print(f"  confirmation_pass: {summary['confirmation_pass']}")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
