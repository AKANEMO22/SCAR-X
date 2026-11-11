"""R7 round 32b — the confound check round 32 needed: T=6 or reduced budget?

ROUND 32 RESULT
    T=3 (round 31): meanF 0.42pp, 15/15 seeds pass
    T=6 (round 32): meanF 27.57pp, 0/12 seeds pass
A 65x increase. That looks like a scale failure — but I changed THREE things at once:

    task count     3 -> 6
    samples/task   384 -> 320
    epochs/task    1600 -> 1200
    buffer        128 -> 100

Any of the last three could be responsible. If the reduced budget alone breaks T=3, then
round 32 did not test scale at all — it tested a budget cut, and I must not report it as a
generalisation failure.

DECISIVE TEST (cheap, 4 cells x 12 seeds)
    T=3 with the T=6 budget (320 samples, 1200 epochs, buffer 100)
        -> if this fails, BUDGET is the cause and round 32 is not a T=6 result
        -> if this passes, T=6 is the cause and round 32 stands

Reported either way.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scarx.evaluation import r7_secs as S  # noqa: E402
import r7_r32_tier3_six as R32  # noqa: E402  (reuse tasks, MLP, feats, softmax)

SEEDS = R32.SEEDS
ORDER3 = ("sim_xor", "order_short", "parity3")     # round 31's best
T3 = ("sim_xor", "parity3", "order_short")


def run_with(head_cls_seed, order, Ftr, ytr, Fte, yte, n_tr, epochs, buf, tail):
    head = R32.MLP(Ftr[order[0]].shape[1], head_cls_seed, 32)
    n = len(order)
    Rm = np.full((n, n), np.nan)
    seq_idx, seq_lab, seq_tk = [], [], []
    rng = np.random.default_rng(head_cls_seed + 909)

    for i_t, tk in enumerate(order):
        A_c = head.z(Ftr[tk]); y_c = ytr[tk]
        k = min(buf, n_tr)
        idx = np.random.default_rng(head_cls_seed + i_t).choice(n_tr, k, replace=False)
        seq_idx.extend(int(j) for j in idx)
        seq_lab.extend(int(ytr[tk][j]) for j in idx)
        seq_tk.extend([tk] * len(idx))
        A_r = np.stack([head.z(Ftr[seq_tk[q]])[seq_idx[q]]
                        for q in range(len(seq_idx))])
        y_r = np.array(seq_lab, np.int32)
        hc = max(1, len(y_c) // 2); hr = max(1, len(y_r) // 2)
        for _ in range(2 * epochs):
            ic = rng.choice(len(y_c), hc, replace=False)
            ir = rng.choice(len(y_r), hr, replace=False)
            head.step(np.concatenate([A_c[ic], A_r[ir]]),
                      np.concatenate([y_c[ic], y_r[ir]]))
        for j_t, tk2 in enumerate(order):
            Rm[i_t, j_t] = head.acc(head.z(Fte[tk2]), yte[tk2])

    nb = min(buf, len(seq_idx))
    A1 = head.z(Ftr[order[0]])[seq_idx[:nb]]
    y1 = np.array(seq_lab[:nb], np.int32)
    Al = head.z(Ftr[order[-1]]); yl = ytr[order[-1]]
    h1 = max(1, len(y1) // 2); hl = max(1, len(yl) // 2)
    for _ in range(tail):
        i1 = rng.choice(len(y1), h1, replace=False)
        il = rng.choice(len(yl), hl, replace=False)
        head.step(np.concatenate([A1[i1], Al[il]]),
                  np.concatenate([y1[i1], yl[il]]))
    for j_t, tk2 in enumerate(order):
        Rm[n - 1, j_t] = head.acc(head.z(Fte[tk2]), yte[tk2])

    diag = np.diag(Rm); final = Rm[n - 1]; delta = diag - final
    meas = delta[:n - 1]
    return {"meanF": float(np.mean(np.maximum(meas, 0)) * 100),
            "worstF": float(np.max(delta) * 100),
            "acq": float(np.mean(diag))}


CONFIGS = [
    ("T3_budget384_1600_buf128", T3, 384, 800, 128, 300),   # round 31 setting
    ("T3_budget320_1200_buf100", T3, 320, 600, 100, 250),   # round 32 setting
]


def main():
    t0 = time.time()
    print("=" * 98)
    print("R7 ROUND 32b — is the T=6 failure about SCALE or about my BUDGET CUT?")
    print("=" * 98)
    print(f"  both configs run T=3 with order {' > '.join(ORDER3)}, 12 fresh seeds\n")

    out_rows = {}
    for label, tasks, n_tr, epochs, buf, tail in CONFIGS:
        mf, wf, aq, ok = [], [], [], 0
        for seed in SEEDS:
            core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
            Ftr, ytr, Fte, yte = {}, {}, {}, {}
            for tk in tasks:
                trx, try_ = R32.GENS[tk](seed + 200_000, n_tr)
                tex, tey = R32.GENS[tk](seed + 300_000, 200)
                Ftr[tk] = R32.feats(core, trx); ytr[tk] = try_
                Fte[tk] = R32.feats(core, tex); yte[tk] = tey
            r = run_with(seed, ORDER3, Ftr, ytr, Fte, yte, n_tr, epochs, buf, tail)
            mf.append(r["meanF"]); wf.append(r["worstF"]); aq.append(r["acq"])
            ok += (r["meanF"] <= 3 and r["worstF"] <= 5 and r["acq"] >= .65)
        out_rows[label] = {"meanF": float(np.mean(mf)), "worstF": float(np.mean(wf)),
                           "acq": float(np.mean(aq)), "seeds_ok": int(ok)}
        s = out_rows[label]
        print(f"  {label:28s} meanF {s['meanF']:6.2f}  worstF {s['worstF']:6.2f}  "
              f"acq {s['acq']:.4f}  {s['seeds_ok']}/12 pass", flush=True)

    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    a = out_rows["T3_budget384_1600_buf128"]
    b = out_rows["T3_budget320_1200_buf100"]
    print(f"  T=3 with full budget : {a['meanF']:6.2f}pp  ({a['seeds_ok']}/12)")
    print(f"  T=3 with cut budget  : {b['meanF']:6.2f}pp  ({b['seeds_ok']}/12)")
    print(f"  T=6 with cut budget  : 27.57pp  (0/12)  <- round 32")
    print(f"\n  budget effect at T=3 : {b['meanF'] - a['meanF']:+.2f}pp")
    print(f"  scale effect (T6-T3) : {27.57 - b['meanF']:+.2f}pp")

    if b["meanF"] >= 10.0:
        print(f"\n  => ROUND 32 IS INVALID AS A T=6 TEST. The reduced budget alone takes")
        print(f"     T=3 from {a['meanF']:.2f}pp to {b['meanF']:.2f}pp. The 27.57pp at T=6")
        print(f"     cannot be attributed to task count. Withdrawing the Tier-3 failure")
        print(f"     claim and re-running T=6 at the FULL budget.")
        verdict = "round32_invalid_budget_confound"
    elif b["meanF"] <= 3.0:
        print(f"\n  => ROUND 32 STANDS. T=3 survives the budget cut at {b['meanF']:.2f}pp,")
        print(f"     so the 27.57pp at T=6 is attributable to task count. Tier 3 fails.")
        verdict = "round32_stands_scale_failure"
    else:
        print(f"\n  => MIXED. Budget costs {b['meanF']-a['meanF']:.2f}pp at T=3 and scale")
        print(f"     costs the rest. Neither factor alone explains T=6.")
        verdict = "mixed"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out = out / "budget-confound.json"
    out.write_text(json.dumps({"configs": out_rows, "verdict": verdict,
                               "t6_reference": 27.57,
                               "seeds": list(SEEDS),
                               "order": list(ORDER3)}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\n  written: budget-confound.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
