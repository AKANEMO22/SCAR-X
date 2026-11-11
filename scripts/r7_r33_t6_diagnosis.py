"""R7 round 33 — why does T=6 fail? Test the REPLAY-RATIO hypothesis.

THE SUSPICION ROUND 32 DID NOT TEST
-----------------------------------
The replay buffer keeps BUF=100 sequences per seen task and never forgets them. So the
old:new ratio in each interleaved batch changes with task count:

    T=3, at the last task : 200 replayed vs 320 fresh   -> 0.63 : 1
    T=6, at the last task : 500 replayed vs 320 fresh   -> 1.56 : 1

At T=6 the head is pulled harder toward old tasks than at T=3, while each task's own share
of the parameters shrank. So "T=6 fails" may be a REPLAY-RATIO artefact, not a capacity or
scale limit.

COMPETING EXPLANATIONS
    H-ratio     the failure is the changing old:new mix. Fix: keep the ratio constant by
                shrinking BUF as tasks accumulate. Predicts: T=6 recovers to near T=3 levels.
    H-capacity  the head genuinely cannot hold six distributions. Predicts: T=6 stays bad
                whatever the buffer does.
    H-budget    six tasks at 320 examples / 1200 epochs each is too little per task.
                Round 32b ruled this out at T=3 (cut budget was slightly BETTER); it may
                still matter at T=6. Predicts: raising epochs at T=6 helps somewhat but
                not to gate level.

DESIGN  (T=6, 12 fresh seeds, the round-32 best order)
    buf100      round 32's setting, total buffer grows to 500   (ratio 1.56)
    buf_ratio   BUF scaled so total stays ~= 1/3 of fresh set: BUF = 320//(6*3) -> total ~106
    epochs_x2   buf100 with 2400 epochs per task (double)
    both        BUF scaled AND epochs doubled

Reported as the discriminating test it is: whichever arm recovers, that hypothesis wins.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scarx.evaluation import r7_secs as S  # noqa: E402
import r7_r32_tier3_six as R32  # noqa: E402

SEEDS = R32.SEEDS
TASKS = R32.TASKS
ORDER = ("xnor_delay", "maj3", "and3", "order_short", "parity3", "sim_xor")  # r32 best
N_TR, N_TE = 320, 200
TAIL = 250

ARMS = [
    ("buf100_ep600", 100, 600),      # round 32 setting
    ("buf_ratio_ep600", 17, 600),    # total ~102 over 6 tasks, ratio ~0.3
    ("buf100_ep1200", 100, 1200),    # double epochs
    ("buf_ratio_ep1200", 17, 1200),  # both
]


def run(seed, order, Ftr, ytr, Fte, yte, buf, epochs):
    head = R32.MLP(Ftr[order[0]].shape[1], seed, 32)
    n = len(order)
    Rm = np.full((n, n), np.nan)
    seq_idx, seq_lab, seq_tk = [], [], []
    rng = np.random.default_rng(seed + 909)

    for i_t, tk in enumerate(order):
        A_c = head.z(Ftr[tk]); y_c = ytr[tk]
        k = min(buf, N_TR)
        idx = np.random.default_rng(seed + i_t).choice(N_TR, k, replace=False)
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
    for _ in range(TAIL):
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
            "acq": float(np.mean(diag)),
            "final": float(np.mean(final)),
            "ratio_last": float(len(seq_idx) / N_TR)}


def main():
    t0 = time.time()
    print("=" * 98)
    print("R7 ROUND 33 — is the T=6 failure a REPLAY-RATIO artefact?")
    print("=" * 98)
    print(f"  T=6, 12 fresh seeds, order {' > '.join(o[:6] for o in ORDER)}\n")

    results = {}
    for label, buf, epochs in ARMS:
        mf, wf, aq, ok, rr = [], [], [], 0, []
        for seed in SEEDS:
            core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
            Ftr, ytr, Fte, yte = {}, {}, {}, {}
            for tk in TASKS:
                trx, try_ = R32.GENS[tk](seed + 200_000, N_TR)
                tex, tey = R32.GENS[tk](seed + 300_000, N_TE)
                Ftr[tk] = R32.feats(core, trx); ytr[tk] = try_
                Fte[tk] = R32.feats(core, tex); yte[tk] = tey
            r = run(seed, ORDER, Ftr, ytr, Fte, yte, buf, epochs)
            mf.append(r["meanF"]); wf.append(r["worstF"]); aq.append(r["acq"])
            rr.append(r["ratio_last"])
            ok += (r["meanF"] <= 3 and r["worstF"] <= 5 and r["acq"] >= .65)
        results[label] = {"meanF": float(np.mean(mf)),
                          "worstF": float(np.mean(wf)),
                          "acq": float(np.mean(aq)),
                          "seeds_ok": int(ok),
                          "replay_ratio": float(np.mean(rr))}
        s = results[label]
        print(f"  {label:18s} meanF {s['meanF']:6.2f}  worstF {s['worstF']:6.2f}  "
              f"acq {s['acq']:.4f}  ratio {s['replay_ratio']:.2f}  "
              f"{s['seeds_ok']}/12 pass", flush=True)

    print("\n" + "=" * 98)
    print("DISCRIMINATION")
    print("=" * 98)
    b100 = results["buf100_ep600"]
    brat = results["buf_ratio_ep600"]
    b2x = results["buf100_ep1200"]
    bboth = results["buf_ratio_ep1200"]
    print(f"  T=3 reference (round 31): meanF 0.42pp, 15/15 seeds")
    print(f"\n  H-ratio    : buf100 {b100['meanF']:.2f} -> buf_ratio {brat['meanF']:.2f}"
          f"  ({brat['meanF']-b100['meanF']:+.2f}pp)")
    print(f"  H-budget   : buf100 {b100['meanF']:.2f} -> epochs x2 {b2x['meanF']:.2f}"
          f"  ({b2x['meanF']-b100['meanF']:+.2f}pp)")
    print(f"  both       : {bboth['meanF']:.2f}pp  ({bboth['seeds_ok']}/12 pass)")

    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    best_of = min(results, key=lambda k: results[k]["meanF"])
    best = results[best_of]
    print(f"  best arm: {best_of}  meanF {best['meanF']:.2f}pp  "
          f"{best['seeds_ok']}/12 seeds pass")
    if best["seeds_ok"] >= 10:
        print(f"\n  => RECOVERED. The T=6 failure was mostly a "
              f"{'REPLAY-RATIO' if best_of.startswith('buf_ratio') else 'BUDGET'} artefact,")
        print(f"     not a scale limit. Round 32's conclusion must be revised.")
        verdict = "artefact_recovered"
    elif best["meanF"] < b100["meanF"] - 5.0:
        print(f"\n  => PARTIAL. Some recovery ({b100['meanF']:.1f} -> "
              f"{best['meanF']:.1f}pp) but nowhere near the gate. The artefact explains")
        print(f"     part of the T=6 failure; a genuine scale limit remains.")
        verdict = "artefact_partial"
    else:
        print(f"\n  => NO RECOVERY. Neither the replay ratio nor the budget explains it.")
        print(f"     The T=6 failure is a genuine scale/capacity limit, and round 32's")
        print(f"     conclusion stands.")
        verdict = "genuine_scale_limit"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t6-diagnosis.json").write_text(json.dumps({
        "tasks": list(TASKS), "order": list(ORDER), "seeds": list(SEEDS),
        "arms": {k: v for k, v in results.items()},
        "best_arm": best_of, "verdict": verdict,
        "t3_reference": {"meanF": 0.42, "seeds_ok": 15, "n": 15},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: t6-diagnosis.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
