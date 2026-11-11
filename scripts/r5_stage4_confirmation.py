"""Round 5B Stage 4 — confirmation on 8 fresh seeds, locked algorithm.

Opened because Stage 3 passed all nine gates. Nothing is tuned.

Temporal stress is separated into the two questions the protocol demands:
  * in-distribution shifted learning : train shift, test shift
  * OOD temporal transfer            : train clean, test shift (no retrain)

The noise change is its own condition, never bundled with a shift, so a timing
effect cannot be mistaken for a noise effect.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r5_probe as P  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

CONF = (98001, 98007, 98013, 98019, 98023, 98029, 98031, 98037)
N_TRAIN = 1024


def qt_of(x):
    """Locate the query tick for THIS data. Never assume tick 10."""
    return G.locate_query_tick(np.asarray(x)[0])


def feats(seed, x, tick):
    cap = P.capture(seed, x, query_tick=tick)
    return P.bilinear_query_map(P.state_vector(cap, "post"), G.query_cue(x, tick=tick))


def evaluate(seed, trx, try_, tex, tey, tex_p=None, pty=None, flips=None):
    q_tr, q_te = qt_of(trx), qt_of(tex)
    f_tr = feats(seed, trx, q_tr)
    f_te = feats(seed, tex, q_te)
    h = P.train_head(f_tr, try_, seed=seed)
    out = {"direct": h.acc(f_te, tey), "train": h.acc(f_tr, try_),
           "query_tick_train": q_tr, "query_tick_test": q_te}
    if tex_p is not None:
        q_p = qt_of(tex_p)
        pr = h.predict(feats(seed, tex_p, q_p))
        a, b = pr[0::2], pr[1::2]
        ya, yb = pty[0::2], pty[1::2]
        fA = flips
        out["both_A"] = float(np.mean((a[fA] == ya[fA]) & (b[fA] == yb[fA])))
        out["both_B"] = float(np.mean((a[~fA] == ya[~fA]) & (b[~fA] == yb[~fA])))
        out["both_all"] = float(np.mean((a == ya) & (b == yb)))
        out["paired"] = out["both_all"]
    # strongest fair baseline on the same rows
    out["firstkey"] = float(np.mean(G.baseline_first_key(tex) == tey))
    return out


def boot_ci(d, n=10000, seed=2101):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    draws = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return {"mean": float(d.mean()), "lower95": float(np.quantile(draws, .025)),
            "upper95": float(np.quantile(draws, .975)), "n_units": len(d)}


def main():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r5/protocol-v1.md"
    psha = __import__("hashlib").sha256(proto.read_bytes()).hexdigest()
    run = new_run(root, "r5-confirmation-v1", {
        "protocol_sha256": psha, "seeds": list(CONF), "n_train": N_TRAIN,
        "conditions": ["clean", "train_shift_test_shift", "ood_clean_to_shift",
                       "reversed_writes", "distractors", "noise_change",
                       "query_tick_variation"]})

    rows = []
    with ResourceMonitor(interval=.1) as mon:
        for seed in CONF:
            trx, try_, *_ = G.independent_query_set_v1(seed + 200_000, N_TRAIN)
            tex, tey, *_ = G.independent_query_set_v1(seed + 300_000, 512)
            ptx, pty, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(seed + 400_000, 256)
            rec = {"seed": seed}

            # clean reference
            rec["clean"] = evaluate(seed, trx, try_, tex, tey, ptx, pty, flips)

            # in-distribution shifted learning: train AND test shifted
            s_tr, s_te = G.shifted(trx, 1, seed=seed), G.shifted(tex, 1, seed=seed)
            s_pt = G.shifted(ptx, 1, seed=seed)
            rec["train_shift_test_shift"] = evaluate(seed, s_tr, try_, s_te, tey,
                                                     s_pt, pty, flips)

            # OOD temporal transfer: trained clean, tested shifted, NO retrain
            q_tr, q_te = qt_of(trx), qt_of(G.shifted(tex, 1))
            f_tr = feats(seed, trx, q_tr)
            h = P.train_head(f_tr, try_, seed=seed)
            rec["ood_clean_to_shift"] = {
                "direct": h.acc(feats(seed, G.shifted(tex, 1, seed=seed), q_te), tey),
                "firstkey": float(np.mean(G.baseline_first_key(G.shifted(tex,1,seed=seed)) == tey))}

            # reversed write order
            r_tr, r_te = G.reversed_writes(trx), G.reversed_writes(tex)
            rec["reversed_writes"] = evaluate(seed, r_tr, try_, r_te, tey,
                                              G.reversed_writes(ptx), pty, flips)

            # distractors
            d_tr = G.with_distractors(trx, seed + 11)
            d_te = G.with_distractors(tex, seed + 12)
            rec["distractors"] = evaluate(seed, d_tr, try_, d_te, tey,
                                          G.with_distractors(ptx, seed + 13), pty, flips)

            # noise change ALONE, no shift
            n_tr = G.noisier(trx, seed + 21, sigma=.16)
            n_te = G.noisier(tex, seed + 22, sigma=.16)
            rec["noise_change"] = evaluate(seed, n_tr, try_, n_te, tey,
                                           G.noisier(ptx, seed + 23, sigma=.16), pty, flips)

            # query tick variation: writes stay, query moved LATER by 1 (tick 11)
            def move_query(x, delta=1):
                out = np.asarray(x, np.float32).copy()
                qmask = out[:, :, 4:6] > .5
                moved = np.zeros_like(out)
                for t in range(out.shape[1]):
                    nt = t + delta if np.any(qmask[:, t]) else t
                    if nt < out.shape[1]:
                        moved[:, nt] += out[:, t]
                return moved
            m_tr = move_query(trx); m_te = move_query(tex); m_pt = move_query(ptx)
            rec["query_tick_variation"] = evaluate(seed, m_tr, try_, m_te, tey,
                                                   m_pt, pty, flips)

            rows.append(rec)
            mon.check()
            print(f"  seed {seed} clean {rec['clean']['direct']:.4f} "
                  f"shift {rec['train_shift_test_shift']['direct']:.4f} "
                  f"ood {rec['ood_clean_to_shift']['direct']:.4f}", flush=True)

    conds = ("clean", "train_shift_test_shift", "ood_clean_to_shift",
             "reversed_writes", "distractors", "noise_change", "query_tick_variation")
    per = {}
    for c in conds:
        d = [r[c] for r in rows]
        per[c] = {"direct": float(np.mean([x["direct"] for x in d])),
                  "worst_seed": float(np.min([x["direct"] for x in d])),
                  "firstkey": float(np.mean([x.get("firstkey", np.nan) for x in d]))}
        if "both_all" in d[0]:
            per[c]["both_all"] = float(np.mean([x["both_all"] for x in d]))
            per[c]["both_A"] = float(np.mean([x["both_A"] for x in d]))
            per[c]["both_B"] = float(np.mean([x["both_B"] for x in d]))

    diffs = [r["clean"]["direct"] - 0.7522 for r in rows]
    ci = boot_ci(diffs)
    gates = {
        "C1_clean_ge_085": per["clean"]["direct"] >= .85,
        "C2_gain_ge_10pp": (per["clean"]["direct"] - .7522) * 100 >= 10,
        "C3_both_A_ge_070": per["clean"]["both_A"] >= .70,
        "C4_both_B_ge_070": per["clean"]["both_B"] >= .70,
        "C5_ci_lower_gt_0": ci["lower95"] > 0,
        "C6_worst_clean_seed_ge_075": per["clean"]["worst_seed"] >= .75,
        "C7_shifted_learning_ge_080": per["train_shift_test_shift"]["direct"] >= .80,
        "C8_all_conditions_above_ceiling": all(per[c]["direct"] > .7522 for c in conds),
    }
    summary = {"per_condition": per, "paired_ci_clean": ci, "gates": gates,
               "confirmation_pass": all(gates.values()),
               "ood_transfer_reported_separately": True}
    write_json(run / "result.json",
               {"trials": rows, "summary": summary, "resources": mon.report()})

    print("\n" + "=" * 90)
    print("STAGE 4 CONFIRMATION — 8 fresh seeds x 7 conditions")
    print("=" * 90)
    print(f"  {'condition':26s} {'direct':>8} {'worst':>8} {'firstkey':>9} "
          f"{'both_all':>9} {'both_A':>8} {'both_B':>8}")
    for c in conds:
        p = per[c]
        ba = f"{p['both_all']:9.4f}" if "both_all" in p else f"{'—':>9}"
        bA = f"{p['both_A']:8.4f}" if "both_A" in p else f"{'—':>8}"
        bB = f"{p['both_B']:8.4f}" if "both_B" in p else f"{'—':>8}"
        print(f"  {c:26s} {p['direct']:8.4f} {p['worst_seed']:8.4f} "
              f"{p['firstkey']:9.4f} {ba} {bA} {bB}")
    print(f"\n  ceiling 0.7522 | paired CI clean {ci['mean']*100:+.2f}pp "
          f"[{ci['lower95']*100:+.2f}, {ci['upper95']*100:+.2f}]")
    print(f"\n  GATES: {json.dumps(gates, indent=1)}")
    print(f"  confirmation_pass: {summary['confirmation_pass']}")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
