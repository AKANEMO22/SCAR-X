"""Round 5B Stage 1 — information-path audit with the H1/H2 separator.

The key measurement is not "can the state decode the answer". It is the triple:

  (a) decode vA alone            -> is bit A there
  (b) decode vB alone            -> is bit B there
  (c) decode the JOINT 4-state   -> are they bound to the right keys

and the control that makes (c) meaningful:

  (d) bag-of-values probe: can the state tell {01} from {10}?

A probe that reads only the multiset {vA, vB} can distinguish 00 / {01,10} / 11 but
CANNOT separate 01 from 10. So:

  * (c) succeeds and (d) fails  -> binding exists       (H2 supported)
  * (c) fails and (d) fails     -> only one bit stored  (H1 refuted)
  * (c) succeeds and (d) succeeds -> the probe is reading the key markers directly,
                                     which is a shortcut, not binding (H2 refuted)

All frozen. No candidate is trained in this stage.
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

DEV = (97001, 97007, 97013)
N_TRAIN, N_TEST = 1024, 512
QT = G.QUERY_TICK


def _bag_labels(vA, vB):
    """0 = both zero, 1 = mixed, 2 = both one. Collapses 01 and 10 together."""
    return (vA + vB).astype(np.int32)


def multiclass_ridge(train_f, train_y, test_f, test_y, n_classes, alpha=.01):
    mu = train_f.mean(0)
    sd = np.maximum(train_f.std(0), 1e-6)
    A = np.column_stack([(train_f - mu) / sd, np.ones(len(train_f))])
    B = np.column_stack([(test_f - mu) / sd, np.ones(len(test_f))])
    T = np.eye(n_classes)[train_y]
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ T)
    return float(np.mean(np.argmax(B @ W, 1) == test_y))


def mixed_only_accuracy(train_f, train_y, test_f, test_y, vA, vB):
    """Restrict to the {01,10} episodes: this is where bag info is useless."""
    m_tr = vA != vB
    m_te = vA != vB
    return multiclass_ridge(train_f[m_tr], train_y[m_tr],
                            test_f[m_te], test_y[m_te], 2)


def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r5/protocol-v1.md"
    psha = __import__("hashlib").sha256(proto.read_bytes()).hexdigest()

    run = new_run(root, "r5-information-path-v1", {
        "protocol_sha256": psha, "seeds": list(DEV),
        "n_train": N_TRAIN, "n_test": N_TEST, "query_tick": QT,
        "note": "frozen diagnostics only; no candidate trained"})

    rows = []
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            trx, try_, tvA, tvB, tq, _ = G.independent_query_set_v1(seed + 200_000, N_TRAIN)
            tex, tey, evA, evB, eq, _ = G.independent_query_set_v1(seed + 300_000, N_TEST)
            ctr, cte = P.capture(seed, trx, query_tick=QT), P.capture(seed, tex, query_tick=QT)

            rec = {"seed": seed,
                   "data_sha256": {"train": G.dataset_sha256(trx, try_),
                                   "test": G.dataset_sha256(tex, tey)}}

            def blk(f_tr, f_te):
                d = {"dims": int(f_tr.shape[1])}
                # (a) vA, (b) vB, joint 4-state, (d) bag 3-state
                d["vA"] = P.ridge(f_tr, tvA, f_te, evA)
                d["vB"] = P.ridge(f_tr, tvB, f_te, evB)
                d["vA_shuf"] = P.ridge(f_tr, np.random.default_rng(1).permutation(tvA), f_te, evA)
                d["vB_shuf"] = P.ridge(f_tr, np.random.default_rng(2).permutation(tvB), f_te, evB)
                j_tr = tvA * 2 + tvB
                j_te = evA * 2 + evB
                d["joint4"] = multiclass_ridge(f_tr, j_tr, f_te, j_te, 4)
                d["joint4_shuf"] = multiclass_ridge(
                    f_tr, np.random.default_rng(3).permutation(j_tr), f_te, j_te, 4)
                d["bag3"] = multiclass_ridge(f_tr, _bag_labels(tvA, tvB),
                                             f_te, _bag_labels(evA, evB), 3)
                # the decisive one: among {01,10} only, can it say which is which
                # among {01,10} only: bag information is useless here
                if (tvA != tvB).any() and (evA != evB).any():
                    d["mixed_pair"] = multiclass_ridge(
                        f_tr[tvA != tvB], tvA[tvA != tvB],
                        f_te[evA != evB], evA[evA != evB], 2)
                # answer itself, and the online head on the same features
                d["answer"] = P.ridge(f_tr, try_, f_te, tey)
                h = P.train_head(f_tr, try_, seed=seed)
                d["head"] = h.acc(f_te, tey)
                d["head_train"] = h.acc(f_tr, try_)
                return d

            rec["R1_pre"] = blk(*[P.state_vector(c, "pre") for c in (ctr, cte)])
            rec["R2_traj"] = blk(ctr["final"]["spike_traj"], cte["final"]["spike_traj"])
            rec["R3_post"] = blk(*[P.state_vector(c, "post") for c in (ctr, cte)])
            for nm in ("decay025", "spikesum", "bank5"):
                rec[f"R4_{nm}"] = blk(ctr["final"][nm], cte["final"][nm])

            # R5 historical mechanism
            b_tr = P.bilinear_query_map(P.state_vector(ctr, "post"),
                                        G.query_cue(trx, tick=QT))
            b_te = P.bilinear_query_map(P.state_vector(cte, "post"),
                                        G.query_cue(tex, tick=QT))
            rec["R5"] = blk(b_tr, b_te)

            rows.append(rec)
            mon.check()
            print(f"  seed {seed} done", flush=True)

    def M(rep, key):
        return float(np.mean([r[rep][key] for r in rows]))

    print("\n" + "=" * 92)
    print("STAGE 1 — information path (frozen, 3 dev seeds, mean)")
    print("=" * 92)
    print(f"  {'rep':12s} {'dims':>5} {'vA':>7} {'vB':>7} {'joint4':>7} "
          f"{'bag3':>7} {'mixed':>7} {'answer':>7} {'head':>7}")
    for rep in ("R1_pre", "R2_traj", "R3_post", "R4_decay025", "R4_spikesum",
                "R4_bank5", "R5"):
        print(f"  {rep:12s} {rows[0][rep]['dims']:>5} {M(rep,'vA'):7.4f} {M(rep,'vB'):7.4f} "
              f"{M(rep,'joint4'):7.4f} {M(rep,'bag3'):7.4f} {M(rep,'mixed_pair'):7.4f} "
              f"{M(rep,'answer'):7.4f} {M(rep,'head'):7.4f}")
    print(f"\n  {'chance':12s} {'':>5} {0.5:7.4f} {0.5:7.4f} {0.25:7.4f} "
          f"{0.3333:7.4f} {0.5:7.4f} {0.5:7.4f} {0.5:7.4f}")

    summary = {"seeds": list(DEV),
               "table": {rep: {k: M(rep, k) for k in
                               ("vA", "vB", "joint4", "bag3", "mixed_pair",
                                "answer", "head", "head_train", "dims")}
                         for rep in ("R1_pre", "R2_traj", "R3_post",
                                     "R4_decay025", "R4_spikesum", "R4_bank5", "R5")},
               "chance": {"vA": .5, "vB": .5, "joint4": .25, "bag3": 1/3,
                          "mixed_pair": .5, "answer": .5, "head": .5},
               "shortcut_ceiling": .7522}

    # H1/H2 verdict logic, decided from the numbers rather than asserted
    r1 = summary["table"]["R1_pre"]
    h1 = r1["vA"] > 0.70 and r1["vB"] > 0.70
    h2 = r1["joint4"] > 0.60 and r1["mixed_pair"] > 0.70
    shortcut_bag = r1["bag3"] > 0.95
    summary["verdicts"] = {
        "H1_storage_both_bits": h1,
        "H2_binding_right_key": h2,
        "bag_probe_also_succeeds": shortcut_bag,
        "note": "if bag3 == joint4 the probe may read key markers directly, so H2 "
                "would be a shortcut rather than binding",
    }
    write_json(run / "result.json",
               {"trials": rows, "summary": summary, "resources": mon.report()})

    print("\n" + "=" * 92)
    print("H1 / H2 SEPARATION (pre-query state)")
    print("=" * 92)
    print(f"  vA {r1['vA']:.4f}   vB {r1['vB']:.4f}   joint4 {r1['joint4']:.4f}   "
          f"bag3 {r1['bag3']:.4f}   mixed-only {r1['mixed_pair']:.4f}")
    print(f"  H1 both bits held      : {h1}")
    print(f"  H2 bound to right key  : {h2}")
    print(f"  bag probe also succeeds: {shortcut_bag}  "
          f"({'shortcut risk' if shortcut_bag else 'binding, not bag'})")
    print(f"\nRESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
