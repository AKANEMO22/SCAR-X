"""Round 5B Stage 2 — functional dissection of R5, and a corrected H1/H2 verdict.

CORRECTION TO MY OWN STAGE-1 LOGIC
----------------------------------
Stage 1 flagged "bag probe also succeeds -> shortcut risk" because bag3 = 0.9518.
That reasoning was wrong. If vA and vB are each decodable (0.9967 / 0.9974), then
their SUM is decodable by construction, so bag3 succeeding is FORCED by storage and
carries no information about shortcutting.

The genuine H1/H2 separator is `mixed_pair`: among episodes with vA != vB, can a
probe say which key holds 0 and which holds 1? A bag-of-values code cannot. Measured
0.9947, so binding is real.

That still leaves the shortcut this round actually has to rule out: could a reader
succeed without using the query, by always answering key A (ceiling 0.75)? Stage 1
shows every un-conditioned head lands at 0.75-0.78, i.e. exactly at that ceiling.

Stage 2 therefore dissects R5 against eight controls.
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


def stats(pred, y, vA, vB, q):
    """Accuracy plus the metrics a 0.75 shortcut cannot fake."""
    acc = float(np.mean(pred == y))
    # per-query breakdown: a first-key reader is 1.00 on q=A and 0.50 on q=B
    acc_qA = float(np.mean(pred[q == 0] == y[q == 0])) if (q == 0).any() else float("nan")
    acc_qB = float(np.mean(pred[q == 1] == y[q == 1])) if (q == 1).any() else float("nan")
    # does it ever answer key B correctly when vB != vA? that needs the query
    need_b = (q == 1) & (vA != vB)
    acc_need_b = float(np.mean(pred[need_b] == y[need_b])) if need_b.any() else float("nan")
    return {"acc": acc, "acc_qA": acc_qA, "acc_qB": acc_qB, "acc_need_b": acc_need_b}


def paired_stats(pred, y, vA, vB, pid, flips):
    """Case A must flip, case B must not. Both-answers-correct is the real metric."""
    a, b = pred[0::2], pred[1::2]
    ya, yb = y[0::2], y[1::2]
    fA = flips
    fB = ~flips
    flip_cons = float(np.mean(a[fA] != b[fA])) if fA.any() else float("nan")
    invar = float(np.mean(a[fB] == b[fB])) if fB.any() else float("nan")
    both_a = float(np.mean((a[fA] == ya[fA]) & (b[fA] == yb[fA]))) if fA.any() else float("nan")
    both_b = float(np.mean((a[fB] == ya[fB]) & (b[fB] == yb[fB]))) if fB.any() else float("nan")
    both = float(np.mean(((a == ya) & (b == yb)))) 
    return {"flip_consistency_A": flip_cons, "invariance_B": invar,
            "both_correct_A": both_a, "both_correct_B": both_b, "both_correct": both,
            "n_caseA": int(fA.sum()), "n_caseB": int(fB.sum())}


def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r5/protocol-v1.md"
    psha = __import__("hashlib").sha256(proto.read_bytes()).hexdigest()
    run = new_run(root, "r5-dissection-v1", {
        "protocol_sha256": psha, "seeds": list(DEV),
        "n_train": N_TRAIN, "n_test": N_TEST,
        "controls": "eight R5 variants per protocol section 9"})

    rows = []
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            trx, try_, tvA, tvB, tq, _ = G.independent_query_set_v1(seed + 200_000, N_TRAIN)
            tex, tey, evA, evB, eq, _ = G.independent_query_set_v1(seed + 300_000, N_TEST)
            ptx, pty, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(
                seed + 400_000, 256)

            ctr = P.capture(seed, trx, query_tick=QT)
            cte = P.capture(seed, tex, query_tick=QT)
            cpt = P.capture(seed, ptx, query_tick=QT)

            post_tr = P.state_vector(ctr, "post")
            post_te = P.state_vector(cte, "post")
            post_pt = P.state_vector(cpt, "post")

            cue_tr = G.query_cue(trx, tick=QT)
            cue_te = G.query_cue(tex, tick=QT)
            cue_pt = G.query_cue(ptx, tick=QT)

            rng = np.random.default_rng(seed + 4242)
            cue_tr_sh = cue_tr[rng.permutation(len(cue_tr))]
            cue_te_sh = cue_te[rng.permutation(len(cue_te))]
            cue_pt_sh = cue_pt[rng.permutation(len(cue_pt))]

            # query zeroed at the neural core; cue still supplied externally
            zx = tex.copy(); zx[:, :, 4:6] = 0.0
            ztr = trx.copy(); ztr[:, :, 4:6] = 0.0
            czt = P.capture(seed, ztr, query_tick=QT)
            cze = P.capture(seed, zx, query_tick=QT)
            zpost_tr = P.state_vector(czt, "post")
            zpost_te = P.state_vector(cze, "post")

            rec = {"seed": seed}
            arms = {}

            def add(name, f_tr, f_te, y_tr=None):
                y_tr = try_ if y_tr is None else y_tr
                h = P.train_head(f_tr, y_tr, seed=seed)
                p = h.predict(f_te)
                arms[name] = {**stats(p, tey, evA, evB, eq),
                              "train": h.acc(f_tr, y_tr),
                              "ridge": P.ridge(f_tr, y_tr, f_te, tey),
                              "dims": int(f_tr.shape[1])}

            # 1 original R5 fixed interaction
            add("R5_original",
                P.bilinear_query_map(post_tr, cue_tr),
                P.bilinear_query_map(post_te, cue_te))
            # 2 post-state linear, same width via projection
            prng = np.random.default_rng(seed + 77)
            Wp = prng.normal(0, 1/np.sqrt(post_tr.shape[1]),
                             (post_tr.shape[1], 258)).astype(np.float32)
            add("poststate_linear_258", post_tr @ Wp, post_te @ Wp)
            # 3 R4 features matched capacity
            spl_tr = np.concatenate([ctr["final"]["decay025"],
                                     ctr["final"]["spikesum"],
                                     ctr["final"]["bank5"]], 1)
            spl_te = np.concatenate([cte["final"]["decay025"],
                                     cte["final"]["spikesum"],
                                     cte["final"]["bank5"]], 1)
            W2 = prng.normal(0, 1/np.sqrt(spl_tr.shape[1]),
                             (spl_tr.shape[1], 258)).astype(np.float32)
            add("R4_matched_258", spl_tr @ W2, spl_te @ W2)
            # 4 query only
            add("query_only", cue_tr, cue_te)
            # 5 query-free state
            add("query_free", post_tr, post_te)
            # 6 true cue (same as 1, kept as the explicit reference arm)
            add("true_cue", P.bilinear_query_map(post_tr, cue_tr),
                P.bilinear_query_map(post_te, cue_te))
            # 7a cue shuffled in TRAIN only (test cue correct)
            add("cue_shuf_train",
                P.bilinear_query_map(post_tr, cue_tr_sh),
                P.bilinear_query_map(post_te, cue_te))
            # 7b cue shuffled at TEST only (train correct)
            add("cue_shuf_test",
                P.bilinear_query_map(post_tr, cue_tr),
                P.bilinear_query_map(post_te, cue_te_sh))
            # 8 query zeroed at the core, cue still supplied
            add("query_zeroed_plus_cue",
                P.bilinear_query_map(zpost_tr, cue_tr),
                P.bilinear_query_map(zpost_te, cue_te))

            # paired counterfactual on the R5 arm
            h5 = P.train_head(P.bilinear_query_map(post_tr, cue_tr), try_, seed=seed)
            ppred = h5.predict(P.bilinear_query_map(post_pt, cue_pt))
            rec["R5_paired"] = paired_stats(ppred, pty, pvA, pvB, pid, flips)

            # paired on the first-key shortcut, as the pair-specific null
            fk = G.baseline_first_key(ptx)
            rec["firstkey_paired"] = paired_stats(fk, pty, pvA, pvB, pid, flips)

            # paired on query-free state
            hq = P.train_head(post_tr, try_, seed=seed)
            rec["queryfree_paired"] = paired_stats(hq.predict(post_pt), pty, pvA, pvB,
                                                  pid, flips)

            rec["arms"] = arms
            rows.append(rec)
            mon.check()
            print(f"  seed {seed} done", flush=True)

    def M(arm, key):
        return float(np.nanmean([r["arms"][arm][key] for r in rows]))

    def MP(key):
        return float(np.nanmean([r["R5_paired"][key] for r in rows]))

    print("\n" + "=" * 96)
    print("STAGE 2 — R5 dissected against eight controls (3 dev seeds, mean)")
    print("=" * 96)
    print(f"  {'arm':24s} {'dims':>5} {'acc':>7} {'q=A':>7} {'q=B':>7} "
          f"{'needB':>7} {'train':>7} {'ridge':>7}")
    order = ("R5_original", "true_cue", "poststate_linear_258", "R4_matched_258",
             "query_only", "query_free", "cue_shuf_train", "cue_shuf_test",
             "query_zeroed_plus_cue")
    for a in order:
        print(f"  {a:24s} {int(M(a,'dims')):>5} {M(a,'acc'):7.4f} {M(a,'acc_qA'):7.4f} "
              f"{M(a,'acc_qB'):7.4f} {M(a,'acc_need_b'):7.4f} {M(a,'train'):7.4f} "
              f"{M(a,'ridge'):7.4f}")

    print("\n" + "=" * 96)
    print("PAIRED COUNTERFACTUAL — the metric a 0.75 shortcut cannot fake")
    print("=" * 96)
    print(f"  {'reader':22s} {'both':>7} {'caseA flips':>12} {'caseA both':>11} "
          f"{'caseB invar':>12} {'caseB both':>11}")
    for nm, src in (("R5", "R5_paired"), ("first-key (null)", "firstkey_paired"),
                    ("query-free", "queryfree_paired")):
        d = [r[src] for r in rows]
        print(f"  {nm:22s} {np.nanmean([x['both_correct'] for x in d]):7.4f} "
              f"{np.nanmean([x['flip_consistency_A'] for x in d]):12.4f} "
              f"{np.nanmean([x['both_correct_A'] for x in d]):11.4f} "
              f"{np.nanmean([x['invariance_B'] for x in d]):12.4f} "
              f"{np.nanmean([x['both_correct_B'] for x in d]):11.4f}")

    summary = {
        "arms": {a: {k: M(a, k) for k in ("acc", "acc_qA", "acc_qB", "acc_need_b",
                                          "train", "ridge", "dims")}
                 for a in order},
        "paired": {nm: {k: float(np.nanmean([r[src][k] for r in rows]))
                        for k in ("both_correct", "flip_consistency_A",
                                  "both_correct_A", "invariance_B", "both_correct_B")}
                   for nm, src in (("R5", "R5_paired"),
                                   ("firstkey", "firstkey_paired"),
                                   ("queryfree", "queryfree_paired"))},
        "shortcut_ceiling": 0.7522,
        "stage1_correction": ("bag3 succeeding is FORCED by storage when vA and vB are "
                              "each decodable; the real H1/H2 separator is mixed_pair"),
    }
    write_json(run / "result.json",
               {"trials": rows, "summary": summary, "resources": mon.report()})
    print(f"\nRESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
