"""Round 4A Stage 1 — registered information-path audit R1-R5 with H4 controls.

Produces the location-of-failure table. No capability claim: every number here is
a frozen-state diagnostic with a shuffled-label control.

Q A = can the stored value be decoded BEFORE the query?
Q B = can the query-selected value be decoded AFTER the query?
Q C = can the registered online head learn the mapping on the same state?
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r4_probe as P  # noqa: E402
from scarx.evaluation import r4_relational as R  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

DEV = (95001, 95007, 95013)
N_TRAIN, N_TEST = 1024, 512


# ------------------------------------------------------------------ helpers
def consistency(pred, y, pid):
    """Fraction of counterfactual pairs whose two predictions differ as required.

    Only counted on pairs whose true labels differ, so equal-value pairs are not
    mis-scored as failures.
    """
    a, b = pred[0::2], pred[1::2]
    ya, yb = y[0::2], y[1::2]
    valid = ya != yb
    if not valid.any():
        return 0.0, 0
    ok = (a != b)[valid]
    return float(np.mean(ok)), int(valid.sum())


def null_consistency(seed=7, n_pairs=64):
    """Expected paired consistency of a RANDOM predictor on the same structure.

    A random predictor flips ~50% of pairs, so this measures the null that the
    observed consistency must beat.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n_pairs)          # prediction on the first sibling
    b = rng.integers(0, 2, n_pairs)          # prediction on the second sibling
    return float(np.mean(a != b))


def controls(seqs):
    """The H4 shortcut feature sets, built from the INPUT only."""
    x = np.asarray(seqs, np.float32)
    n = len(x)
    q = (x[:, 10, 5] > .5).astype(np.float32)
    query_only = np.stack([1 - q, q], 1)
    keys = np.concatenate([x[:, 2, :2], x[:, 6, :2]], 1)      # which key where
    pos = np.tile(np.array([2, 6, 10], np.float32), (n, 1))   # constant ticks
    return {"query_only": query_only, "key_only": keys, "position_only": pos}


def first_key_policy(seqs):
    """Always answer the value of the first-written key. Baseline check."""
    x = np.asarray(seqs, np.float32)
    out = np.zeros(len(x), np.int32)
    for i in range(len(x)):
        vals = {}
        for place in (2, 6):
            k = int(np.argmax(x[i, place, :2]))
            v = int(np.argmax(x[i, place, 2:4]))
            vals[k] = v
        out[i] = vals[0]                       # key A, always
    return out


# ------------------------------------------------------------------- stage
def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r4/protocol-v1.md"
    psha = hashlib.sha256(proto.read_bytes()).hexdigest()

    run = new_run(root, "r4-information-path-v1", {
        "protocol_sha256": psha, "seeds": list(DEV),
        "n_train": N_TRAIN, "n_test": N_TEST, "core": "ALIF H=48 hard threshold",
        "hidden": "frozen", "reader": "ridge diagnostic + R2A head"})

    rows = []
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            trx, try_, _ = R.query_set(seed + 200_000, N_TRAIN)
            tex, tey, _ = R.query_set(seed + 300_000, N_TEST)
            # paired counterfactual test set for consistency scoring
            px, py, pid, _ = R.paired_counterfactual(seed + 400_000, N_TEST // 2)

            ctr = P.capture(seed, trx)
            cte = P.capture(seed, tex)
            cpx = P.capture(seed, px)

            rec = {"seed": seed,
                   "data_sha256": {"train": R.dataset_sha256(trx, try_),
                                   "test": R.dataset_sha256(tex, tey),
                                   "paired": R.dataset_sha256(px, py)}}

            # ---- Q A: pre-query value of key A / key B from R1 and R2
            for rep, where in (("R1_pre_state", "pre"), ("R3_post_state", "post")):
                f_tr = P.state_vector(ctr, ("spikes", "voltage", "adaptation", "features"), where)
                f_te = P.state_vector(cte, ("spikes", "voltage", "adaptation", "features"), where)
                acc = P.ridge(f_tr, try_, f_te, tey)
                shuf = P.ridge(f_tr, np.random.default_rng(1).permutation(try_), f_te, tey)
                rec[rep] = {"ridge": acc, "shuffled": shuf, "dims": int(f_tr.shape[1])}

            # R2: full spike trajectory
            tr2 = ctr["final"]["spike_traj"]; te2 = cte["final"]["spike_traj"]
            rec["R2_spike_traj"] = {
                "ridge": P.ridge(tr2, try_, te2, tey),
                "shuffled": P.ridge(tr2, np.random.default_rng(1).permutation(try_), te2, tey),
                "dims": int(tr2.shape[1])}

            # ---- R4 established features
            for name in ("decay025", "spikesum", "bank5"):
                a = ctr["final"][name]; b = cte["final"][name]
                rec[f"R4_{name}"] = {
                    "ridge": P.ridge(a, try_, b, tey),
                    "shuffled": P.ridge(a, np.random.default_rng(1).permutation(try_), b, tey),
                    "dims": int(a.shape[1])}

            # ---- R5 bounded query-conditioned bilinear map on post-query state
            q_tr = P.query_cue(trx); q_te = P.query_cue(tex)
            post_tr = P.state_vector(ctr, ("spikes", "voltage", "adaptation", "features"), "post")
            post_te = P.state_vector(cte, ("spikes", "voltage", "adaptation", "features"), "post")
            bil_tr = P.bilinear_query_map(post_tr, q_tr)
            bil_te = P.bilinear_query_map(post_te, q_te)
            rec["R5_bilinear"] = {
                "ridge": P.ridge(bil_tr, try_, bil_te, tey),
                "shuffled": P.ridge(bil_tr, np.random.default_rng(1).permutation(try_), bil_te, tey),
                "dims": int(bil_tr.shape[1])}

            # ---- Q C: the registered online head on the same representations
            for name, key in (("decay025", "R4_decay025"), ("spikesum", "R4_spikesum"),
                              ("bank5", "R4_bank5")):
                a = ctr["final"][name]; b = cte["final"][name]
                h = P.train_head(a, try_)
                pr = h.predict(b)
                cons, nvalid = consistency(pr, tey, np.arange(len(py)) // 2)
                rec[key]["head"] = h.acc(b, tey)
                rec[key]["head_train"] = h.acc(a, try_)
                rec[key]["paired_consistency"] = cons
            hb = P.train_head(bil_tr, try_)
            pb = hb.predict(bil_te)
            rec["R5_bilinear"]["head"] = hb.acc(bil_te, tey)
            rec["R5_bilinear"]["head_train"] = hb.acc(bil_tr, try_)

            # ---- H4 controls on the SAME samples
            ctrl = controls(tex)
            ctrl_tr = controls(trx)
            for cname in ctrl:
                rec[f"control_{cname}"] = P.ridge(ctrl_tr[cname], try_, ctrl[cname], tey)
            fk = first_key_policy(tex)
            rec["control_first_key_policy"] = float(np.mean(fk == tey))
            rec["control_majority"] = float(max(np.mean(tey == 0), np.mean(tey == 1)))

            # ---- paired counterfactual decode on the paired set
            cp_te = P.state_vector(cpx, ("spikes", "voltage", "adaptation", "features"), "post")
            qp = P.query_cue(px)
            bil_p = P.bilinear_query_map(cp_te, qp)
            pred_p = np.argmax(
                np.column_stack([bil_p, np.ones(len(bil_p))]) @
                np.linalg.solve(
                    np.column_stack([bil_tr, np.ones(len(bil_tr))]).T @
                    np.column_stack([bil_tr, np.ones(len(bil_tr))]) + .01 * np.eye(bil_tr.shape[1] + 1),
                    np.column_stack([bil_tr, np.ones(len(bil_tr))]).T @ np.eye(2)[try_]), 1)
            cons_p, nv = consistency(pred_p, py, pid)
            rec["R5_paired_consistency"] = {"value": cons_p, "n_pairs": nv,
                                            "null": null_consistency()}

            rows.append(rec)
            mon.check()
            print(f"  seed {seed} done", flush=True)

    def M(key, sub=None):
        vals = []
        for r in rows:
            v = r[key]
            vals.append(v[sub] if sub else v)
        return float(np.mean(vals))

    summary = {"seeds": list(DEV), "n_trials": len(rows),
               "null_paired_consistency": null_consistency(),
               "table": {}}
    for key in ("R1_pre_state", "R2_spike_traj", "R3_post_state",
                "R4_decay025", "R4_spikesum", "R4_bank5", "R5_bilinear"):
        summary["table"][key] = {
            "ridge": M(key, "ridge"), "shuffled": M(key, "shuffled"),
            "dims": rows[0][key]["dims"]}
        if "head" in rows[0][key]:
            summary["table"][key]["head"] = M(key, "head")
            summary["table"][key]["head_train"] = M(key, "head_train")
    for c in ("query_only", "key_only", "position_only"):
        summary["table"][f"control_{c}"] = M(f"control_{c}")
    summary["table"]["control_first_key_policy"] = M("control_first_key_policy")
    summary["table"]["control_majority"] = M("control_majority")
    summary["table"]["R5_paired_consistency"] = {
        "value": M("R5_paired_consistency", "value"),
        "null": M("R5_paired_consistency", "null")}

    result = {"protocol_sha256": psha, "trials": rows, "summary": summary,
              "resources": mon.report()}
    write_json(run / "result.json", result)

    print("\n" + "=" * 82)
    print("LOCATION-OF-FAILURE TABLE  (mean over 3 dev seeds)")
    print("=" * 82)
    print(f"  {'representation':22s} {'dims':>5} {'ridge':>8} {'shuf':>8} {'head':>8} {'htrain':>8}")
    for key in ("R1_pre_state", "R2_spike_traj", "R3_post_state",
                "R4_decay025", "R4_spikesum", "R4_bank5", "R5_bilinear"):
        t = summary["table"][key]
        h = f"{t['head']:8.4f}" if "head" in t else f"{'—':>8}"
        ht = f"{t['head_train']:8.4f}" if "head_train" in t else f"{'—':>8}"
        print(f"  {key:22s} {t['dims']:>5} {t['ridge']:8.4f} {t['shuffled']:8.4f} {h} {ht}")
    print("\n  H4 CONTROLS")
    for c in ("query_only", "key_only", "position_only"):
        print(f"    {c:22s} {summary['table'][f'control_{c}']:8.4f}")
    print(f"    {'first_key_policy':22s} {summary['table']['control_first_key_policy']:8.4f}")
    print(f"    {'majority':22s} {summary['table']['control_majority']:8.4f}")
    pc = summary["table"]["R5_paired_consistency"]
    print(f"    {'R5 paired consistency':22s} {pc['value']:8.4f}  (null {pc['null']:.4f})")
    print(f"\n  wall {mon.report()['elapsed_seconds']:.1f}s  peak commit "
          f"{mon.report()['peak_commit_or_vms_bytes']:,} B")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
