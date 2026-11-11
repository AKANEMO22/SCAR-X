"""R4A Stage 1b — Q A split by key, and the §4 counterfactual controls as a RUN.

Two gaps closed from the first pass:

1. Brief §6 Question A requires the value of key A and key B reported SEPARATELY,
   and asks whether input ordering or channel asymmetry explains any difference.
2. Brief §4 controls 3, 4, 5 (reversed write order, query reversal, shuffled
   association) were verified as unit tests but never measured as decodability
   contrasts. This runs them.

All frozen weights, shuffled-label control on every probe.
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

DEV = (95001, 95007, 95013)
N_TRAIN, N_TEST = 1024, 512


def value_of_key(seqs, key, query_always=None):
    """Oracle: the value bound to `key`, read from the input spec.

    When `query_always` is given, the episode's own query tick is overwritten so
    the written association is unchanged but the requested key is fixed.
    """
    x = np.asarray(seqs, np.float32).copy()
    if query_always is not None:
        x[:, 10, 4:6] = 0.0
        x[:, 10, 4 + query_always] = R.PULSE
    out = np.zeros(len(x), np.int32)
    for i, row in enumerate(x):
        vals = {}
        for place in (2, 6):
            k = int(np.argmax(row[place, :2]))
            v = int(np.argmax(row[place, 2:4]))
            vals[k] = v
        out[i] = vals[key]
    return out


def probe(train_f, train_y, test_f, test_y, seedoff=1):
    return {"ridge": P.ridge(train_f, train_y, test_f, test_y),
            "shuffled": P.ridge(train_f, np.random.default_rng(seedoff).permutation(train_y),
                                test_f, test_y)}


def main():
    root = Path(__file__).resolve().parents[1]
    run = new_run(root, "r4-qa-and-controls-v1", {
        "seeds": list(DEV), "n_train": N_TRAIN, "n_test": N_TEST,
        "note": "Q A split by key; brief controls 3-5 measured as contrasts"})

    rows = []
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            trx, try_, _ = R.query_set(seed + 200_000, N_TRAIN)
            tex, tey, _ = R.query_set(seed + 300_000, N_TEST)
            ctr = P.capture(seed, trx)
            cte = P.capture(seed, tex)
            rec = {"seed": seed}

            # ---------- Q A: decode the value of key A and of key B separately
            for rep, where in (("R1_pre", "pre"), ("R3_post", "post")):
                f_tr = P.state_vector(ctr, ("spikes", "voltage", "adaptation", "features"), where)
                f_te = P.state_vector(cte, ("spikes", "voltage", "adaptation", "features"), where)
                for key in (0, 1):
                    y_tr = value_of_key(trx, key)
                    y_te = value_of_key(tex, key)
                    rec[f"QA_{rep}_key{key}"] = probe(f_tr, y_tr, f_te, y_te)
            # R2 trajectory for the same two decodes
            tr2 = ctr["final"]["spike_traj"]; te2 = cte["final"]["spike_traj"]
            for key in (0, 1):
                rec[f"QA_R2_key{key}"] = probe(tr2, value_of_key(trx, key),
                                               te2, value_of_key(tex, key))
            # asymmetry check: is key A easier than key B?
            rec["QA_asymmetry_R1"] = abs(rec["QA_R1_pre_key0"]["ridge"] -
                                         rec["QA_R1_pre_key1"]["ridge"])

            # ---------- control 3+4: reversed write order and query reversal
            rev_te, _, _ = R.reversed_write_order(seed + 300_000, N_TEST)
            cre = P.capture(seed, rev_te)
            # decode from the REVERSED episodes using the ORIGINAL training set
            f_tr = P.state_vector(ctr, ("spikes", "voltage", "adaptation", "features"), "post")
            f_re = P.state_vector(cre, ("spikes", "voltage", "adaptation", "features"), "post")
            rec["C3_reversed_order"] = probe(f_tr, try_, f_re, tey)

            # query reversal: flip the query tick, the answer must invert
            qflip = tex.copy()
            q = (qflip[:, 10, 5] > .5).astype(np.int32)
            qflip[:, 10, 4:6] = 0.0
            qflip[np.arange(len(tex)), 10, 4 + (1 - q)] = R.PULSE
            cq = P.capture(seed, qflip)
            f_q = P.state_vector(cq, ("spikes", "voltage", "adaptation", "features"), "post")
            rec["C4_query_flipped"] = probe(f_tr, try_, f_q, 1 - tey)

            # control 5: shuffled association
            xs, ys = R.shuffled_association(seed + 300_000, N_TEST)
            cs = P.capture(seed, xs)
            xs_tr, ys_tr = R.shuffled_association(seed + 200_000, N_TRAIN)
            cst = P.capture(seed, xs_tr)
            f_str = P.state_vector(cst, ("spikes", "voltage", "adaptation", "features"), "post")
            f_s = P.state_vector(cs, ("spikes", "voltage", "adaptation", "features"), "post")
            rec["C5_shuffled_assoc"] = probe(f_str, ys_tr, f_s, ys)

            rows.append(rec)
            mon.check()
            print(f"  seed {seed} done", flush=True)

    def M(k, sub):
        return float(np.mean([r[k][sub] for r in rows]))

    print("\n" + "=" * 84)
    print("Q A — decode the value of key A and key B SEPARATELY (ridge, shuffled ctl)")
    print("=" * 84)
    print(f"  {'representation':14s} {'key A':>9} {'shuf':>7} {'key B':>9} {'shuf':>7}")
    for rep in ("R1_pre", "R2", "R3_post"):
        a = M(f"QA_{rep}_key0", "ridge"); as_ = M(f"QA_{rep}_key0", "shuffled")
        b = M(f"QA_{rep}_key1", "ridge"); bs = M(f"QA_{rep}_key1", "shuffled")
        print(f"  {rep:14s} {a:9.4f} {as_:7.4f} {b:9.4f} {bs:7.4f}")
    print(f"\n  |A - B| asymmetry (R1): {np.mean([r['QA_asymmetry_R1'] for r in rows]):.4f}")

    print("\n" + "=" * 84)
    print("BRIEF §4 CONTROLS 3-5, measured as decodability contrasts")
    print("=" * 84)
    for k, desc in (("C3_reversed_order", "reversed write order, original labels"),
                    ("C4_query_flipped", "query flipped, inverted labels"),
                    ("C5_shuffled_assoc", "shuffled association, rewritten labels")):
        print(f"  {desc:44s} ridge {M(k,'ridge'):.4f}  shuf {M(k,'shuffled'):.4f}")

    summary = {"seeds": list(DEV),
               "QA": {rep: {f"key{k}": {"ridge": M(f"QA_{rep}_key{k}", "ridge"),
                                        "shuffled": M(f"QA_{rep}_key{k}", "shuffled")}
                            for k in (0, 1)}
                      for rep in ("R1_pre", "R2", "R3_post")},
               "controls": {k: {"ridge": M(k, "ridge"), "shuffled": M(k, "shuffled")}
                            for k in ("C3_reversed_order", "C4_query_flipped",
                                      "C5_shuffled_assoc")},
               "asymmetry_R1": float(np.mean([r["QA_asymmetry_R1"] for r in rows]))}
    write_json(run / "result.json",
               {"trials": rows, "summary": summary, "resources": mon.report()})
    print(f"\nRESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
