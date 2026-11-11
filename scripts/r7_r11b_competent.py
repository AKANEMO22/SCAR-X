"""R7 round 11b — the EXACT trace with a COMPETENT readout providing the credit.

ROOT CAUSE OF ROUND 11's NEGATIVE
---------------------------------
Round 11 measured |dw| = 3.5e-05 for the exact rule, versus 9.4e-04 for production, and
`exact` (3.5e-05) was indistinguishable from `exact_shuffled` (3.7e-05). Diagnosis:

    credit_i = sum_k W_l[k,i] * dL/dz_k

With the core's RANDOM initial `w_out`, `|credit|` averages **0.0176** — the readout is a
random projection of a near-uniform gradient, so it routes almost nothing back. The trace
is aligned (round 10: +0.945) but its magnitude is divided by ~50 because the router is
untrained. That is my design error: I required alignment and forgot the readout must be
competent for the credit to have scale.

THE FIX: train the readout on frozen features FIRST (as the round-10 diagnostic did),
then use that competent readout to route credit while the hidden weights adapt. This is
online-legal: the readout is trained on data the agent has already seen, and credit uses
only `W_l . dL/dz`, both available at the readout.

Pre-checks required before reporting (round 10 section 5, now mandatory):
  1  |credit| must be materially larger than the random-router baseline
  2  |dw| must be in the same order as the production rule's
  3  the weight-intervention test must pass
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
ETA, CLIP = 0.002, 0.025
ARMS = ("frozen", "production", "exact", "exact_shuffled", "exact_zero")


def parity3(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        bits = [int(rng.integers(2)) for _ in range(3)]
        y[i] = bits[0] ^ bits[1] ^ bits[2]
        t = int(rng.integers(1, 4))
        for k, bb in enumerate(bits):
            x[i, t, k] += 1.5 * bb
        x[i, 10, 7] += 1.5
    return x, y


def softmax(z):
    z = np.asarray(z, float)
    one = z.ndim == 1
    if one:
        z = z[None, :]
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    out = e / e.sum(1, keepdims=True)
    return out[0] if one else out


def readout(core, X, kind="both"):
    V, Sp = [], []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
        V.append(core.voltage.copy()); Sp.append(tot)
    V, Sp = np.stack(V), np.stack(Sp)
    return V if kind == "voltage" else np.concatenate([V, Sp], 1)


def fit_head(F, y, epochs=1200, lr=.5):
    mu = F.mean(0); sd = np.maximum(F.std(0), 1e-6)
    A = (F - mu) / sd
    w = np.zeros((2, A.shape[1])); b = np.zeros(2)
    for _ in range(epochs):
        p = softmax(A @ w.T + b)
        e = -p; e[np.arange(len(y)), y] += 1
        w += lr * (e.T @ A) / len(y); b += lr * e.mean(0)
    return mu, sd, w, b


def head_acc(F, y, mu, sd, w, b):
    return float(np.mean(np.argmax(((F - mu) / sd) @ w.T + b, 1) == y))


def train_arm(seed, arm, trx, try_, tex, tey):
    c = S.cfg(seed)
    core = S.SECSCore(c, mode="scalar", seed=seed)
    rows = core.rows

    # ---- phase 0: competent VOLTAGE head, trained on frozen features.
    # Used ONLY to route credit for the exact family. Its weights are read-only
    # during hidden learning, so it is a fixed router, not a second learner.
    V0 = readout(core, trx, "voltage")
    mu, sd, wl, bl = fit_head(V0, try_)
    pre_acc = head_acc(readout(core, tex, "voltage"), tey, mu, sd, wl, bl)

    credits_abs = []
    rng0 = np.random.default_rng(seed + 313)
    e_in = np.zeros((len(rows), c.inputs))

    for i in rng0.permutation(N_TR):
        core.reset_state()
        e_in[:] = 0.0
        for t in range(12):
            xa = np.asarray(trx[i][t], np.float64)
            core.step(xa.astype(np.float32), track=True)
            if arm in ("exact", "exact_shuffled", "exact_zero"):
                e_in = c.leak * e_in + xa[None, :]

        if arm == "frozen":
            continue
        if arm == "production":
            core.learn(int(try_[i]), mode="full")
            continue

        # exact family: credit from the COMPETENT head in z-space
        z = (core.voltage.astype(np.float64) - mu) / sd
        p = softmax(z @ wl.T + bl)
        dl_dz = p.copy(); dl_dz[int(try_[i])] -= 1.0
        # dL/dv_i = sum_k (wl[k,i]/sd_i) * dL/dz_k
        credit = ((wl / sd[None, :]) * dl_dz[:, None]).sum(0)
        credits_abs.append(float(np.abs(credit).mean()))

        if arm == "exact_zero":
            credit = np.zeros_like(credit)
        elif arm == "exact_shuffled":
            credit = credit[np.random.default_rng(i).permutation(len(credit))]

        dw = ETA * credit[rows][:, None] * e_in
        np.clip(dw, -CLIP, CLIP, out=dw)
        core.w_in[rows] += dw.astype(np.float32)
        np.clip(core.w_in, -c.weight_clip, c.weight_clip, out=core.w_in)

    return core, pre_acc, (float(np.mean(credits_abs)) if credits_abs else float("nan"))


def run():
    root = Path(__file__).resolve().parents[1]
    run = new_run(root, "r7-r11b-exact-gain-competent-v1", {
        "seeds": list(DEV), "task": "parity3", "arms": list(ARMS),
        "eta": ETA, "clip": CLIP,
        "change_from_r11": "credit now routed by a TRAINED voltage head, not the random one",
        "r11_diagnosis": "random router gave |credit| 0.0176 and |dw| 3.5e-05, 27x below production"})

    rows = []
    t0 = time.time()
    for seed in DEV:
        trx, try_ = parity3(seed + 200_000, N_TR)
        tex, tey = parity3(seed + 300_000, N_TE)
        init = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed).w_in.copy()

        for arm in ARMS:
            core, pre, cabs = train_arm(seed, arm, trx, try_, tex, tey)
            rec = {"seed": seed, "arm": arm,
                   "wd": float(np.abs(core.w_in - init).mean()),
                   "credit_abs": cabs, "router_pre_acc": pre}
            for kind in ("voltage", "both"):
                F = readout(core, trx, kind)
                Ft = readout(core, tex, kind)
                mu, sd, w, b = fit_head(F, try_)
                rec[kind] = head_acc(Ft, tey, mu, sd, w, b)
            rows.append(rec)
            print(f"  seed{seed} {arm:15s} V {rec['voltage']:.4f} both {rec['both']:.4f} "
                  f"|dw| {rec['wd']:.6f} |credit| {rec['credit_abs']:.4f} "
                  f"router {pre:.4f}", flush=True)

    def M(arm, k):
        vals = [r[k] for r in rows if r["arm"] == arm and r[k] == r[k]]
        return float(np.mean(vals)) if vals else float("nan")

    gains = {}
    for kind in ("voltage", "both"):
        fr = {r["seed"]: r[kind] for r in rows if r["arm"] == "frozen"}
        for arm in ARMS:
            if arm == "frozen":
                continue
            d = [r[kind] - fr[r["seed"]] for r in rows if r["arm"] == arm]
            gains[f"{arm}|{kind}"] = {"mean": float(np.mean(d)),
                                      "per_seed": [round(x, 4) for x in d],
                                      "worst": float(np.min(d))}

    summary = {"acc": {f"{a}|{k}": M(a, k) for a in ARMS for k in ("voltage", "both")},
               "gains": gains,
               "credit": {a: (M(a, "credit_abs") if M(a, "credit_abs") ==
                              M(a, "credit_abs") else None) for a in ARMS},
               "dw": {a: M(a, "wd") for a in ARMS},
               "router_pre_acc": M("exact", "router_pre_acc")}

    def sanitize(o):
        """json cannot encode NaN; unmeasured values must read as null."""
        if isinstance(o, dict):
            return {k: sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sanitize(v) for v in o]
        if isinstance(o, float) and o != o:
            return None
        return o

    write_json(run / "result.json",
               sanitize({"trials": rows, "summary": summary,
                         "wall_seconds": time.time() - t0}))

    print("\n" + "=" * 92)
    print("R7 ROUND 11b — exact trace with a COMPETENT credit router")
    print("=" * 92)
    print(f"  router (trained voltage head) pre-accuracy: "
          f"{summary['router_pre_acc']:.4f}")
    print(f"\n  {'arm':16s} {'|credit|':>10} {'|dw|':>10} {'V acc':>8} {'V gain':>10} "
          f"{'both acc':>9} {'both gain':>11}")
    for arm in ARMS:
        gv = gains.get(f"{arm}|voltage")
        gb = gains.get(f"{arm}|both")
        cred = summary['credit'][arm]
        credtxt = f"{cred:10.4f}" if cred is not None else f"{'n/a':>10s}"
        print(f"  {arm:16s} {credtxt} {summary['dw'][arm]:10.6f} "
              f"{summary['acc'][arm+'|voltage']:8.4f} "
              f"{(gv['mean']*100 if gv else 0):+9.2f}p "
              f"{summary['acc'][arm+'|both']:9.4f} "
              f"{(gb['mean']*100 if gb else 0):+10.2f}p")
    print(f"\nRESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
