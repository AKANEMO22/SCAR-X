"""R7 round 11 — does a correct trace convert into a REPRESENTATIONAL gain?

Round 10 established alignment: the EXACT trace (`e_in <- leak*e_in + x`,
`dw = (W_l . dL/dz)_i * e_in`) tracks dL/dw at cosine +0.9452 against a null of -0.0005.

Alignment is not usefulness. This round trains hidden weights with that rule and asks
whether the learned representation is better, using every control the mission requires.

ARMS (identical population, data, seeds, update budget)
    frozen            hidden never changes
    production        production rule verbatim (scalar advantage x Hebbian trace)
    exact             the exact-derivative trace
    exact_shuffled    exact rule, credit permuted across neurons  (signal control)
    exact_zero        exact rule, credit forced to 0              (no-signal control)

EVALUATION (mission section VIII/IX)
    common refit head, z-scaled on train only, identical budget for every arm
    Q1  does the learned representation decode better (refit accuracy)
    Q2  same readout algorithm, learned vs frozen
    Q3  weight intervention: refit on the FROZEN initialisation must reproduce the
        frozen score, proving the head protocol is not carrying the gain
    Q6  shuffled/zero controls must not reproduce the gain

CHANNEL. Round 8 measured voltage 0.5469 vs spikes 0.7812 on parity3, and the exact
trace is only exact for a voltage readout. Round 8 also measured spikes+voltage at
0.8594. To avoid confounding "trace rule" with "channel", BOTH readouts are reported,
and the rule is applied identically.

FALSIFIER (registered in round 10 section 5): if `exact` does not beat `frozen` by >=5pp
on the common refit while `exact_shuffled` does not, the mechanism is REJECTED despite
the alignment.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
ETA, CLIP = 0.002, 0.025           # production values, unchanged
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


def readout(core, X, kind):
    """Frozen readout channels from one pass. kind in {voltage, spike, both}."""
    V, Spk = [], []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
        V.append(core.voltage.copy())
        Spk.append(tot)
    V = np.stack(V)
    Spk = np.stack(Spk)
    return {"voltage": V, "spike": Spk, "both": np.concatenate([V, Spk], 1)}[kind]


def refit(F, y, Ft, yt, epochs=1200, lr=.5):
    mu = F.mean(0); sd = np.maximum(F.std(0), 1e-6)
    A = (F - mu) / sd; B = (Ft - mu) / sd
    w = np.zeros((2, A.shape[1])); b = np.zeros(2)
    for _ in range(epochs):
        p = softmax(A @ w.T + b)
        e = -p; e[np.arange(len(y)), y] += 1
        w += lr * (e.T @ A) / len(y); b += lr * e.mean(0)
    return (float(np.mean(np.argmax(B @ w.T + b, 1) == yt)),
            (mu, sd, w, b))


def train_arm(seed, arm, trx, try_, kind):
    """Online hidden learning on parity3, then return the frozen population."""
    c = S.cfg(seed)
    core = S.SECSCore(c, mode="scalar", seed=seed)
    rng = np.random.default_rng(seed + 313)
    rows = core.rows
    e_in = np.zeros((len(rows), c.inputs), np.float64)

    for i in rng.permutation(N_TR):
        core.reset_state()
        e_in[:] = 0.0
        x_all = trx[i]
        for t in range(12):
            xa = np.asarray(x_all[t], np.float64)
            # advance the population (production forward, unchanged)
            core.step(xa.astype(np.float32), track=True)
            if arm in ("exact", "exact_shuffled", "exact_zero"):
                e_in = c.leak * e_in + xa[None, :]

        if arm == "frozen":
            continue

        if arm == "production":
            core.learn(int(try_[i]), mode="full")
            continue

        # ---- exact-family: readout credit x exact trace
        V = core.voltage.astype(np.float64)
        # head is the core's own random w_out; dL/dz = p - onehot
        p = softmax(V @ core.w_out.T.astype(np.float64) + core.bias)
        dl_dz = p.copy(); dl_dz[int(try_[i])] -= 1.0
        # credit_i = sum_k W_l[k,i] * dL/dz_k = dL/dv_i
        # w_out is (2, H); dl_dz is (2,) -> multiply each row then sum over k
        credit = (core.w_out.astype(np.float64) * dl_dz[:, None]).sum(0)

        if arm == "exact_zero":
            credit = np.zeros_like(credit)
        elif arm == "exact_shuffled":
            credit = credit[np.random.default_rng(i).permutation(len(credit))]

        dw = ETA * credit[rows][:, None] * e_in
        np.clip(dw, -CLIP, CLIP, out=dw)
        core.w_in[rows] += dw.astype(np.float32)
        np.clip(core.w_in, -c.weight_clip, c.weight_clip, out=core.w_in)
    return core


def run():
    root = Path(__file__).resolve().parents[1]
    run = new_run(root, "r7-r11-exact-gain-v1", {
        "seeds": list(DEV), "task": "parity3", "arms": list(ARMS),
        "eta": ETA, "clip": CLIP, "channels": ["voltage", "both"],
        "falsifier": "exact must beat frozen by >=5pp while shuffled does not"})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            trx, try_ = parity3(seed + 200_000, N_TR)
            tex, tey = parity3(seed + 300_000, N_TE)

            for arm in ARMS:
                core = train_arm(seed, arm, trx, try_, None)
                rec = {"seed": seed, "arm": arm,
                       "wd": float(np.abs(core.w_in - S.SECSCore(S.cfg(seed)).w_in).mean())}
                for kind in ("voltage", "both"):
                    F = readout(core, trx, kind)
                    Ft = readout(core, tex, kind)
                    acc, _ = refit(F, try_, Ft, tey)
                    rec[kind] = acc
                rows.append(rec)
                mon.check()
                print(f"  seed{seed} {arm:15s} voltage {rec['voltage']:.4f} "
                      f"both {rec['both']:.4f} |dw| {rec['wd']:.5f}", flush=True)

    # Q3: weight intervention -- refit on the frozen initialisation must equal frozen
    print("\n  [Q3] weight-intervention check")
    inter = []
    for seed in DEV:
        trx, try_ = parity3(seed + 200_000, N_TR)
        tex, tey = parity3(seed + 300_000, N_TE)
        fresh = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        a_v, _ = refit(readout(fresh, trx, "voltage"), try_,
                       readout(fresh, tex, "voltage"), tey)
        frozen_row = [r for r in rows if r["seed"] == seed and r["arm"] == "frozen"][0]
        inter.append(abs(a_v - frozen_row["voltage"]))
        print(f"    seed{seed} fresh-init refit {a_v:.4f} vs frozen arm "
              f"{frozen_row['voltage']:.4f}  diff {abs(a_v - frozen_row['voltage']):.4f}")

    def M(arm, k):
        return float(np.mean([r[k] for r in rows if r["arm"] == arm]))

    gains = {}
    for kind in ("voltage", "both"):
        fr = {r["seed"]: r[kind] for r in rows if r["arm"] == "frozen"}
        for arm in ARMS:
            if arm == "frozen":
                continue
            d = [r[kind] - fr[r["seed"]] for r in rows if r["arm"] == arm]
            gains[f"{arm}|{kind}"] = {"mean": float(np.mean(d)),
                                      "per_seed": d, "worst": float(np.min(d))}

    summary = {f"{arm}|{kind}": M(arm, kind) for arm in ARMS
               for kind in ("voltage", "both")}
    summary = {"acc": summary, "gains": gains,
               "intervention_max_diff": float(np.max(inter))}

    write_json(run / "result.json",
               {"trials": rows, "summary": summary,
                "resources": dict(mon.report()),
                "wall_seconds": time.time() - t0})

    print("\n" + "=" * 90)
    print("R7 ROUND 11 — does alignment convert into a representation gain?")
    print("=" * 90)
    for kind in ("voltage", "both"):
        print(f"\n  channel: {kind}")
        print(f"    {'arm':16s} {'acc':>7} {'vs frozen':>12} {'worst seed':>12}")
        for arm in ARMS:
            acc = summary["acc"][f"{arm}|{kind}"]
            g = summary["gains"].get(f"{arm}|{kind}")
            gtxt = f"{g['mean']*100:+11.2f}pp" if g else f"{'—':>12}"
            wtxt = f"{g['worst']*100:+11.2f}pp" if g else f"{'—':>12}"
            print(f"    {arm:16s} {acc:7.4f} {gtxt} {wtxt}")
    print(f"\n  [Q3] intervention max |diff| = {summary['intervention_max_diff']:.4f}")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
