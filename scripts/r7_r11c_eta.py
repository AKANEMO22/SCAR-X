"""R7 round 11c — magnitude or fundamental? one eta sweep, then stop.

Round 11b result: the EXACT trace (+0.945 alignment, competent router) is the WORST arm.

    arm              |dw|      V acc    V gain    both acc  both gain
    frozen         0.000000   0.5560    +0.00p     0.8542    +0.00p
    production     0.000944   0.5339    -2.21p     0.8372    -1.69p
    exact          0.055805   0.5247    -3.12p     0.8125    -4.17p
    exact_shuffled 0.008550   0.5638    +0.78p     0.8542    +0.00p

Two competing explanations:

  H1 MAGNITUDE. |dw| is 60x the production rule (0.0558 vs 0.0009), which exceeds the
     weight scale itself (~0.45 over the run). The direction is right but the step is
     uncontrolled, so the representation is blasted apart. Predicts: a smaller eta
     recovers a gain.

  H2 FUNDAMENTAL. Following the readout-loss gradient on hidden weights collapses
     feature diversity - greedy loss descent is not representation learning. Predicts:
     no eta recovers a gain, and shuffled stays >= exact at every eta.

This distinguishes them with ONE sweep and no further mechanism work. If H1 holds I will
say so and retune once. If H2 holds the credit-assignment line is closed.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
ETAS = (0.002, 0.0004, 0.00008)
CLIP = 0.025


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


def acc(F, y, mu, sd, w, b):
    return float(np.mean(np.argmax(((F - mu) / sd) @ w.T + b, 1) == y))


def run_arm(seed, arm, eta, trx, try_, tex, tey, init):
    c = S.cfg(seed)
    core = S.SECSCore(c, mode="scalar", seed=seed)
    rows = core.rows
    V0 = readout(core, trx, "voltage")
    mu, sd, wl, bl = fit_head(V0, try_)
    e_in = np.zeros((len(rows), c.inputs))

    for i in np.random.default_rng(seed + 313).permutation(N_TR):
        core.reset_state(); e_in[:] = 0.0
        for t in range(12):
            xa = np.asarray(trx[i][t], np.float64)
            core.step(xa.astype(np.float32), track=True)
            if arm.startswith("exact"):
                e_in = c.leak * e_in + xa[None, :]
        if arm == "frozen":
            continue
        if arm == "production":
            core.learn(int(try_[i]), mode="full"); continue
        z = (core.voltage.astype(np.float64) - mu) / sd
        p = softmax(z @ wl.T + bl)
        d = p.copy(); d[int(try_[i])] -= 1.0
        credit = ((wl / sd[None, :]) * d[:, None]).sum(0)
        if arm == "exact_shuffled":
            credit = credit[np.random.default_rng(i).permutation(len(credit))]
        dw = eta * credit[rows][:, None] * e_in
        np.clip(dw, -CLIP, CLIP, out=dw)
        core.w_in[rows] += dw.astype(np.float32)
        np.clip(core.w_in, -c.weight_clip, c.weight_clip, out=core.w_in)

    out = {"wd": float(np.abs(core.w_in - init).mean())}
    for kind in ("voltage", "both"):
        m, s, w, b = fit_head(readout(core, trx, kind), try_)
        out[kind] = acc(readout(core, tex, kind), tey, m, s, w, b)
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    run = new_run(root, "r7-r11c-eta-sweep-v1", {
        "seeds": list(DEV), "task": "parity3", "etas": list(ETAS),
        "purpose": "distinguish H1 magnitude from H2 fundamental"})

    rows = []
    for eta in ETAS:
        for seed in DEV:
            trx, try_ = parity3(seed + 200_000, N_TR)
            tex, tey = parity3(seed + 300_000, N_TE)
            init = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed).w_in.copy()
            for arm in ("frozen", "exact", "exact_shuffled"):
                r = run_arm(seed, arm, eta, trx, try_, tex, tey, init)
                rec = {"eta": eta, "seed": seed, "arm": arm, **r}
                rows.append(rec)
                print(f"  eta{eta:7f} seed{seed} {arm:15s} "
                      f"V {r['voltage']:.4f} both {r['both']:.4f} "
                      f"|dw| {r['wd']:.6f}", flush=True)

    print("\n" + "=" * 86)
    print("R7 ROUND 11c — eta sweep: magnitude (H1) or fundamental (H2)?")
    print("=" * 86)
    print(f"  {'eta':>9} {'arm':15s} {'|dw|':>10} {'V acc':>8} {'V gain':>9} "
          f"{'both':>8} {'both gain':>10}")
    summary = {}
    for eta in ETAS:
        frozen = {r["seed"]: r for r in rows
                  if r["eta"] == eta and r["arm"] == "frozen"}
        for arm in ("frozen", "exact", "exact_shuffled"):
            sel = [r for r in rows if r["eta"] == eta and r["arm"] == arm]
            gv = float(np.mean([r["voltage"] - frozen[r["seed"]]["voltage"] for r in sel]))
            gb = float(np.mean([r["both"] - frozen[r["seed"]]["both"] for r in sel]))
            key = f"{eta}|{arm}"
            summary[key] = {"wd": float(np.mean([r["wd"] for r in sel])),
                            "voltage": float(np.mean([r["voltage"] for r in sel])),
                            "both": float(np.mean([r["both"] for r in sel])),
                            "gain_voltage_pp": gv * 100, "gain_both_pp": gb * 100}
            print(f"  {eta:9.5f} {arm:15s} {summary[key]['wd']:10.6f} "
                  f"{summary[key]['voltage']:8.4f} {gv*100:+8.2f}p "
                  f"{summary[key]['both']:8.4f} {gb*100:+9.2f}p")
        print()

    best_exact = max((summary[f"{e}|exact"]["gain_both_pp"] for e in ETAS))
    best_shuf = max((summary[f"{e}|exact_shuffled"]["gain_both_pp"] for e in ETAS))
    print("=" * 86)
    print("VERDICT")
    print("=" * 86)
    print(f"  best exact gain    : {best_exact:+.2f}pp")
    print(f"  best shuffled gain : {best_shuf:+.2f}pp")
    if best_exact >= 5.0 and best_exact > best_shuf + 3.0:
        print("  => H1 MAGNITUDE. A smaller eta recovers a real gain, and exact")
        print("     beats its shuffled control. The mechanism is viable.")
        verdict = "H1_MAGNITUDE"
    else:
        print("  => H2 FUNDAMENTAL. No eta recovers a gain and shuffled is not")
        print("     worse. Alignment with the readout loss gradient does not")
        print("     convert into a better representation.")
        verdict = "H2_FUNDAMENTAL"

    def san(o):
        if isinstance(o, dict):
            return {k: san(v) for k, v in o.items()}
        if isinstance(o, list):
            return [san(v) for v in o]
        if isinstance(o, float) and o != o:
            return None
        return o

    write_json(run / "result.json", san({"trials": rows, "summary": summary,
                                         "verdict": verdict}))
    print(f"\nRESULT {run}")


if __name__ == "__main__":
    main()
