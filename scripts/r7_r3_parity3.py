"""R7 round 3 — the only configuration where the claim is falsifiable.

Round 2 established: cross-tick XOR is unreachable for a linear readout regardless of
hidden tuning. Round 1 established: on SAME-TICK conjunctions the production channel
discards 13-21pp that the frozen neurons already produced.

So the only testbed where "hidden plasticity improved the representation" is a
falsifiable claim is:

    a same-tick conjunction task, read through a channel that preserves detail,
    where frozen-linear is measurably below the trajectory ceiling.

Measured (round 1, spike-sum channel, frozen):
    sim_xor  linear 0.9635  quad 0.9870   (gap 0.023 - too small)
    parity3  linear 0.8320  quad 0.9544   (gap 0.122 - usable)

parity3 is the only task in the suite with a same-tick conjuction AND real headroom for
a linear reader. This runs the full arm comparison there.
"""
import hashlib
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
N_TR, N_TE = 512, 256
ARMS = ("frozen", "production", "secs", "cts", "shuffled", "zero")


def parity3(seed, n):
    """Three bits on ONE tick; label = a XOR b XOR c. Same-tick conjunction."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        bits = [int(rng.integers(2)) for _ in range(3)]
        y[i] = bits[0] ^ bits[1] ^ bits[2]
        t = int(rng.integers(1, 4))
        for k, b in enumerate(bits):
            x[i, t, k] += 1.5 * b
        x[i, 10, 7] += 1.5
    return x, y


def spikesum(core, x):
    out = []
    for i in range(len(x)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(x[i, t], track=False)
            tot += core.spikes
        out.append(tot)
    return np.stack(out)


def traj(core, x):
    out = []
    for i in range(len(x)):
        core.reset_state()
        rows = []
        for t in range(12):
            core.step(x[i, t], track=False)
            rows.append(core.spikes.copy())
        out.append(np.concatenate(rows))
    return np.stack(out)


def quad(F):
    d = F.shape[1]
    idx = [(i, j) for i in range(d) for j in range(i, d)]
    o = np.empty((len(F), len(idx)), np.float64)
    for c, (i, j) in enumerate(idx):
        o[:, c] = F[:, i] * F[:, j]
    return o


def refit(A, ya, B, yb, epochs=800, lr=.5):
    mu = A.mean(0); sd = np.maximum(A.std(0), 1e-6)
    A = (A - mu) / sd; B = (B - mu) / sd
    w = np.zeros((2, A.shape[1])); b = np.zeros(2)
    for _ in range(epochs):
        lg = A @ w.T + b; lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        e = -p; e[np.arange(len(ya)), ya] += 1
        w += lr * (e.T @ A) / len(ya); b += lr * e.mean(0)
    return (float(np.mean(np.argmax(A @ w.T + b, 1) == ya)),
            float(np.mean(np.argmax(B @ w.T + b, 1) == yb)))


def make_core(arm, seed):
    from r7_r2_mechanisms import CTSCore
    c = S.cfg(seed)
    if arm == "cts":
        return CTSCore(c, seed=seed)
    mode = {"production": "scalar", "secs": "secs",
            "shuffled": "shuffled_credit", "zero": "zero_credit"}.get(arm, "scalar")
    return S.SECSCore(c, mode=mode, seed=seed,
                      anchor_refresh=200 if arm in ("secs", "cts") else 0)


def run():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    run = new_run(root, "r7-r3-parity3-v1", {
        "seeds": list(DEV), "arms": list(ARMS), "task": "parity3",
        "channel": "spike_sum",
        "registered_frozen_linear": 0.8320, "registered_quad": 0.9544,
        "registered_headroom": 0.1224, "note": "DEV only"})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            trx, try_ = parity3(seed + 200_000, N_TR)
            tex, tey = parity3(seed + 300_000, N_TE)

            core_f = make_core("frozen", seed)
            A0 = spikesum(core_f, trx); B0 = spikesum(core_f, tex)
            _, fr = refit(A0, try_, B0, tey)
            _, qd = refit(quad(A0), try_, quad(B0), tey)
            _, tj = refit(traj(core_f, trx), try_, traj(core_f, tex), tey)
            print(f"  seed{seed} BASELINE frozen linear {fr:.4f} quad {qd:.4f} "
                  f"trajectory {tj:.4f}", flush=True)

            for arm in ARMS:
                core = make_core(arm, seed)
                w0 = core.w_in.copy()
                order = np.random.default_rng(seed + 91).permutation(N_TR)
                for i in order:
                    core.reset_state()
                    core.forward(trx[i], track=True)
                    core.learn(int(try_[i]), mode="frozen" if arm == "frozen" else "full")
                A = spikesum(core, trx); B = spikesum(core, tex)
                tr_acc, te_acc = refit(A, try_, B, tey)
                rows.append({"seed": seed, "arm": arm, "refit_train": tr_acc,
                             "refit_test": te_acc,
                             "weight_delta": float(np.abs(core.w_in - w0).mean()),
                             "hidden_digest": S.digest(core)[:16]})
                mon.check()
                print(f"  seed{seed} {arm:11s} train {tr_acc:.4f} test {te_acc:.4f} "
                      f"|dw| {np.abs(core.w_in-w0).mean():.5f}", flush=True)

    def M(arm, k):
        return float(np.mean([r[k] for r in rows if r["arm"] == arm]))

    frozen = {r["seed"]: r["refit_test"] for r in rows if r["arm"] == "frozen"}
    gains = {}
    for a in ARMS:
        if a == "frozen":
            continue
        d = [r["refit_test"] - frozen[r["seed"]] for r in rows if r["arm"] == a]
        gains[a] = {"mean": float(np.mean(d)), "per_seed": d,
                    "worst": float(np.min(d))}

    summary = {a: {k: M(a, k) for k in ("refit_train", "refit_test", "weight_delta")}
               for a in ARMS}
    summary["paired_gain_vs_frozen"] = gains

    def san(o):
        if isinstance(o, dict):
            return {k: san(v) for k, v in o.items()}
        if isinstance(o, list):
            return [san(v) for v in o]
        if isinstance(o, float) and o != o:
            return None
        return o

    write_json(run / "result.json",
               san({"trials": rows, "summary": summary,
                    "resources": dict(mon.report()),
                    "wall_seconds": time.time() - t0}))

    print("\n" + "=" * 84)
    print("R7 ROUND 3 — parity3, same-tick conjunction, spike-sum channel")
    print("=" * 84)
    print(f"  {'arm':12s} {'train':>8} {'test':>8} {'vs frozen':>12} {'worst seed':>12}")
    for a in ARMS:
        s = summary[a]
        g = summary["paired_gain_vs_frozen"].get(a)
        gtxt = f"{g['mean']*100:+11.2f}pp" if g else f"{'—':>12}"
        wtxt = f"{g['worst']*100:+11.2f}pp" if g else f"{'—':>12}"
        print(f"  {a:12s} {s['refit_train']:8.4f} {s['refit_test']:8.4f} {gtxt} {wtxt}")
    print(f"\nRESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
