"""R7 probe 5 — the readout observes a LOSSY summary. Is that the real bottleneck?

Probe 4 showed no task has headroom on the production readout features (decay-0.25).
But earlier rounds (R2A, R6A) established that the SAME frozen population carries far
more information in its raw spike counts than in the decayed feature vector.

If that is true, hidden learning is being asked to do something impossible: change
hidden weights so that a *decayed marginal summary* becomes informative, when the
information sits in the spikes that the decay throws away.

This measures the information in each readout channel from the SAME frozen population:

    features   = decay-0.25 (what the production readout actually sees)
    spikesum   = total spike count over the window
    trajectory = full spike train

If spikesum >> features, the bottleneck is the READOUT CHANNEL, not the hidden weights,
and no amount of hidden plasticity can fix it — because plasticity would have to make
the decayed sum contain information the decay destroys.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

N_TR, N_TE = 512, 256


def tasks():
    def sim_xor(seed, n):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
        y = rng.integers(0, 2, n).astype(np.int32)
        for i in range(n):
            a = int(rng.integers(2)); b = a ^ int(y[i]); t = int(rng.integers(1, 4))
            x[i, t, 0] += 1.5 * a; x[i, t, 1] += 1.5 * b; x[i, 10, 7] += 1.5
        return x, y

    def order_task(seed, n):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
        y = rng.integers(0, 2, n).astype(np.int32)
        for i in range(n):
            first, second = (0, 1) if y[i] == 0 else (1, 0)
            p = int(rng.integers(1, 3)); q = int(rng.integers(5, 8))
            x[i, p, first] += 1.5; x[i, q, second] += 1.5; x[i, 10, 7] += 1.5
        return x, y

    def parity3(seed, n):
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

    def delayed_xor(seed, n):
        """Two bits separated by several silent ticks: needs a memory trace."""
        rng = np.random.default_rng(seed)
        x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
        y = rng.integers(0, 2, n).astype(np.int32)
        for i in range(n):
            a = int(rng.integers(2)); b = a ^ int(y[i])
            x[i, 1, 0] += 1.5 * a
            x[i, 8, 1] += 1.5 * b
            x[i, 10, 7] += 1.5
        return x, y

    return {"sim_xor": sim_xor, "order": order_task,
            "parity3": parity3, "delayed_xor": delayed_xor}


def channels(core, x):
    """Every readout channel from ONE pass of the same frozen population."""
    feats, sums, traj = [], [], []
    for i in range(len(x)):
        core.reset_state()
        spk = np.zeros((12, core.config.hidden), np.float32)
        for t in range(12):
            core.step(x[i, t], track=False)
            spk[t] = core.spikes
        feats.append(core.features.copy())
        sums.append(spk.sum(0))
        traj.append(spk.ravel())
    return np.stack(feats), np.stack(sums), np.stack(traj)


def refit(A, ya, B, yb, epochs=600, lr=.5):
    mu = A.mean(0); sd = np.maximum(A.std(0), 1e-6)
    A = (A - mu) / sd; B = (B - mu) / sd
    w = np.zeros((2, A.shape[1])); b = np.zeros(2)
    for _ in range(epochs):
        lg = A @ w.T + b; lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        e = -p; e[np.arange(len(ya)), ya] += 1
        w += lr * (e.T @ A) / len(ya); b += lr * e.mean(0)
    return float(np.mean(np.argmax(B @ w.T + b, 1) == yb))


def main():
    print("=" * 92)
    print("R7 PROBE 5 — information per readout channel (same frozen population)")
    print("=" * 92)
    print(f"  {'task':13s} {'features':>9} {'spikesum':>9} {'trajectory':>11} "
          f"{'ss-feat':>9} {'verdict':>22}")
    for name, gen in tasks().items():
        f, s, tr = [], [], []
        for seed in (140011, 140017, 140023):
            c = S.cfg(seed)
            core = S.SECSCore(c, mode="scalar", seed=seed)
            a, ya = gen(seed + 200_000, N_TR)
            b, yb = gen(seed + 300_000, N_TE)
            Fa, Sa, Ta = channels(core, a)
            Fb, Sb, Tb = channels(core, b)
            f.append(refit(Fa, ya, Fb, yb))
            s.append(refit(Sa, ya, Sb, yb))
            tr.append(refit(Ta, ya, Tb, yb))
        F, Sp, Tr = map(lambda v: float(np.mean(v)), (f, s, tr))
        gap = Sp - F
        verdict = "CHANNEL LOSS" if gap > .10 else "channel ok"
        print(f"  {name:13s} {F:9.4f} {Sp:9.4f} {Tr:11.4f} {gap:+9.4f} {verdict:>22}")

    print("\n" + "=" * 92)
    print("READING")
    print("=" * 92)
    print("  'features' is decay-0.25, which is what the production readout sees.")
    print("  'spikesum' is the raw count over the same window.")
    print("  If spikesum >> features, the readout channel discards information that")
    print("  the population already produced. Hidden plasticity would then have to")
    print("  make a decayed marginal summary carry a conjunction the decay removes.")


if __name__ == "__main__":
    main()
