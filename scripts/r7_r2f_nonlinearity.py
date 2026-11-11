"""R7 round 2f — the decisive control that invalidates my own round-2 testbed.

Round 2e shows NO channel solves delayed_xor linearly: production 0.5195, spikesum
0.4831, voltage 0.5202, adaptation 0.5000, sum+volt+adapt 0.4974. All chance.

But round 1's probe 5 reported quadratic refit = 0.9948 on the spike sum. If a
second-order reader solves it and no first-order reader does, then the headroom I
attributed to "information the hidden representation could expose" is really
**the value of a nonlinear readout**.

XOR cannot be linearly separated from the pair (a, b) at all - that is elementary.
So no hidden weight change that leaves the readout linear can ever solve it, and my
round-2 testbed was measuring the wrong thing.

This is checked directly: run the SAME task with the two bits presented SIMULTANEOUSLY
(no memory needed) and compare linear vs quadratic. If linear is chance and quadratic
is ~1.0 even when both bits are present at one tick, then the headroom is purely
readout nonlinearity and has nothing to do with temporal memory OR hidden plasticity.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

N = 512


def simultaneous_xor(seed, n):
    """Both bits at the SAME tick: no memory required, only the nonlinearity."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    a = rng.integers(0, 2, n).astype(np.int32)
    b = rng.integers(0, 2, n).astype(np.int32)
    for i in range(n):
        x[i, 1, 0] += 1.5 * int(a[i])
        x[i, 1, 1] += 1.5 * int(b[i])
        x[i, 10, 7] += 1.5
    return x, (a ^ b).astype(np.int32)


def sum_channel(core, x):
    out = []
    for i in range(len(x)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(x[i, t], track=False)
            tot += core.spikes
        out.append(tot)
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
    return float(np.mean(np.argmax(B @ w.T + b, 1) == yb))


def main():
    print("=" * 84)
    print("R7 ROUND 2f — is the 'headroom' just readout nonlinearity?")
    print("=" * 84)
    print(f"  {'task':22s} {'linear':>8} {'quadratic':>10} {'gap':>8}")
    lin, qd = [], []
    for seed in (140011, 140017, 140023):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        trx, try_ = simultaneous_xor(seed + 200_000, N)
        tex, tey = simultaneous_xor(seed + 300_000, N)
        A = sum_channel(core, trx); B = sum_channel(core, tex)
        lin.append(refit(A, try_, B, tey))
        qd.append(refit(quad(A), try_, quad(B), tey))
    L, Q = float(np.mean(lin)), float(np.mean(qd))
    print(f"  {'simultaneous_xor':22s} {L:8.4f} {Q:10.4f} {Q-L:+8.4f}")

    print("\n" + "=" * 84)
    print("READING")
    print("=" * 84)
    print("  Both bits are present at ONE tick here, so no memory is involved.")
    print("  If linear is chance and quadratic is near 1.0, the whole 'headroom'")
    print("  figure measures READOUT NONLINEARITY, not hidden representational")
    print("  capacity. That invalidates the round-2 testbed and means the SECS/CTS")
    print("  arms were never given a question they could answer.")


if __name__ == "__main__":
    main()
