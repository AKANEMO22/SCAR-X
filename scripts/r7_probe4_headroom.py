"""R7 probe 4 — does this benchmark have ANY headroom for hidden learning?

The SECS DEV run produced a result that invalidates the whole arm comparison:
`frozen` + common refit reached **0.8268**, essentially the ceiling. If frozen already
solves the task, no hidden-learning mechanism can show a representation gain, and the
experiment cannot answer the question it was built for. Brief question Q7 applies:
"is the problem the learning algorithm, or has the test already been nearly solved by
frozen?"

This probe measures headroom directly on several task families, using the SAME frozen
population and the SAME common refit. A task is only useful as a hidden-learning
testbed if `frozen << ceiling`, where the ceiling is what a much more expressive
readout (or an oracle over the same features) can reach.

Ceiling proxy: polynomial-order-2 expansion of the same features, read by the same
common head. This is a READOUT ceiling, not an information ceiling, and it is used only
to decide whether a task has headroom.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402
from scarx.evaluation.tasks21 import ridge_probe  # noqa: E402

N_TR, N_TE = 512, 256


def sim_xor(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.int32)
    for i in range(n):
        a = int(rng.integers(2)); b = a ^ int(y[i]); t = int(rng.integers(1, 4))
        x[i, t, 0] += 1.5 * a; x[i, t, 1] += 1.5 * b; x[i, 10, 7] += 1.5
    return x, y


def order_task(seed, n):
    """Label = which of two events came first. Requires temporal order."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.int32)
    for i in range(n):
        first, second = (0, 1) if y[i] == 0 else (1, 0)
        p = int(rng.integers(1, 3)); q = int(rng.integers(5, 8))
        x[i, p, first] += 1.5; x[i, q, second] += 1.5; x[i, 10, 7] += 1.5
    return x, y


def parity3(seed, n):
    """Three bits on one tick, label = a XOR b XOR c. Order-3 in the raw tape."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.int32)
    for i in range(n):
        bits = [int(rng.integers(2)) for _ in range(3)]
        yy = bits[0] ^ bits[1] ^ bits[2]
        y[i] = yy
        t = int(rng.integers(1, 4))
        for k, b in enumerate(bits):
            x[i, t, k] += 1.5 * b
        x[i, 10, 7] += 1.5
    return x, y


def quad_feats(F):
    """Second-order expansion: the readout ceiling proxy."""
    d = F.shape[1]
    idx = [(i, j) for i in range(d) for j in range(i, d)]
    out = np.empty((len(F), len(idx)), np.float64)
    for c, (i, j) in enumerate(idx):
        out[:, c] = F[:, i] * F[:, j]
    return out


def feats(core, x):
    out = []
    for i in range(len(x)):
        core.reset_state()
        core.forward(x[i], track=False)
        out.append(core.features.copy())
    return np.stack(out)


def refit(A, ya, B, yb, epochs=600, lr=.5):
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


def main():
    print("=" * 90)
    print("R7 PROBE 4 — which task families have HEADROOM for hidden learning?")
    print("=" * 90)
    print(f"  {'task':14s} {'linear':>8} {'quad':>8} {'headroom':>9} {'verdict':>16}")
    tasks = {"sim_xor": sim_xor, "order": order_task, "parity3": parity3}
    for name, gen in tasks.items():
        lin, quad = [], []
        for seed in (140011, 140017, 140023):
            c = S.cfg(seed)
            core = S.SECSCore(c, mode="scalar", seed=seed)
            trx, try_ = gen(seed + 200_000, N_TR)
            tex, tey = gen(seed + 300_000, N_TE)
            F = feats(core, trx); T = feats(core, tex)
            _, te_l = refit(F, try_, T, tey)
            _, te_q = refit(quad_feats(F), try_, quad_feats(T), tey)
            lin.append(te_l); quad.append(te_q)
        L, Q = float(np.mean(lin)), float(np.mean(quad))
        print(f"  {name:14s} {L:8.4f} {Q:8.4f} {Q-L:+9.4f} "
              f"{('USEFUL' if Q - L > .10 else 'NO HEADROOM'):>16}")

    print("\n" + "=" * 90)
    print("READING")
    print("=" * 90)
    print("  'headroom' = quad-refit minus linear-refit on the SAME frozen features.")
    print("  A task is a valid hidden-learning testbed only if headroom is large:")
    print("  otherwise frozen already answers it and no representation gain can show.")


if __name__ == "__main__":
    main()
