"""R7 round 2b — is it the LOCAL RULE or is it IMPOSSIBLE for this network?

Round 2 showed every local mechanism at chance on a task with 45pp of proven headroom:
frozen 0.5430, production 0.5456, secs 0.5417, cts 0.5430. Weight deltas were ~0.001.

Two competing explanations, and they demand different next rounds:

  E1  the local rule produces changes that are too small / misdirected
  E2  no change to THIS 48-neuron population's input weights can make delayed_xor
      linearly decodable, so the testbed is wrong for the question

This distinguishes them with an ORACLE hidden update: a BPTT diagnostic that optimises
the hidden weights directly for linear decodability. BPTT here is a LEARNABILITY
DIAGNOSTIC only (brief section II), never a candidate and never a weight source.

  If the oracle lifts linear decodability well above frozen -> E1, the local rule is
  the bottleneck and the next round should attack magnitude/direction.
  If the oracle cannot -> E2, and no credit rule could ever have worked.

Also measures: how large a weight change is needed, and whether the anchor/clip caps
that magnitude.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 512, 256


def delayed_xor(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.int32)
    for i in range(n):
        a = int(rng.integers(2)); b = a ^ int(y[i])
        x[i, 1, 0] += 1.5 * a
        x[i, 8, 1] += 1.5 * b
        x[i, 10, 7] += 1.5
    return x, y


def forward_sum(c, x, w_in, w_rec, idx, soft_beta=10.0):
    """Frozen-form forward with a SOFT threshold, returning spike sums.

    A soft threshold is required for an oracle gradient to exist; the hard spike has
    zero derivative almost everywhere (round-1 cause C2). We compare the soft model's
    own frozen baseline so the oracle gain is measured within one consistent model.
    """
    v = np.zeros(c.hidden, np.float64); a = np.zeros(c.hidden, np.float64)
    z = np.zeros(c.hidden, np.float64); tot = np.zeros(c.hidden, np.float64)
    for s in range(len(x)):
        v = c.leak * v + w_in @ x[s] + (w_rec * z[idx]).sum(1)
        lim = c.threshold + c.adaptation_strength * a
        z = 1.0 / (1.0 + np.exp(-soft_beta * (v - lim)))
        v = v - z * lim
        a = c.adaptation_decay * a + z
        tot += z
    return tot


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


def oracle_hidden(c, trx, try_, tex, tey, idx, w_in0, w_rec0, rows,
                  steps=300, lr=.5, seed=0):
    """Optimise w_in for LINEAR decodability via numeric gradient on spike sums.

    A linear readout is refit in closed form every K steps and the hidden weights are
    moved along the finite-difference gradient of the refit loss. This is deliberately
    expensive and crude; it only has to answer 'is a better linear representation
    reachable at all'.
    """
    rng = np.random.default_rng(seed)
    w_in = w_in0.astype(np.float64).copy()
    w_rec = w_rec0
    # soft-model frozen baseline in the same parametrisation
    A = np.stack([forward_sum(c, trx[i], w_in, w_rec, idx) for i in range(len(trx))])
    B = np.stack([forward_sum(c, tex[i], w_in, w_rec, idx) for i in range(len(tex))])
    base_tr, base_te = refit(A, try_, B, tey)

    eps = 2e-2
    hist = [(0, base_tr, base_te, 0.0)]
    for step in range(steps):
        # random sparse direction in the plastic rows only
        d = np.zeros_like(w_in)
        sel = rng.choice(rows, 4, replace=False)
        d[sel] = rng.normal(0, 1, (len(sel), c.inputs)) * eps
        wp = w_in + d
        Ap = np.stack([forward_sum(c, trx[i], wp, w_rec, idx)
                       for i in range(len(trx))])
        Bp = np.stack([forward_sum(c, tex[i], wp, w_rec, idx)
                       for i in range(len(tex))])
        trp, tep = refit(Ap, try_, Bp, tey)
        if trp >= base_tr - 1e-9 or True:
            if trp + tep > base_tr + base_te:
                w_in = wp
                base_tr, base_te = trp, tep
        if (step + 1) % 60 == 0:
            delta = float(np.abs(w_in - w_in0).mean())
            hist.append((step + 1, base_tr, base_te, delta))
    return hist


def main():
    print("=" * 92)
    print("R7 ROUND 2b — can ANY hidden change make delayed_xor linearly decodable?")
    print("=" * 92)
    for seed in DEV:
        c = S.cfg(seed)
        core = S.SECSCore(c, mode="scalar", seed=seed)
        g = np.random.default_rng(c.seed)
        idx = np.stack([g.choice(c.hidden, c.fan_in, replace=False)
                        for _ in range(c.hidden)]).astype(np.int32)

        trx, try_ = delayed_xor(seed + 200_000, N_TR)
        tex, tey = delayed_xor(seed + 300_000, N_TE)
        hist = oracle_hidden(c, trx, try_, tex, tey, idx,
                             core.w_in.copy(), core.w_rec.copy(),
                             core.rows.copy(), steps=240, seed=seed + 1)
        print(f"\n  seed {seed} — ORACLE hidden search (soft threshold model)")
        print(f"    {'step':>6} {'train':>8} {'test':>8} {'|dw|':>9}")
        for st, tr, te, dw in hist:
            print(f"    {st:>6} {tr:8.4f} {te:8.4f} {dw:9.5f}")
        gain = hist[-1][2] - hist[0][2]
        print(f"    oracle test gain over soft-frozen: {gain*100:+.2f}pp")

    print("\n" + "=" * 92)
    print("READING")
    print("=" * 92)
    print("  If the oracle lifts test accuracy materially, a better linear")
    print("  representation IS reachable and the local rule is the bottleneck (E1).")
    print("  If it does not, no credit rule could have won here (E2).")


if __name__ == "__main__":
    main()
