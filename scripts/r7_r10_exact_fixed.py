"""R7 round 10 — EXACT trace, with the harness defects from round 9 fixed.

ROOT CAUSE OF ROUND 9's FAILURE (found, not guessed)
----------------------------------------------------
`SpikingCore.probabilities()` computes `w_out @ self.features + bias`, i.e. the head
always reads the DECAY-0.25 FEATURE, never `voltage`. Round 9 installed a voltage-trained
head into `core.w_out` and then read `core.probabilities()`. Those two are different
quantities:

    head trained on  v_T          (48-dim voltage)
    core evaluates   w_out @ f_T  (48-dim decay feature)

so every `credit` computed as `(core.w_out * err).sum(0)` was the gradient of a function
the head was never trained on. That is why `credit` was 0.000 in one probe and 3.551 in
another: the probes differed in whether `w_out` still held the random initialisation or
the installed voltage head, and neither was consistent with `self.features`.

THE FIX: never use `core.probabilities()` on the voltage path. Compute the voltage
readout explicitly from `core.voltage`, and take `credit` from that same head.

PRE-CHECKS THAT MUST PASS BEFORE ANY COSINE IS REPORTED (round 9 section 5):
  1  raw-voltage head must reproduce the z-space head's probabilities to < 1e-5
  2  credit norm must be non-zero and match an independent recomputation to < 1e-6
  3  the finite-difference gradient must be reproducible across two calls
  4  only then measure cosine, against a same-episode random null

If any pre-check fails the script raises and prints nothing interpretable.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.neural.core import Config, SpikingCore  # noqa: E402

SEED = 140011
CHECK_TOL_PROB = 1e-5
CHECK_TOL_CREDIT = 1e-6


def cfg(seed):
    return Config(seed=seed, inputs=8, hidden=48, outputs=2, fan_in=8,
                  plastic_rows=12, leak=.65, threshold=.5,
                  feature_decay=.25, adaptation_strength=.02)


def task(seed, n):
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


class ExactCore(SpikingCore):
    """Forward pass inherited unchanged; adds a per-synapse leaky input trace.

    e_in[i,j] <- leak*e_in[i,j] + x_t[j]      exact for dv/dw_in, direct path
    """

    def __init__(self, config):
        self.e_in = np.zeros((config.plastic_rows, config.inputs), np.float32)
        self.e_rec = np.zeros((config.plastic_rows, config.fan_in), np.float32)
        super().__init__(config)
        self.e_in = np.zeros_like(self.w_in[self.rows])
        self.e_rec = np.zeros_like(self.w_rec[self.rows])

    def reset_state(self):
        super().reset_state()
        if hasattr(self, "e_in"):
            self.e_in[:] = 0.0
            self.e_rec[:] = 0.0

    def step(self, x, track=True):
        c = self.config
        prev_rec = (self.pre[c.inputs:][self.indices[self.rows]].astype(np.float32).copy()
                    if track else None)
        p = super().step(x, track=False)
        if track:
            xa = np.asarray(x, np.float32)
            self.e_in = c.leak * self.e_in + xa[None, :]
            self.e_rec = c.leak * self.e_rec + prev_rec
        return p


def voltage_of(core, seq):
    core.reset_state()
    for t in range(len(seq)):
        core.step(seq[t], track=False)
    return core.voltage.copy()


def voltage_batch(core, X):
    return np.stack([voltage_of(core, X[i]) for i in range(len(X))])


def softmax(z):
    """Row-wise softmax. Accepts a 1-D vector too, returning a 1-D vector."""
    z = np.asarray(z, float)
    one_d = z.ndim == 1
    if one_d:
        z = z[None, :]
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    out = e / e.sum(1, keepdims=True)
    return out[0] if one_d else out


def train_head(V, y, epochs=1200, lr=.5):
    """Head in z-space; returns the z-space params plus the scaler."""
    mu = V.mean(0)
    sd = np.maximum(V.std(0), 1e-6)
    A = (V - mu) / sd
    w = np.zeros((2, A.shape[1]))
    b = np.zeros(2)
    for _ in range(epochs):
        p = softmax(A @ w.T + b)
        e = -p
        e[np.arange(len(y)), y] += 1
        w += lr * (e.T @ A) / len(y)
        b += lr * e.mean(0)
    return mu, sd, w, b


def raw_head(mu, sd, w, b):
    """Convert a z-space head to an equivalent RAW-voltage head.

    z = (v - mu)/sd  =>  w·z + b = (w/sd)·v + (b - w·mu/sd)
    So w_raw = w/sd and b_raw = b - (w * mu/sd).sum(1). Keeping b_raw is what round 9
    got wrong (it tried to fold the offset into a symmetric bias).
    """
    w_raw = w / sd[None, :]
    b_raw = b - (w * (mu / sd)[None, :]).sum(1)
    return w_raw, b_raw


def main():
    print("=" * 92)
    print("R7 ROUND 10 — EXACT trace, harness fixed, pre-checks first")
    print("=" * 92)

    c = cfg(SEED)
    core = ExactCore(c)
    trx, try_ = task(SEED + 200_000, 512)
    tex, tey = task(SEED + 300_000, 256)

    V = voltage_batch(core, trx)
    Vt = voltage_batch(core, tex)
    mu, sd, w, b = train_head(V, try_)
    acc = float(np.mean(np.argmax(((Vt - mu) / sd) @ w.T + b, 1) == tey))
    print(f"\n  voltage head held-out accuracy: {acc:.4f}")

    w_raw, b_raw = raw_head(mu, sd, w, b)

    # ---------- PRE-CHECK 1: raw head reproduces z-space probabilities
    p_z = softmax(((V - mu) / sd) @ w.T + b)
    p_raw = softmax(V @ w_raw.T + b_raw)
    d1 = float(np.abs(p_z - p_raw).max())
    print(f"  [check 1] raw-vs-z probability max |diff| = {d1:.3e} "
          f"(tol {CHECK_TOL_PROB:.0e})  {'PASS' if d1 < CHECK_TOL_PROB else 'FAIL'}")
    if d1 >= CHECK_TOL_PROB:
        raise SystemExit("pre-check 1 failed: raw head does not reproduce the loss")

    # ---------- PRE-CHECK 2: credit is non-zero and reproducible independently
    i0 = 0
    v0 = voltage_of(core, trx[i0])
    err0 = -softmax(v0 @ w_raw.T + b_raw)
    err0[int(try_[i0])] += 1.0
    credit_a = (w_raw * err0[:, None]).sum(0)
    credit_b = np.array([float(w_raw[:, i] @ err0) for i in range(w_raw.shape[1])])
    d2 = float(np.abs(credit_a - credit_b).max())
    n2 = float(np.linalg.norm(credit_a))
    print(f"  [check 2] credit norm = {n2:.4f}, two independent forms max |diff| "
          f"= {d2:.3e}  {'PASS' if n2 > 1e-6 and d2 < CHECK_TOL_CREDIT else 'FAIL'}")
    if n2 <= 1e-6 or d2 >= CHECK_TOL_CREDIT:
        raise SystemExit("pre-check 2 failed: credit is zero or irreproducible")

    # ---------- PRE-CHECK 3: finite-difference gradient reproducible
    eps = 1e-2

    def loss_of(Vb, yb):
        return -float(np.mean(np.log(np.maximum(
            softmax(Vb @ w_raw.T + b_raw)[np.arange(len(yb)), yb], 1e-12))))

    def grad_at(i, row_i, j):
        saved = core.w_in.copy()
        base = loss_of(voltage_batch(core, trx[i:i + 1]), try_[i:i + 1])
        core.w_in[core.rows[row_i], j] += eps
        up = loss_of(voltage_batch(core, trx[i:i + 1]), try_[i:i + 1])
        core.w_in[:] = saved
        return (up - base) / eps

    g1 = grad_at(0, 0, 0)
    g2 = grad_at(0, 0, 0)
    print(f"  [check 3] FD gradient reproducible: {g1:+.6e} vs {g2:+.6e}  "
          f"{'PASS' if abs(g1 - g2) < 1e-12 else 'FAIL'}")
    if abs(g1 - g2) >= 1e-12:
        raise SystemExit("pre-check 3 failed: gradient not reproducible")

    # ---------- the measurement
    print("\n  all pre-checks passed; measuring cosine")
    rng = np.random.default_rng(11)
    cos_exact, cos_prod, cos_rand = [], [], []
    mag_exact, mag_true = [], []

    for i in range(20):
        core.reset_state()
        for t in range(12):
            core.step(trx[i, t], track=True)
        vi = core.voltage.copy()
        # SIGN DERIVATION, verified against finite differences:
        #   L = -log p_y,  dL/dz = p - onehot,  dL/dv_i = sum_k w[k,i] dL/dz_k
        #   T_exact[i,j] = dL/dv_i * e_in[i,j]  =  dL/dw_in[i,j]   (direct path)
        # Using err = onehot - p instead of dL/dz flips the sign and yields cosine -1.
        erri = softmax(vi @ w_raw.T + b_raw)
        erri[int(try_[i])] -= 1.0            # erri == dL/dz
        credit = (w_raw * erri[:, None]).sum(0)   # == dL/dv, NO extra minus

        T_exact = credit[core.rows][:, None] * core.e_in + \
            credit[core.rows][:, None] * core.e_rec
        T_prod = core.elig_in.copy()

        base = loss_of(voltage_batch(core, trx[i:i + 1]), try_[i:i + 1])
        grad = np.zeros_like(T_exact)
        saved = core.w_in.copy()
        for r_i, row in enumerate(core.rows):
            for j in range(c.inputs):
                core.w_in[row, j] += eps
                grad[r_i, j] = (loss_of(voltage_batch(core, trx[i:i + 1]),
                                        try_[i:i + 1]) - base) / eps
                core.w_in[:] = saved

        g = grad.ravel()
        for T, store in ((T_exact, cos_exact), (T_prod, cos_prod)):
            a = T.ravel()
            den = np.linalg.norm(a) * np.linalg.norm(g)
            if den > 1e-12:
                store.append(float(a @ g / den))
        r = rng.normal(0, 1, g.shape)
        den = np.linalg.norm(r) * np.linalg.norm(g)
        if den > 1e-12:
            cos_rand.append(float(r @ g / den))
        mag_exact.append(float(np.linalg.norm(T_exact)))
        mag_true.append(float(np.linalg.norm(g)))

    def st(v):
        v = np.asarray(v)
        return f"{v.mean():+.4f}  (sd {v.std():.3f}, n={len(v)})"

    print(f"\n  {'trace':14s} {'cosine with true dL/dw':>40s}")
    print(f"  {'EXACT':14s} {st(cos_exact):>40s}")
    print(f"  {'production':14s} {st(cos_prod):>40s}")
    print(f"  {'random null':14s} {st(cos_rand):>40s}")
    print(f"\n  |EXACT| mean {np.mean(mag_exact):.4e}   "
          f"|true grad| mean {np.mean(mag_true):.4e}")

    e, p, r = np.mean(cos_exact), np.mean(cos_prod), np.mean(cos_rand)
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"  EXACT {e:+.4f} | production {p:+.4f} | null {r:+.4f}")
    if e > 0.30 and e > abs(r) + 0.20:
        print("  => EXACT trace IS aligned with the gradient where the Hebbian")
        print("     correlation is not. The derivation holds and the production")
        print("     trace was simply not a derivative.")
    elif abs(e) < 0.10:
        print("  => EXACT trace is still near null. The derivation is right about")
        print("     the direct path but the recurrent term dominates at this horizon,")
        print("     or the trace decays away before the readout tick.")
    else:
        print("  => intermediate: aligned but not decisively so. Report as partial.")


if __name__ == "__main__":
    main()
