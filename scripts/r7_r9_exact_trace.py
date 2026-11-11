"""R7 round 9 — EXACT: an eligibility trace that is a true local derivative.

THE DERIVATION (not a guess; it follows from the forward pass)
--------------------------------------------------------------
With a VOLTAGE readout the readout sees `v_T`, and

    v_t = leak·v_{t-1} + w_in @ x_t + rec_t
 => dv_t/dw_in[i,j] = Σ_{s≤t} leak^{t-s} · x_s[j]

which is a **leaky trace of the input**, one scalar per synapse:

    e_in[i,j]  <-  leak · e_in[i,j] + x_t[j]

and the per-neuron credit is available at the readout itself:

    c_i = W_out[:, i] · error

so the exact readout gradient through the voltage is

    dL/dw_in[i,j]  =  c_i · e_in[i,j]        (+ recurrent terms, dropped)

This is EXACT for the direct path and *approximate* only in that it ignores how
changing v_i propagates through the recurrent weights into later ticks. That residual
is stated as a limitation, not hidden.

WHY THIS IS DIFFERENT FROM EVERYTHING ALREADY REJECTED
-------------------------------------------------------
                  production trace        EXACT trace
  quantity        spike·pre − .5·post·x   leaky sum of input
  is it d(v)/dw?  only at threshold       exactly, for all t
  readout channel gives gradient          only if soft           any
  round-8 cosine  +0.059 (null +0.067)    to be measured here

The prediction is quantitative and falsifiable: if the derivation is right,
`cosine(EXACT trace, dL/dw)` must be substantially above the random null, where the
production trace was indistinguishable from it.

Cost: one float per plastic synapse (96 floats) and one add-multiply per tick.
No BPTT, no future ticks, streaming, checkpointable.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.neural.core import Config, SpikingCore  # noqa: E402

SEED = 140011
N_TR, N_TE = 512, 256


def cfg(seed):
    c = SpikingCore
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


class ExactTraceCore(SpikingCore):
    """Forward pass inherited verbatim; adds a per-synapse input trace."""

    def __init__(self, config):
        # allocate BEFORE super().__init__, because it calls reset_state()
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
        # capture the PREVIOUS presynaptic recurrent activity BEFORE super().step
        # overwrites self.pre
        prev_rec = self.pre[c.inputs:][self.indices[self.rows]].copy() \
            if track else None
        p = super().step(x, track=False)
        if track:
            x = np.asarray(x, np.float32)
            # e_in[i,j] <- leak*e_in[i,j] + x_j   (exact dv/dw_in, direct path)
            self.e_in = c.leak * self.e_in + x[None, :]
            # recurrent analogue: trace of the previous presynaptic activity
            self.e_rec = c.leak * self.e_rec + prev_rec
        return p


def channel_voltage(core, x):
    out = []
    for i in range(len(x)):
        core.reset_state()
        for t in range(12):
            core.step(x[i, t], track=False)
        out.append(core.voltage.copy())
    return np.stack(out)


def train_linear(F, y, epochs=1200, lr=.5):
    mu = F.mean(0); sd = np.maximum(F.std(0), 1e-6)
    A = (F - mu) / sd
    w = np.zeros((2, F.shape[1])); b = np.zeros(2)
    for _ in range(epochs):
        lg = A @ w.T + b; lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        e = -p; e[np.arange(len(y)), y] += 1
        w += lr * (e.T @ A) / len(y); b += lr * e.mean(0)
    return mu, sd, w, b


def main():
    print("=" * 88)
    print("R7 ROUND 9 — EXACT trace: is it finally aligned with dL/dw?")
    print("=" * 88)
    c = cfg(SEED)
    core = ExactTraceCore(c)
    trx, try_ = task(SEED + 200_000, N_TR)
    tex, tey = task(SEED + 300_000, N_TE)

    # train the readout on frozen voltage so dL/dw is the real objective
    F = channel_voltage(core, trx)
    Ft = channel_voltage(core, tex)
    mu, sd, w, b = train_linear(F, try_)
    acc = float(np.mean(np.argmax(((Ft - mu) / sd) @ w.T + b, 1) == tey))
    print(f"\n  readout trained on voltage: held-out {acc:.4f}")

    # Install an equivalent head in RAW voltage units so `credit` is the gradient of
    # the SAME loss the finite differences measure.
    #
    # The trained head operates on z = (v - mu)/sd. The loss is
    #     L = CE(w · z + b)  with  z = (v - mu)/sd
    # so in raw-voltage units the equivalent weight is  w_raw = w / sd  and the
    # constant offset (w·mu/sd + b) drops out of the softmax difference. Without this
    # conversion the credit vector is the gradient of a DIFFERENT function and every
    # dot product is meaningless - which is exactly what produced credit = 0.
    w_raw = (w / sd[None, :]).astype(np.float32)
    core.w_out = w_raw.copy()
    core.bias = np.zeros(2, np.float32)
    # subtract the offset so probabilities are unchanged
    offset = float((w * (mu / sd)[None, :]).sum(1)[0])
    core.bias[0] = -offset
    core.bias[1] = offset

    # verify the raw head reproduces the same probabilities
    core.reset_state()
    for t in range(12):
        core.step(trx[0, t], track=False)
    p_raw = core.probabilities()
    p_z = np.exp(((F[0] - mu) / sd) @ w.T + b)
    p_z = p_z / p_z.sum()
    print(f"  raw-vs-z probability check: max |diff| = "
          f"{float(np.abs(p_raw - p_z).max()):.3e}")

    def loss_at(F1, yy):
        A = (F1 - mu) / sd
        lg = A @ w.T + b
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg); p = p / p.sum(1, keepdims=True)
        return -float(np.mean(np.log(np.maximum(p[np.arange(len(yy)), yy], 1e-12))))

    eps = 1e-2
    cfg_inputs = c.inputs
    rng = np.random.default_rng(11)
    cos_exact, cos_prod, cos_rand = [], [], []
    mags_exact, mags_true = [], []

    for i in range(20):
        # EXACT trace through one episode
        core.reset_state()
        for t in range(12):
            core.step(trx[i, t], track=True)
        err = -np.eye(2)[try_[i]]
        err[try_[i]] += 1.0
        credit = (core.w_out * err[:, None]).sum(0)          # c_i = W_out[:,i] . error
        T_exact = credit[core.rows][:, None] * core.e_in       # c_i * e_in
        T_exact_rec = credit[core.rows][:, None] * core.e_rec

        # PRODUCTION trace, for comparison on the same episode
        T_prod = core.elig_in.copy()

        # true finite-difference gradient of the trained loss
        saved = core.w_in.copy()
        base = loss_at(channel_voltage(core, trx[i:i + 1]), try_[i:i + 1])
        grad = np.zeros_like(T_exact)
        for r_i, row in enumerate(core.rows):
            for j in range(cfg_inputs):
                core.w_in[row, j] += eps
                lp = loss_at(channel_voltage(core, trx[i:i + 1]), try_[i:i + 1])
                core.w_in[:] = saved
                grad[r_i, j] = (lp - base) / eps

        g = grad.ravel()
        for name, T, store in (("exact", T_exact + T_exact_rec, cos_exact),
                               ("prod", T_prod, cos_prod)):
            a = T.ravel()
            den = np.linalg.norm(a) * np.linalg.norm(g)
            if den > 1e-12:
                store.append(float(a @ g / den))
        r = rng.normal(0, 1, g.shape)
        den = np.linalg.norm(r) * np.linalg.norm(g)
        if den > 1e-12:
            cos_rand.append(float(r @ g / den))
        mags_exact.append(float(np.linalg.norm(T_exact + T_exact_rec)))
        mags_true.append(float(np.linalg.norm(g)))

    def st(v):
        v = np.array(v)
        return f"{v.mean():+.4f}  (sd {v.std():.3f}, n={len(v)})"

    print(f"\n  cosine with the TRUE dL/dw over 20 episodes:")
    print(f"    EXACT trace    {st(cos_exact)}")
    print(f"    production     {st(cos_prod)}")
    print(f"    random null    {st(cos_rand)}")
    print(f"\n  |EXACT trace| mean {np.mean(mags_exact):.4e}   "
          f"|true grad| mean {np.mean(mags_true):.4e}")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    e, p, r = np.mean(cos_exact), np.mean(cos_prod), np.mean(cos_rand)
    if e > max(abs(r), abs(p)) * 3:
        print(f"  EXACT cosine {e:+.4f} is far above the null {r:+.4f} and the")
        print(f"  production trace {p:+.4f}. The derivation holds: a trace that is")
        print(f"  an actual derivative IS aligned with the gradient, where the")
        print(f"  Hebbian correlation was not.")
        print(f"\n  => credit assignment was never impossible; the production trace")
        print(f"     simply was not a derivative.")
    else:
        print(f"  EXACT {e:+.4f} vs null {r:+.4f} vs production {p:+.4f}.")
        print(f"  NOT a clear separation - the derivation does not survive contact.")


if __name__ == "__main__":
    main()
