"""R7 probe 3 — the decisive measurement: is the eligibility trace aligned with the
true gradient AFTER the readout is trained?

Probe 2 measured cosine = -0.038 on ONE episode with an UNTRAINED random readout. That
is weak evidence on both counts: one sample, and a random w_out makes dL/dw an
arbitrary projection.

This probe fixes both:
  * the readout is trained to convergence first, so dL/dw reflects the real objective
  * the alignment is measured over many episodes with a proper confidence interval

If the trace remains orthogonal to the gradient after the readout is competent, then
the scalar-advantage modulation is hopeless by construction: it multiplies a trace that
points nowhere. That would localise the bottleneck to the TRACE, not the reward scalar,
not the threshold.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.neural.core import Config, SpikingCore  # noqa: E402


def cfg(seed):
    return Config(seed=seed, inputs=8, hidden=48, outputs=2, fan_in=8,
                  plastic_rows=12, leak=.65, threshold=.5,
                  feature_decay=.25, adaptation_strength=.02)


def episode(seed, n=64):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.int32)
    for i in range(n):
        a = int(rng.integers(2)); b = a ^ int(y[i])
        t = int(rng.integers(1, 4))
        x[i, t, 0] += 1.5 * a
        x[i, t, 1] += 1.5 * b
        x[i, 10, 7] += 1.5
    return x, y


def rollout(c, x, w_in, w_rec, idx, hard=True, beta=8.0):
    H = c.hidden
    v = np.zeros(H, np.float32); a = v.copy(); z = v.copy()
    f = np.zeros(H, np.float32)
    for s in range(len(x)):
        v = c.leak * v + w_in @ x[s] + np.sum(w_rec * z[idx], axis=1)
        lim = c.threshold + c.adaptation_strength * a
        z = (v >= lim).astype(np.float32) if hard else \
            1.0 / (1.0 + np.exp(-beta * (v - lim)))
        v = v - z * lim
        a = c.adaptation_decay * a + z
        f = c.feature_decay * f + (1 - c.feature_decay) * z
    return f


def main():
    print("=" * 84)
    print("R7 PROBE 3 — trace vs true gradient, with a TRAINED readout")
    print("=" * 84)
    seed = 140011
    c = cfg(seed)
    core = SpikingCore(c)
    x, y = episode(seed)
    rows = core.rows.copy()
    g = np.random.default_rng(c.seed)
    IDX = np.stack([g.choice(c.hidden, c.fan_in, replace=False)
                    for _ in range(c.hidden)]).astype(np.int32)
    w_in = core.w_in.copy(); w_rec = core.w_rec.copy()

    # ---- train the READOUT to convergence on frozen hidden (frozen features)
    F = np.stack([rollout(c, x[i], w_in, w_rec, IDX, hard=True) for i in range(64)])
    w_out = np.zeros((2, c.hidden), np.float32); b_out = np.zeros(2, np.float32)
    for ep in range(400):
        lg = F @ w_out.T + b_out
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg); p = p / p.sum(1, keepdims=True)
        err = -p; err[np.arange(len(y)), y] += 1
        w_out += .5 * (err.T @ F) / len(y)
        b_out += .5 * err.mean(0)
    lg = F @ w_out.T + b_out
    pred = np.argmax(lg, 1)
    print(f"\n  trained readout on frozen features: train acc = "
          f"{np.mean(pred == y):.4f}  (this is the ceiling for any hidden change)")

    def loss_of(f, t):
        lg = w_out @ f + b_out
        lg = lg - lg.max()
        p = np.exp(lg); p = p / p.sum()
        return -float(np.log(max(float(p[t]), 1e-12)))

    # ---- alignment over many episodes, hard vs soft
    eps = 1e-2
    print(f"\n  cosine(trace, dL/dw) over episodes, readout TRAINED")
    print(f"  {'mode':6s} {'n':>4} {'mean cos':>10} {'95% CI':>20} {'frac|cos|>0.2':>14}")
    results = {}
    for mode, hard in (("hard", True), ("soft", False)):
        coss = []
        for i in range(40):
            core.reset_state()
            core.forward(x[i], track=True)
            tr = np.zeros_like(w_in)
            tr[rows] = core.elig_in

            f0 = rollout(c, x[i], w_in, w_rec, IDX, hard=hard)
            base = loss_of(f0, int(y[i]))
            grad = np.zeros_like(w_in)
            for row in rows:
                wp = w_in.copy(); wp[row] += eps
                grad[row] = (loss_of(rollout(c, x[i], wp, w_rec, IDX, hard=hard),
                                     int(y[i])) - base) / eps

            a = tr[rows].ravel(); b = grad[rows].ravel()
            den = np.linalg.norm(a) * np.linalg.norm(b)
            if den > 1e-12:
                coss.append(float(a @ b / den))
        coss = np.array(coss)
        results[mode] = coss
        rng = np.random.default_rng(7)
        boots = rng.choice(coss, (4000, len(coss)), replace=True).mean(1)
        print(f"  {mode:6s} {len(coss):>4} {coss.mean():+10.4f} "
              f"[{np.quantile(boots,.025):+.4f},{np.quantile(boots,.975):+.4f}]"
              f"{(np.abs(coss)>.2).mean():>14.3f}")

    # ---- the crucial comparison: is the trace worse than a random vector?
    print("\n  CONTROL: cosine of a RANDOM vector with the same gradient")
    rand_cos = []
    rng = np.random.default_rng(11)
    for i in range(40):
        f0 = rollout(c, x[i], w_in, w_rec, IDX, hard=False)
        base = loss_of(f0, int(y[i]))
        grad = np.zeros_like(w_in)
        for row in rows:
            wp = w_in.copy(); wp[row] += eps
            grad[row] = (loss_of(rollout(c, x[i], wp, w_rec, IDX, hard=False),
                                 int(y[i])) - base) / eps
        r = rng.normal(0, 1, grad[rows].shape).astype(np.float32)
        a = r.ravel(); b = grad[rows].ravel()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        if den > 1e-12:
            rand_cos.append(float(a @ b / den))
    rc = np.array(rand_cos)
    print(f"  {'random':6s} {len(rc):>4} {rc.mean():+10.4f}   (this is the null)")

    print("\n" + "=" * 84)
    print("READING")
    print("=" * 84)
    soft = results["soft"].mean() if len(results["soft"]) else float("nan")
    rand = rc.mean()
    print(f"  trace cosine (soft) = {soft:+.4f}")
    print(f"  random  cosine      = {rand:+.4f}")
    if abs(soft) <= abs(rand) * 1.5:
        print("  => the eligibility trace is NO BETTER THAN RANDOM at pointing")
        print("     toward the true gradient. Scalar modulation of this trace")
        print("     cannot produce directed learning; it random-walks weights.")
    else:
        print("  => the trace carries real directional information, so the")
        print("     failure lies elsewhere (threshold or modulation).")


if __name__ == "__main__":
    main()
