"""R7 probe 2 — locate the exact source of the zero gradient.

Probe 1 found dL/dw = 0 in 39/40 episodes. An exact zero is structural, not small.
This isolates which part of the forward pass kills the derivative:

  H_a  the hard threshold  (spike = 1[v >= limit]) has derivative 0 a.e.
  H_b  the sparse activity means most neurons never contribute at all
  H_c  the plastic rows specifically are silent / disconnected

The discriminating measurement: replace ONLY the threshold with a soft one, keep
everything else identical, and re-measure dL/dw and the trace alignment. If the
gradient appears and aligns with the trace, the threshold is the bottleneck.
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
    """Frozen-population forward with the threshold optionally softened."""
    H = c.hidden
    v = np.zeros(H, np.float32); a = v.copy(); z = v.copy()
    f = np.zeros(H, np.float32)
    spk = np.zeros((len(x), H), np.float32)
    for s in range(len(x)):
        v = c.leak * v + w_in @ x[s] + np.sum(w_rec * z[idx], axis=1)
        lim = c.threshold + c.adaptation_strength * a
        if hard:
            z = (v >= lim).astype(np.float32)
        else:
            z = 1.0 / (1.0 + np.exp(-beta * (v - lim)))
        v = v - z * lim
        a = c.adaptation_decay * a + z
        f = c.feature_decay * f + (1 - c.feature_decay) * z
        spk[s] = z
    return f, spk


def main():
    print("=" * 80)
    print("R7 PROBE 2 — where does the zero gradient come from?")
    print("=" * 80)
    seed = 140011
    c = cfg(seed)
    core = SpikingCore(c)
    x, y = episode(seed)
    rows = core.rows.copy()

    # attach the static random index table used by the rollout helper
    g = np.random.default_rng(c.seed)
    IDX = np.stack([g.choice(c.hidden, c.fan_in, replace=False)
                    for _ in range(c.hidden)]).astype(np.int32)

    w_in = core.w_in.copy(); w_rec = core.w_rec.copy()
    w_out = core.w_out.copy(); b_out = core.bias.copy()

    def loss(f, t):
        lg = w_out @ f + b_out
        lg = lg - lg.max()
        p = np.exp(lg); p = p / p.sum()
        return -float(np.log(max(float(p[t]), 1e-12)))

    # ---- baseline sparsity
    f0, spk0 = rollout(c, x[0], w_in, w_rec, IDX, hard=True)
    print(f"\n  HARD threshold: total spikes = {int(spk0.sum())}, "
          f"plastic-row spikes = {int(spk0[:, rows].sum())}")
    print(f"  neurons that EVER spike  = {int((spk0.sum(0) > 0).sum())}/48")
    print(f"  plastic rows that spiked = {int((spk0[:, rows].sum(0) > 0).sum())}/12")

    # ---- dL/dw for hard vs soft threshold
    print("\n  dL/dw_in over 30 episodes, hard vs soft threshold (eps=1e-2)")
    eps = 1e-2
    res = {"hard": [], "soft": []}
    for i in range(30):
        for mode, hard in (("hard", True), ("soft", False)):
            f, _ = rollout(c, x[i], w_in, w_rec, IDX, hard=hard)
            base = loss(f, int(y[i]))
            tot = 0.0
            for row in rows:
                wp = w_in.copy(); wp[row] += eps
                fp, _ = rollout(c, x[i], wp, w_rec, IDX, hard=hard)
                tot += abs(loss(fp, int(y[i])) - base)
            res[mode].append(tot)
    for mode in ("hard", "soft"):
        v = np.array(res[mode])
        print(f"    {mode:5s}: mean |dL| = {v.mean():.6e}  max = {v.max():.6e}  "
              f"zero-episodes = {int((v==0).sum())}/30")

    # ---- trace alignment under each
    print("\n  cosine(eligibility trace, dL/dw) under each threshold")
    for mode, hard in (("hard", True), ("soft", False)):
        core.reset_state()
        core.forward(x[0], track=True)
        tr = np.zeros_like(w_in)
        tr[rows] = core.elig_in
        # true gradient w.r.t. the SAME rows
        f, _ = rollout(c, x[0], w_in, w_rec, IDX, hard=hard)
        base = loss(f, int(y[0]))
        grad = np.zeros_like(w_in)
        for row in rows:
            wp = w_in.copy(); wp[row] += eps
            fp, _ = rollout(c, x[0], wp, w_rec, IDX, hard=hard)
            grad[row] = (loss(fp, int(y[0])) - base) / eps
        a = tr[rows].ravel(); b = grad[rows].ravel()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(a @ b / den) if den > 1e-12 else float("nan")
        print(f"    {mode:5s}: |trace|={np.linalg.norm(a):.4e}  "
              f"|grad|={np.linalg.norm(b):.4e}  cosine={cos:+.4f}")

    # ---- does a soft threshold change the INFORMATION (not just the gradient)?
    print("\n  Ridge decodability of the soft-threshold features (information check)")
    from scarx.evaluation.tasks21 import ridge_probe
    fh = np.stack([rollout(c, x[i], w_in, w_rec, IDX, hard=True)[0] for i in range(64)])
    fs = np.stack([rollout(c, x[i], w_in, w_rec, IDX, hard=False)[0] for i in range(64)])
    xh, yh = episode(seed + 900, 64)
    fh2 = np.stack([rollout(c, xh[i], w_in, w_rec, IDX, hard=True)[0] for i in range(64)])
    fs2 = np.stack([rollout(c, xh[i], w_in, w_rec, IDX, hard=False)[0] for i in range(64)])
    print(f"    hard  ridge = {ridge_probe(fh, y, fh2, yh):.4f}")
    print(f"    soft  ridge = {ridge_probe(fs, y, fs2, yh):.4f}")

    print("\n" + "=" * 80)
    print("READING")
    print("=" * 80)
    print("  If hard gives |grad| = 0 and soft gives |grad| > 0 while ridge is")
    print("  comparable, then the threshold destroys the LEARNING PATH only, not")
    print("  the INFORMATION. That separates 'no signal' from 'no information'.")


if __name__ == "__main__":
    main()
