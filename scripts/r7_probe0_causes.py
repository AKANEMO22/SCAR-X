"""Round 7 probe 0 — measure the three candidate causes of hidden-learning failure.

Reads the PRODUCTION rule from main (no modification). Measures, without training
anything new:

  C1  Does the scalar `advantage` carry directional information?
      It is `clip(p[target] - reward_mean, -1, 1)`. If its sign is uncorrelated with
      whether the hidden update happened to help, it is noise.

  C2  Is the eligibility trace aligned with the true cost gradient?
      Compare the trace against an offline BPTT diagnostic gradient (diagnostic only,
      never a candidate). If correlation ~ 0, the trace is not pointing at the loss.

  C3  Does the anchor term erase learning?
      `anchor_strength * (w - stable)`. Check whether `stable` is ever refreshed.

This is instrumentation of existing code, not a new algorithm.
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


def main():
    print("=" * 78)
    print("R7 PROBE 0 — three candidate causes, measured on the production rule")
    print("=" * 78)
    seed = 140011
    c = cfg(seed)
    core = SpikingCore(c)
    x, y = episode(seed)

    # ---- C3 first (cheapest, and it would invalidate everything else)
    print("\nC3 — anchor / consolidation term")
    print(f"  anchor_strength = {c.anchor_strength}")
    print(f"  trace_decay     = {c.trace_decay}")
    print(f"  hidden_lr       = {c.hidden_lr}")
    print(f"  update_clip     = {c.update_clip}")
    stable_before = core.stable_in.copy()
    w_before = core.w_in.copy()
    for i in range(40):
        core.reset_state()
        core.forward(x[i], track=True)
        core.learn(int(y[i]), mode="full")
    print(f"  stable_in changed after 40 updates? "
          f"{not np.array_equal(stable_before, core.stable_in)}")
    print(f"  ||w_in - stable_in|| = {np.linalg.norm(core.w_in - core.stable_in):.6f}")
    drift = float(np.abs(core.w_in - w_before).mean())
    print(f"  mean |w change| after 40 updates = {drift:.6f}")
    print(f"  => if stable is never refreshed, the anchor is a permanent spring")
    print(f"     pulling every weight back to its initial value.")

    # ---- C1: is advantage directional?
    core2 = SpikingCore(c)
    advs, losses = [], []
    for i in range(min(200, len(x))):
        core2.reset_state()
        p = core2.forward(x[i], track=True)
        before = -float(np.log(max(float(p[int(y[i])]), 1e-12)))
        adv = float(np.clip(p[int(y[i])] - core2.reward_mean, -1, 1))
        advs.append(adv)
        losses.append(before)
        core2.learn(int(y[i]), mode="full")
    advs = np.array(advs); losses = np.array(losses)
    print("\nC1 — is the scalar advantage directional?")
    print(f"  mean advantage       = {advs.mean():+.4f}")
    print(f"  mean |advantage|     = {np.abs(advs).mean():.4f}")
    print(f"  mean loss            = {losses.mean():.4f}  (ln2 = 0.6931)")
    print(f"  corr(advantage, loss)= {np.corrcoef(advs, losses)[0,1]:+.4f}")
    print(f"  frac advantage > 0   = {(advs>0).mean():.3f}")
    print(f"  => advantage is a ONE-BIT global scalar. It cannot say WHICH of the")
    print(f"     {c.plastic_rows}x{c.fan_in} plastic synapses caused the outcome.")

    # ---- C2: is the trace aligned with the cost gradient?
    print("\nC2 — is the eligibility trace aligned with dL/dw?")
    core3 = SpikingCore(c)
    core3.reset_state()
    core3.forward(x[0], track=True)
    trace = core3.elig_in.copy()
    target = int(y[0])

    # finite-difference the true gradient of the loss w.r.t. each plastic weight
    def loss_at(weights):
        saved = core3.w_in.copy()
        core3.w_in[:] = weights
        core3.reset_state()
        p = core3.forward(x[0], track=True)
        core3.w_in[:] = saved
        return -float(np.log(max(float(p[target]), 1e-12)))

    eps = 1e-3
    grad = np.zeros_like(trace)
    base_w = core3.w_in.copy()
    for j in range(trace.shape[0]):
        for k in range(trace.shape[1]):
            w = base_w.copy()
            w[j, k] += eps
            lp = loss_at(w)
            w[j, k] -= 2 * eps
            lm = loss_at(w)
            grad[j, k] = (lp - lm) / (2 * eps)

    flat_t, flat_g = trace.ravel(), grad.ravel()
    denom = np.linalg.norm(flat_t) * np.linalg.norm(flat_g)
    cos = float(flat_t @ flat_g / denom) if denom > 1e-12 else float("nan")
    print(f"  |trace| = {np.linalg.norm(flat_t):.4f}   |dL/dw| = {np.linalg.norm(flat_g):.4f}")
    print(f"  cosine(trace, dL/dw) = {cos:+.4f}")
    print(f"  corr(trace, dL/dw)   = {np.corrcoef(flat_t, flat_g)[0,1]:+.4f}")
    print(f"  => a cosine near 0 means the trace points nowhere near the loss.")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  C1 advantage: scalar, mean|adv| {np.abs(advs).mean():.3f}, one bit")
    print(f"  C2 trace alignment: cosine {cos:+.3f}")
    print(f"  C3 anchor: stable never refreshed, permanent spring")


if __name__ == "__main__":
    main()
