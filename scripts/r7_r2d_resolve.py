"""R7 round 2d — resolve a contradiction between two of my own measurements.

  probe 5   : spike-sum QUADRATIC refit on delayed_xor = 0.9948 (information is there)
  probe 2c  : a single pulse survives only 3 ticks at leak 0.65 (information decays)

Both cannot be naively true. The reconciling hypothesis: the signal at tick 8 is
SMALL but CONSISTENT, so a second-order reader can amplify it while a first-order
reader cannot. That would mean the information is present in a form that hidden
plasticity could in principle expose to a linear readout — a much softer conclusion
than "beyond reach of the dynamics".

Decisive test: at tick 8, how well does a LINEAR probe on the instantaneous state
decode the tick-1 bit `a`? And how much of that survives through only the spike
channel (discrete) versus the analog voltage?
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

N = 1024


def delayed_bits(seed, n):
    """Return sequences plus the TWO raw bits separately (a at t=1, b at t=8)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    a = rng.integers(0, 2, n).astype(np.int32)
    b = rng.integers(0, 2, n).astype(np.int32)
    y = a ^ b
    for i in range(n):
        x[i, 1, 0] += 1.5 * int(a[i])
        x[i, 8, 1] += 1.5 * int(b[i])
        x[i, 10, 7] += 1.5
    return x, a, b, y


def snapshot(core, x, at_tick, part):
    out = []
    for i in range(len(x)):
        core.reset_state()
        for t in range(at_tick + 1):
            core.step(x[i, t], track=False)
        if part == "spikes":
            out.append(core.spikes.copy())
        elif part == "voltage":
            out.append(core.voltage.copy())
        else:
            out.append(np.concatenate([core.spikes, core.voltage]))
    return np.stack(out)


def refit(A, ya, B, yb, epochs=800, lr=.5):
    mu = A.mean(0); sd = np.maximum(A.std(0), 1e-6)
    A = (A - mu) / sd; B = (B - mu) / sd
    w = np.zeros((2, A.shape[1])); bb = np.zeros(2)
    for _ in range(epochs):
        lg = A @ w.T + bb; lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        e = -p; e[np.arange(len(ya)), ya] += 1
        w += lr * (e.T @ A) / len(ya); bb += lr * e.mean(0)
    return float(np.mean(np.argmax(B @ w.T + bb, 1) == yb))


def main():
    print("=" * 84)
    print("R7 ROUND 2d — does the tick-1 bit survive to tick 8 in LINEAR form?")
    print("=" * 84)
    seed = 140011
    for seed in (140011, 140017, 140023):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        trx, ta, tb, ty = delayed_bits(seed + 200_000, N)
        tex, ea, eb, ey = delayed_bits(seed + 300_000, N)

        print(f"\n  seed {seed}")
        print(f"    {'snapshot':>18} {'decode bit a':>13} {'decode XOR':>12}")
        for at in (2, 8):
            for part in ("spikes", "voltage"):
                A = snapshot(core, trx, at, part)
                B = snapshot(core, tex, at, part)
                acc_a = refit(A, ta, B, ea)
                acc_y = refit(A, ty, B, ey)
                lab = f"t={at} {part}"
                print(f"    {lab:>18} {acc_a:13.4f} {acc_y:12.4f}")

    print("\n" + "=" * 84)
    print("READING")
    print("=" * 84)
    print("  If 'decode bit a at t=8' is well above 0.5, the value IS carried")
    print("  across the 7-tick gap in a linearly readable form and the horizon")
    print("  measurement (which used a one-off amplitude perturbation) understated")
    print("  the true memory. If it is at 0.5, the information is genuinely gone")
    print("  from the analog state and probe 5's quadratic result must come from")
    print("  the spike path.")


if __name__ == "__main__":
    main()
