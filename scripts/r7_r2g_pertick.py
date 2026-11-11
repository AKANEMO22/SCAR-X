"""R7 round 2g — at which tick does the XOR become LINEARLY readable?

Round 2f showed the point neuron solves simultaneous XOR linearly at 0.9792. So the
spiking nonlinearity does provide the product term when both bits arrive at once.

Round 2d showed the tick-1 bit survives in VOLTAGE at tick 8 (0.967-0.980) but not in
spikes (chance). Round 2e showed that reading voltage only at the END (tick 11) is
chance.

The missing hypothesis: the XOR becomes linearly readable at an INTERMEDIATE tick,
when the recurrent input from the b-pulse mixes with the still-live voltage carrying a.
If some tick t has high linear XOR decodability, then the information is available in
linear form at that moment, and the production readout's decay filter is averaging it
away.

This scans every tick and reports linear XOR decodability from spikes, voltage, and
both, so the answer is a table rather than a guess.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

N = 512


def delayed_bits(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    a = rng.integers(0, 2, n).astype(np.int32)
    b = rng.integers(0, 2, n).astype(np.int32)
    for i in range(n):
        x[i, 1, 0] += 1.5 * int(a[i])
        x[i, 8, 1] += 1.5 * int(b[i])
        x[i, 10, 7] += 1.5
    return x, (a ^ b).astype(np.int32), a, b


def all_ticks(core, x):
    """Per-tick spike and voltage snapshots from ONE frozen pass."""
    sp, vo = [], []
    for i in range(len(x)):
        core.reset_state()
        s_row, v_row = [], []
        for t in range(12):
            core.step(x[i, t], track=False)
            s_row.append(core.spikes.copy())
            v_row.append(core.voltage.copy())
        sp.append(s_row); vo.append(v_row)
    return np.array(sp), np.array(vo)   # (n, 12, H)


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
    print("=" * 88)
    print("R7 ROUND 2g — per-tick LINEAR decodability of a XOR b")
    print("=" * 88)
    seed = 140011
    core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
    trx, try_, ta, tb = delayed_bits(seed + 200_000, N)
    tex, tey, ea, eb = delayed_bits(seed + 300_000, N)
    SP_tr, VO_tr = all_ticks(core, trx)
    SP_te, VO_te = all_ticks(core, tex)

    print(f"\n  {'tick':>5} {'spikes':>8} {'voltage':>8} {'both':>8} "
          f"{'bit a':>8} {'bit b':>8}  (bit a/b from voltage)")
    best = (0, None)
    for t in range(12):
        a_s = refit(SP_tr[:, t], try_, SP_te[:, t], tey)
        a_v = refit(VO_tr[:, t], try_, VO_te[:, t], tey)
        both = refit(np.concatenate([SP_tr[:, t], VO_tr[:, t]], 1), try_,
                     np.concatenate([SP_te[:, t], VO_te[:, t]], 1), tey)
        da = refit(VO_tr[:, t], ta, VO_te[:, t], ea)
        db = refit(VO_tr[:, t], tb, VO_te[:, t], eb)
        star = ""
        if a_v > best[0]:
            best = (a_v, t); star = " <-"
        print(f"  {t:>5} {a_s:8.4f} {a_v:8.4f} {both:8.4f} {da:8.4f} {db:8.4f}{star}")

    print(f"\n  best linear XOR decodability: {best[0]:.4f} at tick {best[1]}")
    print(f"  simultaneous-XOR reference (round 2f): 0.9792")

    print("\n" + "=" * 88)
    print("READING")
    print("=" * 88)
    print("  If NO tick exceeds ~0.6 linearly, delayed XOR is never linearly")
    print("  available in this population's instantaneous state, and a linear")
    print("  readout cannot recover it regardless of how hidden weights are tuned.")
    print("  That makes the round-2 testbed unable to answer the hidden-learning")
    print("  question, which is itself the finding.")


if __name__ == "__main__":
    main()
