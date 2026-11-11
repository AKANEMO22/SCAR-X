"""R7 probe 1 — verify the zero-gradient finding, which would explain everything.

Probe 0 measured |dL/dw| = 0.0000 on a single episode. If the readout loss is
INSENSITIVE to the plastic hidden weights, then no credit signal can help: the
learning target has no derivative w.r.t. the parameters being updated.

That would be a strictly stronger diagnosis than "the signal is noisy". It must be
verified on many episodes and its cause identified, because an exact zero usually
means a dead path, not a small path.
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
    print("R7 PROBE 1 — is the readout loss really insensitive to plastic weights?")
    print("=" * 78)
    seed = 140011
    c = cfg(seed)
    core = SpikingCore(c)
    x, y = episode(seed)

    # Which weights does `learn` actually change?
    print(f"\n  plastic_rows        = {c.plastic_rows} of {c.hidden}")
    print(f"  rows touched        = {core.rows[:6]}... (first 6)")
    print(f"  w_in shape          = {core.w_in.shape}")
    print(f"  elig_in shape       = {core.elig_in.shape}")
    print(f"  elig_rec shape      = {core.elig_rec.shape}")
    print(f"\n  => elig_in/elig_rec have {core.elig_in.shape[0]} rows.")
    print(f"     learn() does weights[self.rows] where rows has {len(core.rows)} entries,")
    print(f"     so only {len(core.rows)} rows of the trace are ever used.")

    # Does perturbing a PLASTIC row change the loss?
    print("\n  Perturbing a PLASTIC row (in core.rows) vs a NON-plastic row:")
    core.reset_state()
    core.forward(x[0], track=True)
    p0 = core.probabilities()[int(y[0])]
    L0 = -np.log(max(p0, 1e-12))

    def loss_with(rows, eps):
        saved = core.w_in.copy()
        core.w_in[rows] += eps
        core.reset_state()
        p = core.forward(x[0], track=True)
        val = -float(np.log(max(float(p[int(y[0])]), 1e-12)))
        core.w_in[:] = saved
        return val

    plastic_row = int(core.rows[0])
    other_row = int([r for r in range(c.hidden) if r not in set(core.rows.tolist())][0])
    print(f"    base loss                 = {L0:.6f}")
    for eps in (1e-2, 1e-1, 1.0):
        dl_p = loss_with([plastic_row], eps) - L0
        dl_o = loss_with([other_row], eps) - L0
        print(f"    eps={eps:<5} plastic row dL={dl_p:+.6e}   other row dL={dl_o:+.6e}")

    # Finite difference over many episodes: is dL/dw ever non-zero?
    print("\n  Finite-difference gradient magnitude over 40 episodes:")
    mags = []
    for i in range(40):
        core.reset_state()
        core.forward(x[i], track=True)
        base = -np.log(max(float(core.probabilities()[int(y[i])]), 1e-12))
        saved = core.w_in.copy()
        g = 0.0
        for row in core.rows[:4]:
            core.w_in[int(row)] += 1e-2
            core.reset_state()
            p = core.forward(x[i], track=True)
            up = -np.log(max(float(p[int(y[i])]), 1e-12))
            core.w_in[:] = saved
            g += abs(up - base)
        mags.append(g)
    mags = np.array(mags)
    print(f"    mean |dL|           = {mags.mean():.6e}")
    print(f"    max  |dL|           = {mags.max():.6e}")
    print(f"    episodes with |dL|=0: {int((mags == 0).sum())}/40")

    # How many neurons spike at all?
    print("\n  Spike activity (if the population is silent, nothing is causally live):")
    core.reset_state()
    spikes = []
    for t in range(12):
        core.step(x[0, t], track=False)
        spikes.append(core.spikes.sum())
    spk = np.array(spikes)
    print(f"    spikes per tick     = {spk.astype(int).tolist()}")
    print(f"    total spikes        = {int(spk.sum())} over 12 ticks x {c.hidden} neurons")
    print(f"    mean firing rate    = {spk.sum()/(12*c.hidden):.4f}")

    print("\n  Plastic rows' own activity:")
    core.reset_state()
    tot = 0
    for t in range(12):
        core.step(x[0, t], track=False)
        tot += int(core.spikes[core.rows].sum())
    print(f"    spikes in plastic rows = {tot}")
    print(f"    => if 0, the plastic synapses literally never fired and no")
    print(f"       eligibility can exist for them.")


if __name__ == "__main__":
    main()
