"""R7 round 2c — the memory horizon: a mechanistic explanation for the ceiling.

Round 2b's oracle was a random-direction hill climber. It lifted test accuracy by only
+1.6 to +7.4pp over 240 steps, which is too weak a search to prove that 45pp is
unreachable. **I therefore do not use it to settle E1 vs E2.** It is reported as a
failed instrument, not as evidence.

A cleaner and decisive question: delayed_xor needs a value written at tick 1 to still
be readable at tick 8. A single ALIF neuron with leak λ holds a perturbation as
approximately λ^Δt. So the population's memory horizon is fixed by λ, and it either
reaches 7 ticks or it does not.

    λ = 0.65  ->  0.65^7 = 0.049   (4.9% survival)

This probe measures the horizon empirically: inject one pulse at tick 1, read the
population state at every later tick, and see when the signal is indistinguishable
from the no-pulse baseline. If the horizon is ~2-3 ticks, then a 7-tick delay is beyond
what this network can encode in a single neuron, and NO local credit rule can create a
representation the dynamics cannot hold. That would relocate the bottleneck from
learning to memory capacity — a different and more actionable finding.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402
from scarx.neural.core import Config  # noqa: E402


def base_cfg(seed):
    return S.cfg(seed)


def cfg_with_leak(seed, leak, threshold=None):
    """A Config with a different leak, to map how the horizon changes."""
    c = S.cfg(seed)
    fields = {f: getattr(c, f) for f in c.__dataclass_fields__}
    fields["leak"] = leak
    if threshold is not None:
        fields["threshold"] = threshold
    return Config(**fields)


def probe_horizon(core, tick_inject=1, n_seq=64, seed=0):
    """Distance between 'pulse at tick_inject' and 'no pulse', per later tick."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, .08, (n_seq, 12, 8)).astype(np.float32)
    pulse = base.copy()
    pulse[:, tick_inject, 0] += 1.5

    def states(seqs):
        out = []
        for i in range(n_seq):
            core.reset_state()
            rows = []
            for t in range(12):
                core.step(seqs[i, t], track=False)
                rows.append(np.concatenate([core.spikes.copy(), core.voltage.copy()]))
            out.append(rows)
        return np.array(out)   # (n, 12, 2H)

    a = states(base)
    b = states(pulse)
    # normalise by the response at the injection tick itself
    scale = np.abs(b[:, tick_inject] - a[:, tick_inject]).mean() + 1e-12
    rel = [float(np.abs(b[:, t] - a[:, t]).mean() / scale) for t in range(12)]
    return rel


def main():
    print("=" * 88)
    print("R7 ROUND 2c — how long does a single pulse survive in the population?")
    print("=" * 88)
    seed = 140011
    core = S.SECSCore(base_cfg(seed), mode="scalar", seed=seed)
    rel = probe_horizon(core, tick_inject=1, seed=seed)
    print(f"\n  relative signal remaining after the pulse at tick 1 (leak=0.65):")
    print(f"  {'tick':>5} {'Δ':>3} {'relative response':>18} {'bar'}")
    for t, v in enumerate(rel):
        if t < 1:
            continue
        bar = "#" * int(min(v, 1.0) * 40)
        print(f"  {t:>5} {t-1:>3} {v:18.4f} {bar}")

    # 10% of the initial response is a reasonable 'still readable' line
    horizon = max([t for t, v in enumerate(rel) if v > 0.10], default=0) - 1
    print(f"\n  memory horizon at 10% threshold: {horizon} ticks")
    print(f"  delayed_xor needs: 7 ticks")
    print(f"  => {'WITHIN' if horizon >= 7 else 'BEYOND'} reach of the dynamics")

    # does a slower leak help? (diagnostic only - production leak is unchanged)
    print(f"\n  how the horizon depends on leak (diagnostic only, production "
          f"leak stays 0.65):")
    print(f"  {'leak':>6} {'horizon@10%':>13} {'reaches 7?':>11}")
    for leak in (0.65, 0.80, 0.90, 0.95):
        c = cfg_with_leak(seed, leak)
        core2 = S.SECSCore(c, mode="scalar", seed=seed)
        r = probe_horizon(core2, tick_inject=1, seed=seed)
        h = max([t for t, v in enumerate(r) if v > 0.10], default=0) - 1
        print(f"  {leak:>6.2f} {h:>13} {('yes' if h >= 7 else 'no'):>11}")

    print("\n" + "=" * 88)
    print("READING")
    print("=" * 88)
    print("  A single pulse either survives 7 ticks or it does not. If it does not,")
    print("  then delayed_xor asks the population to hold a value longer than its")
    print("  membrane constant allows, and no credit-assignment rule can fix that:")
    print("  the information is not merely unlearned, it has decayed.")


if __name__ == "__main__":
    main()
