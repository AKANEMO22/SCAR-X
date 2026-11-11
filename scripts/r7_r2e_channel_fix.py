"""R7 round 2e — THE DECISIVE TEST: is the readout blind to the neuron's memory?

Round 2d found that at tick 8 the tick-1 bit is decodable from **voltage** at
0.967-0.980 but from **spikes** at 0.498-0.513 — chance.

The production readout sees only:

    features = feature_decay * features + (1 - feature_decay) * spikes

which is a decayed function of SPIKES ONLY. Voltage is never exposed.

If that is right, then delayed_xor is unsolvable for a reason that has nothing to do
with credit assignment: the representation holds the answer in an analog state the
readout never samples. No local learning rule can fix a blind sensory channel.

Decisive test: give a linear readout access to voltage (and adaptation) as well, with
NO hidden learning at all. If that alone lifts delayed_xor from chance to well above
it, then the bottleneck is the READOUT CHANNEL, and hidden plasticity was never the
right lever.
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
    return x, (a ^ b).astype(np.int32)


def channels(core, x):
    """Every candidate readout channel from ONE frozen pass."""
    prod, spike, volt, adapt, combo = [], [], [], [], []
    for i in range(len(x)):
        core.reset_state()
        ss = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(x[i, t], track=False)
            ss += core.spikes
        prod.append(core.features.copy())        # what production actually sees
        spike.append(ss)
        volt.append(core.voltage.copy())
        adapt.append(core.adaptation.copy())
        combo.append(np.concatenate([ss, core.voltage, core.adaptation]))
    return {k: np.stack(v) for k, v in
            (("production(decay)", prod), ("spikesum", spike), ("voltage", volt),
             ("adaptation", adapt), ("sum+volt+adapt", combo))}


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


def main():
    print("=" * 86)
    print("R7 ROUND 2e — which readout channel solves delayed_xor? (NO hidden learning)")
    print("=" * 86)
    names = ("production(decay)", "spikesum", "voltage", "adaptation", "sum+volt+adapt")
    agg = {n: [] for n in names}
    for seed in (140011, 140017, 140023):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        trx, try_ = delayed_bits(seed + 200_000, N)
        tex, tey = delayed_bits(seed + 300_000, N)
        A = channels(core, trx); B = channels(core, tex)
        for n in names:
            _, te = refit(A[n], try_, B[n], tey)
            agg[n].append(te)
        print(f"  seed {seed} done", flush=True)

    print(f"\n  {'channel':>18} {'dims':>6} {'test':>8} {'vs production':>15}")
    base = float(np.mean(agg["production(decay)"]))
    dims = {"production(decay)": 48, "spikesum": 48, "voltage": 48,
            "adaptation": 48, "sum+volt+adapt": 144}
    for n in names:
        m = float(np.mean(agg[n]))
        print(f"  {n:>18} {dims[n]:>6} {m:8.4f} {m-base:+15.4f}")

    print("\n" + "=" * 86)
    print("READING")
    print("=" * 86)
    print("  If 'voltage' or 'sum+volt+adapt' is well above 'production(decay)',")
    print("  then the production readout is blind to the neuron's analog memory")
    print("  and the hidden-learning bottleneck is really a CHANNEL bottleneck.")
    print("  No credit-assignment rule can recover information the readout")
    print("  never receives.")


if __name__ == "__main__":
    main()
