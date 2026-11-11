"""R7 round 8 — the one intersection I had not tested: voltage as the readout channel.

WHY THIS IS THE RIGHT NEXT PROBE
--------------------------------
Rounds 1-7 produced two findings that point at the same untested corner:

  Round 2d : at tick 8 the tick-1 bit reads 0.967-0.980 from VOLTAGE but 0.498-0.513
             from SPIKES. Voltage holds the memory; spikes do not.
  Round 1  : the trace-gradient cosine is -0.0303 (CI contains 0) AND the hard threshold
             gives dL/dw = 0 on 25/30 episodes. The spike is piecewise constant in w, so
             the readout loss barely depends on the plastic weights at all.

Both are consequences of reading SPIKES. If the readout instead consumes VOLTAGE:

    features = v_t           (continuous in w_in, and carries the memory)

then (a) dL/dw_in is non-zero almost everywhere because v is an affine function of w_in,
and (b) the channel retains the analog memory that spikes discard.

So the question this probe answers, and nothing else:

    With a voltage readout, does the eligibility trace become directionally aligned with
    the true gradient, where it was not with spikes?

If YES: credit assignment was never the root cause. The root cause was that the readout
consumed a quantised signal with zero derivative, and the "directionless trace" was an
artefact of differentiating a step function. That relocates the bottleneck one layer up
and makes the whole credit-rule line of work mis-targeted.

If NO: the trace is directionless for its own reasons and the spike explanation is
incomplete.

This is a DIAGNOSTIC. No candidate, no training of hidden weights, no production change.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

SEED = 140011
N_TR, N_TE = 512, 256


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


def channel(core, x, kind):
    """Spike sum, voltage at end, or both. One frozen pass per call."""
    out = []
    for i in range(len(x)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(x[i, t], track=False)
            tot += core.spikes
        if kind == "spikes":
            out.append(tot)
        elif kind == "voltage":
            out.append(core.voltage.copy())
        elif kind == "adaptation":
            out.append(core.adaptation.copy())
        else:
            out.append(np.concatenate([tot, core.voltage]))
    return np.stack(out)


def train_linear(F, y, epochs=1500, lr=.5):
    mu = F.mean(0); sd = np.maximum(F.std(0), 1e-6)
    A = (F - mu) / sd
    w = np.zeros((2, F.shape[1])); b = np.zeros(2)
    for _ in range(epochs):
        lg = A @ w.T + b; lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        e = -p; e[np.arange(len(y)), y] += 1
        w += lr * (e.T @ A) / len(y); b += lr * e.mean(0)
    return mu, sd, w, b


def loss_of(F, mu, sd, w, b, y):
    A = (F - mu) / sd
    lg = A @ w.T + b
    lg = lg - lg.max(1, keepdims=True)
    p = np.exp(lg); p = p / p.sum(1, keepdims=True)
    return -float(np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))))


def main():
    print("=" * 88)
    print("R7 ROUND 8 — does a VOLTAGE readout restore a usable gradient?")
    print("=" * 88)
    c = S.cfg(SEED)
    core = S.SECSCore(c, mode="scalar", seed=SEED)
    trx, try_ = task(SEED + 200_000, N_TR)
    tex, tey = task(SEED + 300_000, N_TE)

    # ---------- part 1: which channel carries the task, and does it have a gradient?
    print(f"\n  {'channel':12s} {'dims':>5} {'frozen linear':>14} {'|dL/dw|':>12} "
          f"{'zero-episodes':>14}")
    results = {}
    eps = 1e-2
    for kind in ("spikes", "voltage", "adaptation", "spikes+voltage"):
        F = channel(core, trx, kind)
        Ft = channel(core, tex, kind)
        mu, sd, w, b = train_linear(F, try_)
        acc = float(np.mean(np.argmax(((Ft - mu) / sd) @ w.T + b, 1) == tey))

        # finite-difference sensitivity of the trained loss to plastic weights
        mags, zeros = [], 0
        for i in range(20):
            base_in = channel(core, trx[i:i + 1], kind)
            L0 = loss_of(base_in, mu, sd, w, b, try_[i:i + 1])
            saved = core.w_in.copy()
            tot = 0.0
            for row in core.rows[:4]:
                core.w_in[row] += eps
                Lp = loss_of(channel(core, trx[i:i + 1], kind), mu, sd, w, b,
                             try_[i:i + 1])
                core.w_in[:] = saved
                tot += abs(Lp - L0)
            mags.append(tot)
            if tot == 0.0:
                zeros += 1
        results[kind] = {"acc": acc, "grad": float(np.mean(mags)), "zeros": zeros,
                         "dims": F.shape[1]}
        print(f"  {kind:12s} {F.shape[1]:>5} {acc:14.4f} {np.mean(mags):12.3e} "
              f"{zeros:>10}/20")

    # ---------- part 2: trace alignment under each channel
    print(f"\n  {'channel':12s} {'|trace|':>10} {'|grad|':>10} {'cosine':>10} "
          f"{'random null':>12}")
    rng = np.random.default_rng(7)
    for kind in ("spikes", "voltage"):
        F = channel(core, trx, kind)
        Ft = channel(core, tex, kind)
        mu, sd, w, b = train_linear(F, try_)

        coss, rands = [], []
        for i in range(15):
            core.reset_state()
            core.forward(trx[i], track=True)
            tr = np.zeros_like(core.w_in)
            tr[core.rows] = core.elig_in

            saved = core.w_in.copy()
            base = loss_of(channel(core, trx[i:i + 1], kind), mu, sd, w, b,
                           try_[i:i + 1])
            grad = np.zeros_like(core.w_in)
            for row in core.rows:
                core.w_in[row] += eps
                grad[row] = (loss_of(channel(core, trx[i:i + 1], kind), mu, sd, w, b,
                                     try_[i:i + 1]) - base) / eps
                core.w_in[:] = saved

            a = tr[core.rows].ravel(); bb = grad[core.rows].ravel()
            den = np.linalg.norm(a) * np.linalg.norm(bb)
            if den > 1e-12:
                coss.append(float(a @ bb / den))
            r = rng.normal(0, 1, a.shape)
            den2 = np.linalg.norm(r) * np.linalg.norm(bb)
            if den2 > 1e-12:
                rands.append(float(r @ bb / den2))
        cos_m = float(np.mean(coss)) if coss else float("nan")
        rnd_m = float(np.mean(rands)) if rands else float("nan")
        print(f"  {kind:12s} {'':>10} {'':>10} {cos_m:+10.4f} {rnd_m:+12.4f}")

    print("\n" + "=" * 88)
    print("READING")
    print("=" * 88)
    sp, vo = results["spikes"], results["voltage"]
    print(f"  spikes : acc {sp['acc']:.4f}  |dL/dw| {sp['grad']:.3e}  "
          f"zeros {sp['zeros']}/20")
    print(f"  voltage: acc {vo['acc']:.4f}  |dL/dw| {vo['grad']:.3e}  "
          f"zeros {vo['zeros']}/20")
    ratio = (vo["grad"] / sp["grad"]) if sp["grad"] > 0 else float("inf")
    print(f"\n  gradient ratio voltage/spikes = {ratio:.1f}x")
    print(f"  => if voltage has a materially larger and non-zero gradient while")
    print(f"     carrying comparable information, then the readout CHANNEL is what")
    print(f"     starved credit assignment, not the credit rule itself.")


if __name__ == "__main__":
    main()
