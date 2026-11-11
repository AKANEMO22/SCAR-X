"""R7 round 13 — HTA: homeostatic threshold adaptation, targeting the measured defect.

ROUND 12 ESTABLISHED THE TARGET
-------------------------------
  effective rank 4.60 of 48 (9.6%), dead units 25.7 of 48, corr(rank, accuracy) = +0.795
  across a 4-point leak sweep.

The representation does not visit enough distinct states. 25.7 neurons never fire at all.

MECHANISM (local, no new neurons, no new topology, no BPTT)
-----------------------------------------------------------
Each neuron keeps a slow estimate of its own firing rate and adjusts its OWN threshold:

    r_i      <- (1-alpha) r_i + alpha * spike_i          per-neuron running rate
    b_i      <- b_i + beta * (r_i - r_star)              threshold adaptation
    threshold_i = threshold + b_i                         used in the spike test

A neuron that never fires has r_i ~ 0 < r_star, so b_i DECREASES and its threshold falls
until it fires. A neuron that fires constantly has r_i > r_star, so b_i INCREASES and it
quiets down. `r_star` is the target population firing rate.

Every signal is available at the neuron: its own spike, its own rate estimate, its own
threshold. No error transport, no future ticks, streaming, 1 float of state per neuron.

PREDICTION (registered before running)
  1  dead units drop well below 25.7
  2  effective rank rises above 4.60
  3  decodability rises with it, because round 12 measured rank predicts accuracy at +0.795
  4  if rank rises but accuracy does NOT, the +0.795 was correlational and I will say so

CONTROLS
  frozen        no adaptation (baseline)
  hta           homeostatic adaptation on
  hta_shuffled  the SAME threshold deltas, permuted across neurons each step. Same
                magnitude of threshold perturbation, no per-neuron feedback. If this
                matches HTA, the benefit is generic noise, not homeostasis.
  hta_scale     b_i multiplied by a constant, i.e. adaptation at matched magnitude but
                fixed sign - tests whether the FEEDBACK (sign of r_i - r_star) matters.

Production `src/scarx/neural/` untouched: the mechanism lives in a research subclass.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.neural.core import Config, SpikingCore  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 384, 256
ALPHA, BETA, R_STAR = 0.05, 0.02, 0.10
ARMS = ("frozen", "hta", "hta_shuffled", "hta_scale")


def cfg(seed):
    return Config(seed=seed, inputs=8, hidden=48, outputs=2, fan_in=8,
                  plastic_rows=12, leak=.65, threshold=.5, feature_decay=.25,
                  adaptation_strength=.02)


def parity3(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        bits = [int(rng.integers(2)) for _ in range(3)]
        y[i] = bits[0] ^ bits[1] ^ bits[2]
        t = int(rng.integers(1, 4))
        for k, bb in enumerate(bits):
            x[i, t, k] += 1.5 * bb
        x[i, 10, 7] += 1.5
    return x, y


class HTACore(SpikingCore):
    """Forward pass identical to production except the per-neuron threshold."""

    def __init__(self, config, mode="hta", seed=0):
        self.b = np.zeros(config.hidden, np.float64)
        self.rate = np.full(config.hidden, R_STAR, np.float64)
        self.mode = mode
        self._rng = np.random.default_rng(seed + 777)
        super().__init__(config)

    def reset_state(self):
        super().reset_state()

    def step(self, x, track=True):
        c = self.config
        x = np.asarray(x, dtype=np.float32)
        old = self.spikes.copy()
        current = self.w_in @ x + np.sum(self.w_rec * old[self.indices], axis=1)
        self.voltage = c.leak * self.voltage + current
        # ---- the only change: per-neuron threshold
        limit = c.threshold + c.adaptation_strength * self.adaptation + self.b
        self.spikes = (self.voltage >= limit).astype(np.float32)
        self.voltage -= self.spikes * limit
        self.adaptation = c.adaptation_decay * self.adaptation + self.spikes
        self.features = c.feature_decay * self.features + (1 - c.feature_decay) * self.spikes
        if track:
            r = self.rows
            self.elig_in *= c.trace_decay
            self.elig_in += (self.spikes[r, None] * self.pre[None, :c.inputs]
                             - .5 * self.post[r, None] * x[None, :])
            self.elig_rec *= c.trace_decay
            self.elig_rec += (self.spikes[r, None] * self.pre[c.inputs:][self.indices[r]]
                              - .5 * self.post[r, None] * old[self.indices[r]])
            np.clip(self.elig_in, -5, 5, out=self.elig_in)
            np.clip(self.elig_rec, -5, 5, out=self.elig_rec)
            self.pre *= c.trace_decay
            self.pre[:c.inputs] += x
            self.pre[c.inputs:] += old
            self.post = c.trace_decay * self.post + self.spikes
        # ---- homeostasis update
        if self.mode != "frozen":
            self.rate = (1 - ALPHA) * self.rate + ALPHA * self.spikes
            drive = BETA * (self.rate - R_STAR)
            if self.mode == "hta":
                self.b = self.b + drive
            elif self.mode == "hta_shuffled":
                self.b = self.b + drive[self._rng.permutation(len(drive))]
            elif self.mode == "hta_scale":
                self.b = self.b + np.abs(drive)
            np.clip(self.b, -0.4, 0.4, out=self.b)
        return self.probabilities()


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def spike_sum(core, X):
    out, fire = [], []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
        out.append(tot)
        fire.append(core.spikes.copy())
    return np.stack(out), np.stack(fire)


def eff_rank(F):
    C = np.cov(F - F.mean(0), rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def train_head(F, y, epochs=1200, lr=.5):
    mu = F.mean(0)
    # Floor at 0.05 (RELATIVE to the 0..12 spike-sum range), not 1e-6.
    # Local scaling mistakes are how a "1e6" z-feature blew up an earlier run.
    sd = np.maximum(F.std(0), 0.05)
    A = (F - mu) / sd
    w = np.zeros((2, A.shape[1])); b = np.zeros(2)
    for _ in range(epochs):
        p = softmax(A @ w.T + b)
        e = -p; e[np.arange(len(y)), y] += 1
        w += lr * (e.T @ A) / len(y); b += lr * e.mean(0)
    return mu, sd, w, b


def acc(F, y, mu, sd, w, b):
    return float(np.mean(np.argmax(((F - mu) / sd) @ w.T + b, 1) == y))


def run():
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "runs" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-r7-r13-hta-v1"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in DEV:
        trx, try_ = parity3(seed + 200_000, N_TR)
        tex, tey = parity3(seed + 300_000, N_TE)
        for arm in ARMS:
            core = HTACore(cfg(seed), mode=arm, seed=seed)
            # one pass over the training stream so the homeostasis can act
            for i in np.random.default_rng(seed + 5).permutation(N_TR):
                core.reset_state()
                for t in range(12):
                    core.step(trx[i, t], track=True)
            F, fire = spike_sum(core, trx)
            Ft, _ = spike_sum(core, tex)
            ever = fire.reshape(-1, fire.shape[-1]).max(0)
            r = eff_rank(F)
            mu, sd, w, b = train_head(F, try_)
            a = acc(Ft, tey, mu, sd, w, b)
            rec = {"seed": seed, "arm": arm, "rank": r,
                   "dead": int((ever == 0).sum()),
                   "rate": float(fire.mean()),
                   "acc": a,
                   "b_spread": float(np.abs(core.b).mean())}
            rows.append(rec)
            print(f"  seed{seed} {arm:14s} rank {r:6.2f} dead {rec['dead']:2d} "
                  f"rate {rec['rate']:.3f} acc {a:.4f} |b| {rec['b_spread']:.4f}",
                  flush=True)

    print("\n" + "=" * 88)
    print("R7 ROUND 13 — homeostatic threshold adaptation vs the measured defect")
    print("=" * 88)
    print(f"  {'arm':14s} {'rank':>7} {'dead':>6} {'rate':>7} {'acc':>8} "
          f"{'rank gain':>10} {'acc gain':>10}")
    base = {r["seed"]: r for r in rows if r["arm"] == "frozen"}
    summary = {}
    for arm in ARMS:
        sel = [r for r in rows if r["arm"] == arm]
        rk = float(np.mean([r["rank"] for r in sel]))
        ac = float(np.mean([r["acc"] for r in sel]))
        dr = float(np.mean([r["rank"] - base[r["seed"]]["rank"] for r in sel]))
        da = float(np.mean([r["acc"] - base[r["seed"]]["acc"] for r in sel]))
        summary[arm] = {"rank": rk, "dead": float(np.mean([r["dead"] for r in sel])),
                        "rate": float(np.mean([r["rate"] for r in sel])),
                        "acc": ac, "rank_gain": dr, "acc_gain": da}
        print(f"  {arm:14s} {rk:7.2f} {summary[arm]['dead']:6.1f} "
              f"{summary[arm]['rate']:7.3f} {ac:8.4f} {dr:+10.2f} {da*100:+9.2f}p")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    h = summary["hta"]
    s = summary["hta_shuffled"]
    print(f"  HTA      rank {h['rank']:.2f} (+{h['rank_gain']:.2f})  "
          f"dead {h['dead']:.1f}  acc {h['acc']:.4f} ({h['acc_gain']*100:+.2f}pp)")
    print(f"  shuffled rank {s['rank']:.2f} (+{s['rank_gain']:.2f})  "
          f"dead {s['dead']:.1f}  acc {s['acc']:.4f} ({s['acc_gain']*100:+.2f}pp)")
    if h["rank_gain"] > 0.5 and h["acc_gain"] >= 0.05 and \
            h["acc_gain"] > s["acc_gain"] + 0.03:
        print("  => HTA RAISES RANK AND ACCURACY, and beats its shuffled control.")
        print("     The geometry claim survives, and the mechanism acts on it.")
        verdict = "PROMISING"
    elif h["rank_gain"] > 0.5:
        print("  => HTA RAISES RANK but NOT accuracy, or the shuffled control matches.")
        print("     Round 12's +0.795 was correlational; rank is not the lever.")
        verdict = "RANK_NO_GAIN"
    else:
        print("  => HTA does not raise rank. The mechanism does not reach the defect.")
        verdict = "NO_EFFECT"

    (run_dir / "result.json").write_text(
        json.dumps({"trials": rows, "summary": summary, "verdict": verdict,
                    "params": {"alpha": ALPHA, "beta": BETA, "r_star": R_STAR}},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\nRESULT {run_dir}")


if __name__ == "__main__":
    run()
