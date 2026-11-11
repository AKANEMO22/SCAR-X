"""R7 round 2 — mechanism comparison on a testbed with verified headroom.

TASK: delayed_xor (bits at ticks 1 and 8, gate at 10). CHANNEL: spike sum.
Headroom measured: spike-sum linear 0.5404 vs quadratic 0.9948 = +0.4544.

The question is whether a local rule can make that 45pp of nonlinear information
LINEARLY available by changing hidden weights.

Arms: frozen, production, secs, cts, shuffled, zero.
Every arm: same population seed, same data, same spike-sum channel, same common refit
head (z-scaled, fixed budget), train accuracy reported so an undertrained head cannot
masquerade as a weak representation.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

DEV = (140011, 140017, 140023)
N_TR, N_TE = 512, 256
ARMS = ("frozen", "production", "secs", "cts", "shuffled", "zero")
K_TRACES = 8


def delayed_xor(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.int32)
    for i in range(n):
        a = int(rng.integers(2)); b = a ^ int(y[i])
        x[i, 1, 0] += 1.5 * a
        x[i, 8, 1] += 1.5 * b
        x[i, 10, 7] += 1.5
    return x, y


def spike_sum(core, x):
    out = []
    for i in range(len(x)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(x[i, t], track=False)
            tot += core.spikes
        out.append(tot)
    return np.stack(out)


def refit(A, ya, B, yb, epochs=800, lr=.5):
    """Common refit. z-scaled on TRAIN stats only; returns (train, test)."""
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


class CTSCore(S.SECSCore):
    """Mechanism B: contrastive trace selection.

    Keeps the last K traces with the error magnitude observed when each was recorded,
    and weights each by the improvement that followed it. Local, streaming, no future
    ticks beyond the update instant.
    """

    def __init__(self, config, seed=0, k=K_TRACES):
        super().__init__(config, mode="scalar", seed=seed)
        self.K = int(k)
        self._buf = []
        self._err_mag = []

    def learn(self, target, mode="full", reward=None):
        c = self.config
        p = self.probabilities()
        loss = -float(np.log(max(float(p[target]), 1e-12)))
        error = -p.copy(); error[target] += 1
        self._last_error = error.astype(np.float32)

        self.w_out += np.clip(c.readout_lr * error[:, None] * self.features[None, :],
                              -c.update_clip, c.update_clip)
        self.bias += np.clip(c.readout_lr * error, -c.update_clip, c.update_clip)
        np.clip(self.w_out, -c.weight_clip, c.weight_clip, out=self.w_out)

        if mode == "frozen":
            return loss

        # record the CURRENT trace with the error magnitude that produced it
        self._buf.append((self.elig_in.copy(), self.elig_rec.copy()))
        self._err_mag.append(abs(float(self._last_error[target])))
        if len(self._buf) > self.K:
            self._buf.pop(0); self._err_mag.pop(0)

        # improvement attributable to each retained trace
        now = abs(float(self._last_error[target]))
        gains = np.array([max(0.0, now - m) for m in self._err_mag])
        if gains.sum() <= 1e-12:
            wts = np.zeros(len(self._buf))
        else:
            wts = gains / gains.sum()

        eff_in = sum(w * b[0] for w, b in zip(wts, self._buf)) if self._buf else self.elig_in
        eff_rec = sum(w * b[1] for w, b in zip(wts, self._buf)) if self._buf else self.elig_rec

        advantage = float(np.clip(p[target] - self.reward_mean, -1, 1))
        self.reward_mean = .95 * self.reward_mean + .05 * float(p[target])

        for name, trace in (("in", eff_in), ("rec", eff_rec)):
            weights = getattr(self, "w_" + name)
            stable = getattr(self, "stable_" + name)
            delta = c.hidden_lr * advantage * trace \
                - c.anchor_strength * (weights[self.rows] - stable[self.rows])
            weights[self.rows] += np.clip(delta, -c.update_clip, c.update_clip)
            np.clip(weights, -c.weight_clip, c.weight_clip, out=weights)

        self.updates += 1
        return loss


def make_core(arm, seed):
    c = S.cfg(seed)
    if arm == "cts":
        return CTSCore(c, seed=seed)
    mode = {"production": "scalar", "secs": "secs",
            "shuffled": "shuffled_credit", "zero": "zero_credit"}.get(arm, "scalar")
    return S.SECSCore(c, mode=mode, seed=seed,
                      anchor_refresh=200 if arm in ("secs", "cts") else 0)


def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r7/protocol-round2.md"
    psha = hashlib.sha256(proto.read_bytes()).hexdigest()
    run = new_run(root, "r7-r2-delayed-xor-v1", {
        "protocol_sha256": psha, "seeds": list(DEV), "arms": list(ARMS),
        "task": "delayed_xor", "channel": "spike_sum",
        "registered_headroom": 0.4544, "note": "DEV only"})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            trx, try_ = delayed_xor(seed + 200_000, N_TR)
            tex, tey = delayed_xor(seed + 300_000, N_TE)

            for arm in ARMS:
                core = make_core(arm, seed)
                w0 = core.w_in.copy()
                order = np.random.default_rng(seed + 77).permutation(N_TR)
                for i in order:
                    core.reset_state()
                    core.forward(trx[i], track=True)
                    core.learn(int(try_[i]), mode="frozen" if arm == "frozen" else "full")

                A = spike_sum(core, trx); B = spike_sum(core, tex)
                tr_acc, te_acc = refit(A, try_, B, tey)

                rows.append({
                    "seed": seed, "arm": arm,
                    "refit_train": tr_acc, "refit_test": te_acc,
                    "weight_delta": float(np.abs(core.w_in - w0).mean()),
                    "updates": core.updates,
                    "hidden_digest": S.digest(core)[:16],
                })
                mon.check()
                print(f"  seed{seed} {arm:11s} refit train {tr_acc:.4f} test {te_acc:.4f} "
                      f"|dw| {np.abs(core.w_in-w0).mean():.5f}", flush=True)

    def M(arm, k):
        return float(np.mean([r[k] for r in rows if r["arm"] == arm]))

    summary = {a: {k: M(a, k) for k in ("refit_train", "refit_test", "weight_delta")}
               for a in ARMS}

    # paired per-seed gain vs frozen
    frozen = {r["seed"]: r["refit_test"] for r in rows if r["arm"] == "frozen"}
    gains = {}
    for a in ARMS:
        if a == "frozen":
            continue
        d = [r["refit_test"] - frozen[r["seed"]] for r in rows if r["arm"] == a]
        gains[a] = {"mean": float(np.mean(d)), "per_seed": d}

    summary["paired_gain_vs_frozen"] = gains

    def sanitize(o):
        if isinstance(o, dict):
            return {k: sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sanitize(v) for v in o]
        if isinstance(o, float) and o != o:
            return None
        return o

    write_json(run / "result.json",
               sanitize({"protocol_sha256": psha, "trials": rows, "summary": summary,
                         "resources": dict(mon.report()),
                         "wall_seconds": time.time() - t0}))

    print("\n" + "=" * 88)
    print("R7 ROUND 2 — delayed_xor, spike-sum channel (headroom 0.4544)")
    print("=" * 88)
    print(f"  {'arm':12s} {'train':>8} {'test':>8} {'vs frozen':>12} {'|dw|':>10}")
    for a in ARMS:
        s = summary[a]
        g = summary["paired_gain_vs_frozen"].get(a)
        gtxt = f"{g['mean']*100:+11.2f}pp" if g else f"{'—':>12}"
        print(f"  {a:12s} {s['refit_train']:8.4f} {s['refit_test']:8.4f} {gtxt} "
              f"{s['weight_delta']:10.5f}")
    print(f"\n  ceiling (quadratic refit, round 1) = 0.9948")
    print(f"\nRESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
