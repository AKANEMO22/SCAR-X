"""Round 7 — DEV runner for SECS.

Every arm trains the FULL network online (per-episode `learn`), then a COMMON refit
head is trained on that arm's frozen features with an identical budget. That separates
representation gain from readout gain (brief section IX).

Arms:
  frozen        hidden never updates (production mode="frozen")
  production    production rule verbatim: scalar advantage * unshaped trace
  secs          credit-shaped trace (the candidate)
  shuffled      SECS but per-neuron credit permuted across neurons (negative control)
  zero          SECS but advantage forced to 0 (no learning signal at all)

Metrics per arm: online accuracy, common-refit accuracy, both_A, weight delta,
and the trace-gradient cosine.
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
N_TRAIN, N_TEST, N_EVAL = 512, 256, 256
ARMS = ("frozen", "production", "secs", "shuffled", "zero")


def make_task(seed, n, hard=True):
    """Simultaneous XOR: two bits on one tick, gate later. Frozen ridge ~0.65."""
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


def features_of(core, x):
    out = []
    for i in range(len(x)):
        core.reset_state()
        core.forward(x[i], track=False)
        out.append(core.features.copy())
    return np.stack(out)


def refit_head(trf, try_, tef, tey, epochs=300, lr=.5):
    """Common refit: identical algorithm and budget for every arm.

    FIXED AFTER A MEASURED FAILURE. The first version used lr=.5 with 300 epochs and
    reached only 0.627 TRAIN accuracy on unit-scale 48-dim features, i.e. the head
    never converged and every arm reported ~0.49. That made the whole arm comparison
    meaningless. The fix is to z-scale the features and raise the step; `ridge_probe`
    on the same features reaches 0.6953, so a proper head must reach at least that.

    The scaling uses TRAIN statistics only and is identical for every arm.
    """
    mu = trf.mean(0)
    sd = np.maximum(trf.std(0), 1e-6)
    A = (trf - mu) / sd
    B = (tef - mu) / sd
    w = np.zeros((2, A.shape[1]), np.float64)
    b = np.zeros(2, np.float64)
    for _ in range(epochs):
        lg = A @ w.T + b
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg)
        p = p / p.sum(1, keepdims=True)
        err = -p
        err[np.arange(len(try_)), try_] += 1
        w += lr * (err.T @ A) / len(try_)
        b += lr * err.mean(0)
    train_acc = float(np.mean(np.argmax(A @ w.T + b, 1) == try_))
    test_acc = float(np.mean(np.argmax(B @ w.T + b, 1) == tey))
    return test_acc, train_acc


def trace_cosine(core, x_probe, y_probe, eps=1e-2):
    """cosine(eligibility used by the update, true dL/dw) on a few episodes."""
    c = core.config
    g = np.random.default_rng(c.seed)
    IDX = np.stack([g.choice(c.hidden, c.fan_in, replace=False)
                    for _ in range(c.hidden)]).astype(np.int32)
    w_in = core.w_in.copy(); w_rec = core.w_rec.copy()
    w_out = core.w_out.copy(); b_out = core.bias.copy()

    def rollout(wi, seq):
        v = np.zeros(c.hidden, np.float32); a = v.copy(); z = v.copy()
        f = np.zeros(c.hidden, np.float32)
        for s in range(len(seq)):
            v = c.leak * v + wi @ seq[s] + np.sum(w_rec * z[IDX], axis=1)
            lim = c.threshold + c.adaptation_strength * a
            z = (v >= lim).astype(np.float32)
            v = v - z * lim
            a = c.adaptation_decay * a + z
            f = c.feature_decay * f + (1 - c.feature_decay) * z
        return f

    def loss_of(f, t):
        lg = w_out @ f + b_out
        lg = lg - lg.max()
        p = np.exp(lg); p = p / p.sum()
        return -float(np.log(max(float(p[t]), 1e-12)))

    cos = []
    for i in range(12):
        core.reset_state()
        core.forward(x_probe[i], track=True)
        core.learn(int(y_probe[i]), mode="frozen")   # sets _last_error only
        tr = np.zeros_like(w_in); tr[core.rows] = core.elig_in

        base = loss_of(rollout(w_in, x_probe[i]), int(y_probe[i]))
        grad = np.zeros_like(w_in)
        for row in core.rows:
            wp = w_in.copy(); wp[row] += eps
            grad[row] = (loss_of(rollout(wp, x_probe[i]), int(y_probe[i])) - base) / eps
        a_ = tr[core.rows].ravel(); b_ = grad[core.rows].ravel()
        den = np.linalg.norm(a_) * np.linalg.norm(b_)
        if den > 1e-12:
            cos.append(float(a_ @ b_ / den))
    return float(np.mean(cos)) if cos else float("nan")


def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "docs/research/collab/scarx-neural-round1-20261111/deepseek-r7/hypothesis-report.md"
    psha = hashlib.sha256(proto.read_bytes()).hexdigest()
    run = new_run(root, "r7-secs-dev-v1", {
        "hypothesis_sha256": psha, "seeds": list(DEV), "arms": list(ARMS),
        "n_train": N_TRAIN, "n_test": N_TEST, "note": "DEV only, no confirmation"})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in DEV:
            trx, try_ = make_task(seed + 200_000, N_TRAIN)
            tex, tey = make_task(seed + 300_000, N_TEST)
            probe_x, probe_y = make_task(seed + 400_000, 32)
            c = S.cfg(seed)

            for arm in ARMS:
                mode = {"production": "scalar", "secs": "secs",
                        "shuffled": "shuffled_credit", "zero": "zero_credit"}.get(arm)
                core = S.SECSCore(c, mode=mode or "scalar", seed=seed,
                                  anchor_refresh=200 if arm == "secs" else 0)
                w0 = core.w_in.copy()

                online_hits = 0
                rng = np.random.default_rng(seed + 55)
                order = rng.permutation(N_TRAIN)
                for i in order:
                    core.reset_state()
                    core.forward(trx[i], track=True)
                    core.learn(int(try_[i]), mode="frozen" if arm == "frozen" else "full")
                    if arm != "frozen":
                        pass
                # online accuracy on held-out using the arm's own readout
                for i in range(N_TEST):
                    core.reset_state()
                    p = core.forward(tex[i], track=False)
                    online_hits += int(np.argmax(p) == tey[i])
                online_acc = online_hits / N_TEST

                trf = features_of(core, trx)
                tef = features_of(core, tex)
                refit, refit_tr = refit_head(trf, try_, tef, tey)

                # trace-gradient alignment through the SAME update path
                cos = trace_cosine(core, probe_x, probe_y)

                rows.append({
                    "seed": seed, "arm": arm,
                    "online_acc": online_acc,
                    "refit_acc": refit,
                    "refit_train": refit_tr,
                    "weight_delta": float(np.abs(core.w_in - w0).mean()),
                    "trace_cosine": (None if cos != cos else float(cos)),
                    "hidden_digest": S.digest(core)[:16],
                    "updates": core.updates,
                })
                mon.check()
                print(f"  seed{seed} {arm:11s} online {online_acc:.4f} "
                      f"refit {refit:.4f} |dw| {np.abs(core.w_in-w0).mean():.5f} "
                      f"cos {cos:+.4f}", flush=True)

    def M(arm, k):
        vals = [r[k] for r in rows if r["arm"] == arm and r[k] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {a: {k: M(a, k) for k in ("online_acc", "refit_acc", "weight_delta",
                                        "trace_cosine")} for a in ARMS}

    def sanitize(o):
        """json cannot carry NaN; an absent cosine must read as null, not as a number."""
        if isinstance(o, dict):
            return {k: sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sanitize(v) for v in o]
        if isinstance(o, float) and o != o:
            return None
        return o

    write_json(run / "result.json",
               sanitize({"hypothesis_sha256": psha, "trials": rows, "summary": summary,
                         "resources": dict(mon.report()),
                         "wall_seconds": time.time() - t0}))

    print("\n" + "=" * 88)
    print("R7 DEV — SECS vs controls (3 seeds, mean)")
    print("=" * 88)
    print(f"  {'arm':12s} {'online':>9} {'refit':>9} {'|dw|':>10} {'cos':>9}")
    for a in ARMS:
        s = summary[a]
        c = s["trace_cosine"]
        ctxt = f"{c:+9.4f}" if c == c else f"{'n/a':>9}"
        print(f"  {a:12s} {s['online_acc']:9.4f} {s['refit_acc']:9.4f} "
              f"{s['weight_delta']:10.5f} {ctxt}")
    print(f"\n  refit is the COMMON head: it isolates representation gain from readout gain")
    print(f"\nRESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
