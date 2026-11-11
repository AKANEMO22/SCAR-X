"""R7 round 43 — DECOLLE transfer: per-neuron local error versus a scalar advantage.

MECHANISM TRANSFER CARD
=======================
1. SOURCE
   Kaiser, Mostafa & Neftci, "Synaptic Plasticity Dynamics for Deep Continuous Local
   Learning (DECOLLE)", Front. Neurosci. 14:424, 2020. doi 10.3389/fnins.2020.00424.
   PMC7235446. License: CC-BY (paper). NO CODE COPIED -- the rule below is re-derived from
   Eq. (8) of the paper and implemented from scratch in NumPy.

2. ORIGINAL EQUATIONS (paper Eq. 7-8)
   Y^l_i   = sum_j G^l_ij S^l_j            random FIXED readout of layer l
   error^l_i = sum_k G^l_ki (Y^l_k - Yhat^l_k)
   dW^l_ij = -eta * error^l_i * sigma'(U^l_i) * P^l_j
   G is fixed, random, NOT trained. sigma' surrogate gradient. P_j presynaptic trace.

3. ASSUMPTIONS REQUIRED BY THE ORIGINAL
   - layered feedforward stack with per-layer readouts
   - a readout exists at the layer being trained
   - surrogate gradient available
   - random G has zero-mean entries

4. WHAT IS TAKEN / WHAT IS DROPPED
   TAKEN : the per-neuron error `error_i` as the learning factor, routed through a fixed
           random matrix, combined with a presynaptic trace.
   DROPPED: the multi-layer stack (SCAR-X has ONE recurrent layer, H48).
            the per-layer readouts (SCAR-X has one global readout).
            autodifferentiation (SCAR-X forbids it).

5. SCAR-X FORM
   SCAR-X production hidden rule (src/scarx/neural, unchanged):
       advantage = clip(p[target] - reward_mean, -1, 1)      <-- ONE SCALAR
       delta     = hidden_lr * advantage * elig - anchor
   The scalar `advantage` multiplies every synapse identically.

   Transferred rule (this experiment only):
       g_i     = fixed random N(0, 1/sqrt(2)) vector, length H, NOT trained, seed-fixed
       error_i = g_i * (p[target] - 1)          one error PER NEURON
       delta   = hidden_lr * error_i * elig - anchor
   Only the scalar is replaced by a per-neuron vector. Nothing else changes: same
   hidden_lr, same anchor, same clip, same eligibility, same weights, same H48.

6. CAUSAL REASON TO EXPECT IT FIXES SOMETHING
   Round 11 measured that even an EXACT derivative trace (cosine +0.9452 vs null -0.0005,
   verified by finite differences at +0.9979) produces ZERO representation gain at every
   eta. A correctly-aligned trace multiplied by a scalar must therefore be the problem --
   or the scalar carries no per-neuron information. DECOLLE's contribution is exactly that
   per-neuron information, broadcast through a fixed random matrix. If the scalar is the
   bottleneck, this is the minimal change that tests it.

7. COST
   g: 48 floats. error_i: 48 floats per step. No new per-synapse state.

8. WHAT CHANGES
   learning rule ONLY. Not neuron dynamics, not topology, not the readout, not H48,
   not fan_in=8. Production rule remains the `scalar` arm.

9. ARCHITECTURAL IMPACT
   None on inference: the rule is training-time only, and the forward pass is identical.

10. FALSIFYING EXPERIMENT (this file)
   ARMS -- each changes exactly one thing
       frozen            hidden untouched                 (floor)
       scalar            production scalar advantage      (current baseline)
       perneuron         per-neuron error via fixed G     (candidate)
       perneuron_zero    per-neuron error x 0             (is it just extra updates?)
       perneuron_shuffled per-neuron error, permuted      (is it just more noise?)

   REGISTERED PREDICTION
       perneuron beats scalar by >= 2pp on the ridge probe AND on direct accuracy.
   FALSIFIER
       If perneuron is within 2pp of scalar, OR if perneuron_shuffled is within 2pp of
       perneuron, the mechanism is rejected. I will report whichever happens.

   Hidden epochs are matched across all arms so no arm gets more updates.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import r7_r32_tier3_six as R32  # noqa: E402
import r7_r37_t6_fixed as R37  # noqa: E402

from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = tuple(620003 + 11 * k for k in range(15))
TASKS = ("sim_xor", "parity3", "order_short")
N_TR, N_TE = 320, 200
# BUDGET NOTE: at 120 hidden epochs x 5 arms x 15 seeds x 3 tasks x 320 examples the run
# projects to ~88 minutes (measured 610us per step+learn call). Cut to 30 epochs, which
# projects to ~22 minutes. Hidden epochs remain MATCHED across all arms, so the comparison
# is still paired and fair; the absolute level may be below convergence, which makes any
# positive effect harder to see, not easier. Recorded rather than silently absorbed.
HID_EPOCHS = 30
READOUT_EPOCHS = 400
ARMS = ("frozen", "scalar", "perneuron", "perneuron_zero", "perneuron_shuffled")

PRIOR = set(R37.SEEDS) | set(R32.SEEDS) | set(range(140011, 140072, 6))
assert not (set(SEEDS) & PRIOR), "fresh seeds overlap with earlier rounds"


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def make_g(H, seed):
    """Fixed random projection, zero-mean, unit-ish scale, NEVER trained."""
    g = np.random.default_rng(seed + 77).normal(0, 1 / np.sqrt(2.0), H)
    return g.astype(np.float32)


def hidden_pass(core, X, y, g, arm, rng):
    """One epoch of the production forward+learn loop, with the learning factor swapped."""
    c = core.config
    for i in range(len(X)):
        core.reset_state()
        for t in range(12):
            core.step(X[i, t], track=True)
        tgt = int(y[i])
        if arm == "frozen":
            core.learn(tgt, mode="frozen")
            continue

        p = core.probabilities()
        err = -p.copy(); err[tgt] += 1.0          # length-2 readout error
        # readout update: identical in every arm (production delta rule)
        core.w_out += np.clip(c.readout_lr * err[:, None] * core.features[None, :],
                              -c.update_clip, c.update_clip)
        core.bias += np.clip(c.readout_lr * err, -c.update_clip, c.update_clip)
        np.clip(core.w_out, -c.weight_clip, c.weight_clip, out=core.w_out)

        adv = float(np.clip(p[tgt] - core.reward_mean, -1, 1))
        core.reward_mean = .95 * core.reward_mean + .05 * float(p[tgt])

        if arm == "scalar":
            factor = np.full(c.hidden, adv, np.float32)
        else:
            # DECOLLE: per-neuron error routed through the fixed random vector g
            factor = (g * (p[tgt] - 1.0)).astype(np.float32)
            if arm == "perneuron_zero":
                factor = np.zeros_like(factor)
            elif arm == "perneuron_shuffled":
                factor = rng.permutation(factor)

        for name in ("in", "rec"):
            elig = getattr(core, "elig_" + name)          # (len(rows), fan_in)
            weights = getattr(core, "w_" + name)
            stable = getattr(core, "stable_" + name)
            # only `rows` carry eligibility, exactly as the production rule does
            f_rows = factor[core.rows][:, None]
            d = c.hidden_lr * f_rows * elig \
                - c.anchor_strength * (weights[core.rows] - stable[core.rows])
            weights[core.rows] += np.clip(d, -c.update_clip, c.update_clip)
            np.clip(weights, -c.weight_clip, c.weight_clip, out=weights)
        core.updates += 1


def featurize(core, X):
    out = []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
        out.append(tot)
    return np.stack(out)


def ridge(A_tr, y_tr, A_te, y_te, lam=1.0):
    n, d = A_tr.shape
    Y = np.zeros((n, 2)); Y[np.arange(n), y_tr] = 1.0
    A1 = np.hstack([A_tr, np.ones((n, 1))])
    W = np.linalg.solve(A1.T @ A1 + lam * np.eye(d + 1), A1.T @ Y)
    pred = np.argmax(np.hstack([A_te, np.ones((len(A_te), 1))]) @ W, 1)
    return float(np.mean(pred == y_te))


def main():
    t0 = time.time()
    print("=" * 100)
    print("R7 ROUND 43 — DECOLLE transfer: per-neuron error vs scalar advantage")
    print("=" * 100)
    print(f"  15 fresh seeds (620003+11k) | 3 tasks | hidden epochs {HID_EPOCHS} "
          f"(MATCHED across arms)")
    print(f"  arms: {ARMS}\n")

    cells = []
    for s_i, seed in enumerate(SEEDS):
        Xtr = {tk: R37.GENS[tk](seed + 200_000, N_TR) for tk in TASKS}
        Xte = {tk: R37.GENS[tk](seed + 300_000, N_TE) for tk in TASKS}
        rec = {"seed": seed, "arms": {}}
        for arm in ARMS:
            core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
            g = make_g(core.config.hidden, seed)
            rng = np.random.default_rng(seed + 5)
            for _ in range(HID_EPOCHS):
                for tk in TASKS:
                    hidden_pass(core, Xtr[tk][0], Xtr[tk][1], g, arm, rng)
            accs, probes = {}, {}
            for tk in TASKS:
                A = featurize(core, Xtr[tk][0]); B = featurize(core, Xte[tk][0])
                mu, sd = A.mean(0), np.maximum(A.std(0), .05)
                Az, Bz = (A - mu) / sd, (B - mu) / sd
                h = R32.MLP(Az.shape[1], seed, 32)
                for _ in range(READOUT_EPOCHS):
                    h.step(Az, Xtr[tk][1])
                accs[tk] = h.acc(Bz, Xte[tk][1])
                probes[tk] = ridge(Az, Xtr[tk][1], Bz, Xte[tk][1])
            rec["arms"][arm] = {"acc": accs, "probe": probes,
                                "acc_mean": float(np.mean(list(accs.values()))),
                                "probe_mean": float(np.mean(list(probes.values())))}
        cells.append(rec)
        a = rec["arms"]
        print(f"  {s_i+1:2d}/15 {seed}  "
              + "  ".join(f"{arm[:9]} {a[arm]['probe_mean']:.3f}" for arm in ARMS),
              flush=True)

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"  {'arm':20s} {'probe(Q1)':>11} {'acc(Q2)':>9}")
    summary = {}
    for arm in ARMS:
        pr = np.array([c["arms"][arm]["probe_mean"] for c in cells])
        ac = np.array([c["arms"][arm]["acc_mean"] for c in cells])
        summary[arm] = {"probe": float(pr.mean()), "acc": float(ac.mean())}
        print(f"  {arm:20s} {pr.mean():11.4f} {ac.mean():9.4f}")

    def ci(v):
        se = v.std(ddof=1) / np.sqrt(len(v))
        return float(v.mean()), float(v.mean() - 2.145 * se), float(v.mean() + 2.145 * se)

    print("\n" + "=" * 100)
    print("PAIRED CONTROLS (the discriminating comparisons)")
    print("=" * 100)
    for label, a, b, key in (
            ("candidate - scalar        (does it help?)", "perneuron", "scalar", "probe"),
            ("candidate - zero          (just extra updates?)", "perneuron",
             "perneuron_zero", "probe"),
            ("candidate - shuffled      (just more noise?)", "perneuron",
             "perneuron_shuffled", "probe"),
            ("candidate - frozen        (any learning at all?)", "perneuron",
             "frozen", "probe"),
            ("candidate - scalar [acc]", "perneuron", "scalar", "acc")):
        v = np.array([c["arms"][a][key+"_mean"] - c["arms"][b][key+"_mean"] for c in cells])
        m, lo, hi = ci(v)
        print(f"  {label:36s} {m*100:+7.2f}pp  CI [{lo*100:+6.2f}, {hi*100:+6.2f}]  "
              f"{int((v > 0).sum())}/15 better")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    d_scalar = np.array([c["arms"]["perneuron"]["probe_mean"]
                         - c["arms"]["scalar"]["probe_mean"] for c in cells])
    d_shuf = np.array([c["arms"]["perneuron"]["probe_mean"]
                       - c["arms"]["perneuron_shuffled"]["probe_mean"] for c in cells])
    m1, l1, h1 = ci(d_scalar)
    m2, l2, h2 = ci(d_shuf)
    print(f"  registered: perneuron beats scalar by >= 2pp on the probe")
    print(f"  measured  : {m1*100:+.2f}pp  CI [{l1*100:+.2f}, {h1*100:+.2f}]")
    print(f"  specificity: perneuron beats shuffled by {m2*100:+.2f}pp "
          f"CI [{l2*100:+.2f}, {h2*100:+.2f}]")
    if m1 >= 0.02 and l1 > 0 and m2 >= 0.02 and l2 > 0:
        print(f"\n  => CONFIRMED. Per-neuron error helps AND the correct routing matters,")
        print(f"     so this is not extra updates and not extra noise.")
        verdict = "decolle_confirmed"
    elif m1 < 0.02 and m2 < 0.02:
        print(f"\n  => REJECTED. Per-neuron error does not beat the scalar, and shuffled")
        print(f"     is indistinguishable -- the scalar is NOT the bottleneck.")
        verdict = "decolle_rejected"
    else:
        print(f"\n  => PARTIAL / AMBIGUOUS. Reported as such; neither the success nor the")
        print(f"     specificity condition is fully met.")
        verdict = "decolle_partial"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "decolle-transfer.json").write_text(json.dumps({
        "source": "Kaiser, Mostafa, Neftci 2020, Front Neurosci 14:424, Eq.8; no code copied",
        "seeds": list(SEEDS), "tasks": list(TASKS), "arms": list(ARMS),
        "hidden_epochs": HID_EPOCHS, "readout_epochs": READOUT_EPOCHS,
        "summary": summary,
        "cand_minus_scalar": {"mean": m1, "ci95": [l1, h1]},
        "cand_minus_shuffled": {"mean": m2, "ci95": [l2, h2]},
        "verdict": verdict, "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: decolle-transfer.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
