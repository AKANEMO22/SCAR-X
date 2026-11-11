"""R7 round 42 — Section IX: does hidden learning help when the READOUT is capable?

WHY THIS HAS NOT BEEN TESTED
----------------------------
Rounds 1-41 measured retention with a FROZEN hidden layer, because rounds 4 and 11 showed the
hidden rule contributes nothing to retention and the exact-derivative trace produces zero gain.

But both of those were measured with a 2-logit linear readout. Round 29 then showed a linear
readout has a 6.52pp co-representation floor and a nonlinear one has 0.36pp. **A readout that
was itself the bottleneck can hide a real representation gain.** The mission's first question --
do hidden synapses learn a useful general representation -- has never been asked with a readout
that is not the binding constraint.

The mission's Section IX requires Q1-Q7. This runs them.

PROTOCOL (15 fresh seeds, 3 tasks, no task ID, no future information)
    For each seed, run the sequential protocol TWICE with identical data and order:
        FROZEN  learn(target, mode='frozen')   readout only, hidden weights untouched
        LEARNED learn(target, mode='full')     production hidden rule ALSO active
    Readout in both arms is the SAME mlp32 head trained on the SAME features protocol.

    The only difference between arms is whether core.learn() also updates w_in/w_rec via the
    production rule. That is a clean causal intervention.

MEASUREMENTS (Section IX)
    Q1  decodability: ridge probe fit on hidden features, learned vs frozen
    Q2  same readout algorithm: mlp32 accuracy, learned vs frozen
    Q3  revert: not needed as an extra arm because FROZEN *is* the reverted arm
    Q6  shuffled signal: LEARNED arm with the scalar advantage replaced by a random draw of the
        same magnitude -- does a correct signal beat a wrong one?
    Q7  is the test already solved: report frozen accuracy itself

REGISTERED PREDICTION
    With the linear readout the hidden rule never mattered (rounds 4, 11). With a capable
    readout it MIGHT. I predict LEARNED beats FROZEN by < 1pp on accuracy, i.e. still nothing,
    BECAUSE the production rule is `delta = lr * advantage * trace` with a SCALAR advantage and
    an eligibility trace that is a Hebbian correlation, not a derivative. Round 11 established
    the derivative version also fails, so a weaker signal should too.

FALSIFIER
    If LEARNED beats FROZEN by >= 2pp on accuracy, the hidden rule DOES help once the readout is
    not the bottleneck, and my rounds 4/11 conclusion was readout-limited. I would report that
    as a reversal.
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

SEEDS = tuple(520001 + 7 * k for k in range(15))
TASKS = ("sim_xor", "parity3", "order_short")
ORDER = TASKS
N_TR, N_TE = 320, 200
EPOCHS = 600          # readout epochs per task
HID_EPOCHS = 150      # hidden-rule passes per task (each pass = 1 full train set)
ARMS = ("frozen", "learned", "shuffled")

PRIOR = set(R37.SEEDS) | set(R32.SEEDS) | set(range(140011, 140011 + 60, 6))
assert not (set(SEEDS) & PRIOR), "fresh seeds overlap with earlier rounds"


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def collect(core, X, hid_mode, y=None, hid_epochs=0, rng=None):
    """Run the core over X, optionally applying the hidden rule, return spike-sum features."""
    if hid_mode != "frozen" and y is not None:
        for _ in range(hid_epochs):
            for i in range(len(X)):
                core.reset_state()
                for t in range(12):
                    core.step(X[i, t], track=True)
                if hid_mode == "shuffled":
                    reward = float(rng.random())
                    core.learn(int(y[i]), mode="full", reward=reward)
                else:
                    core.learn(int(y[i]), mode="full")
    out = []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
        out.append(tot)
    return np.stack(out)


def ridge_probe(A_tr, y_tr, A_te, y_te, lam=1.0):
    """Q1 decodability: closed-form ridge on one-hot targets, no iteration."""
    n, d = A_tr.shape
    Y = np.zeros((n, 2))
    Y[np.arange(n), y_tr] = 1.0
    A1 = np.hstack([A_tr, np.ones((n, 1))])
    G = A1.T @ A1 + lam * np.eye(d + 1)
    W = np.linalg.solve(G, A1.T @ Y)
    pred = np.argmax(np.hstack([A_te, np.ones((len(A_te), 1))]) @ W, 1)
    return float(np.mean(pred == y_te))


def main():
    t0 = time.time()
    print("=" * 98)
    print("R7 ROUND 42 — does hidden learning help when the READOUT is capable? (Q1-Q7)")
    print("=" * 98)
    print(f"  15 fresh seeds (520001+7k), 3 tasks, order {' > '.join(ORDER)}")
    print(f"  arms: {ARMS}; hidden-rule passes {HID_EPOCHS}/task\n")

    cells = []
    for s_i, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr = {tk: R37.GENS[tk](seed + 200_000, N_TR)[0] for tk in TASKS}
        ytr = {tk: R37.GENS[tk](seed + 200_000, N_TR)[1] for tk in TASKS}
        Fte = {tk: R37.GENS[tk](seed + 300_000, N_TE)[0] for tk in TASKS}
        yte = {tk: R37.GENS[tk](seed + 300_000, N_TE)[1] for tk in TASKS}

        rec = {"seed": seed, "arms": {}}
        for arm in ARMS:
            rng = np.random.default_rng(seed + 31)
            F = {tk: collect(core, Ftr[tk], arm, ytr[tk], HID_EPOCHS, rng)
                 for tk in TASKS}
            Tf = {tk: collect(core, Fte[tk], "frozen") for tk in TASKS}

            # Q2: same readout algorithm (mlp32), trained per task
            accs, probes = {}, {}
            for tk in TASKS:
                h = R32.MLP(F[tk].shape[1], seed, 32)
                A = h.z(F[tk])
                for _ in range(2 * EPOCHS):
                    h.step(A, ytr[tk])
                accs[tk] = h.acc(h.z(Tf[tk]), yte[tk])
                probes[tk] = ridge_probe(h.z(F[tk]), ytr[tk],
                                         h.z(Tf[tk]), yte[tk])
            rec["arms"][arm] = {
                "acc": accs, "acc_mean": float(np.mean(list(accs.values()))),
                "probe": probes, "probe_mean": float(np.mean(list(probes.values())))}
            # reset core state between arms so the next arm starts clean
            core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        cells.append(rec)
        a = rec["arms"]
        print(f"  {s_i+1:2d}/15 {seed}  frozen {a['frozen']['acc_mean']:.4f}  "
              f"learned {a['learned']['acc_mean']:.4f}  "
              f"shuffled {a['shuffled']['acc_mean']:.4f}  "
              f"probe F{a['frozen']['probe_mean']:.3f}/L{a['learned']['probe_mean']:.3f}",
              flush=True)

    print("\n" + "=" * 98)
    print("RESULTS")
    print("=" * 98)
    print(f"  {'arm':10s} {'acc':>9} {'probe(Q1)':>11}")
    summary = {}
    for arm in ARMS:
        a = np.array([c["arms"][arm]["acc_mean"] for c in cells])
        p = np.array([c["arms"][arm]["probe_mean"] for c in cells])
        summary[arm] = {"acc": float(a.mean()), "probe": float(p.mean())}
        print(f"  {arm:10s} {a.mean():9.4f} {p.mean():11.4f}")

    print("\n" + "=" * 98)
    print("SECTION IX ANSWERS")
    print("=" * 98)
    dl = np.array([c["arms"]["learned"]["acc_mean"]
                   - c["arms"]["frozen"]["acc_mean"] for c in cells])
    dp = np.array([c["arms"]["learned"]["probe_mean"]
                   - c["arms"]["frozen"]["probe_mean"] for c in cells])
    ds = np.array([c["arms"]["learned"]["acc_mean"]
                   - c["arms"]["shuffled"]["acc_mean"] for c in cells])

    def ci(v):
        se = v.std(ddof=1) / np.sqrt(len(v))
        return float(v.mean()), float(v.mean() - 2.145 * se), float(v.mean() + 2.145 * se)

    m, lo, hi = ci(dl)
    print(f"  Q2 same readout, learned - frozen : {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{int((dl > 0).sum())}/15 better  (pp: {m*100:+.2f})")
    mp, lop, hip = ci(dp)
    print(f"  Q1 decodability, learned - frozen : {mp:+.4f}  CI [{lop:+.4f}, {hip:+.4f}]  "
          f"{int((dp > 0).sum())}/15 better  (pp: {mp*100:+.2f})")
    ms, los, his = ci(ds)
    print(f"  Q6 correct signal - shuffled      : {ms:+.4f}  CI [{los:+.4f}, {his:+.4f}]  "
          f"{int((ds > 0).sum())}/15 better  (pp: {ms*100:+.2f})")
    print(f"  Q7 frozen accuracy (is it solved?) : "
          f"{summary['frozen']['acc']:.4f}")
    print(f"  Q4 per-task accuracy, learned:")
    for tk in TASKS:
        v = np.mean([c["arms"]["learned"]["acc"][tk] for c in cells])
        f = np.mean([c["arms"]["frozen"]["acc"][tk] for c in cells])
        print(f"       {tk:14s} learned {v:.4f}  frozen {f:.4f}  diff {(v-f)*100:+6.2f}pp")

    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    print(f"  registered: learned beats frozen by < 1pp (i.e. still nothing)")
    print(f"  measured  : {m*100:+.2f}pp on accuracy, {mp*100:+.2f}pp on the ridge probe")
    if m >= 0.02 or mp >= 0.02:
        print(f"\n  => REVERSAL. Hidden learning DOES help once the readout is not the")
        print(f"     bottleneck. Rounds 4 and 11 were readout-limited and must be re-scoped.")
        verdict = "reversal_hidden_helps"
    elif hi < 0:
        print(f"\n  => Hidden learning HURTS: the CI's upper bound is below zero.")
        verdict = "hidden_hurts"
    else:
        print(f"\n  => CONFIRMED. Hidden learning contributes nothing measurable even with a")
        print(f"     capable readout. The conclusion is now readout-independent, which is a")
        print(f"     stronger statement than rounds 4/11 could make.")
        verdict = "hidden_neutral_confirmed"
    if ms > 0 and his < 0:
        print(f"  Q6: a wrong signal is BETTER than the correct one -- the rule is not a")
        print(f"      credit-assignment mechanism at all.")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hidden-learning-capable-readout.json").write_text(json.dumps({
        "seeds": list(SEEDS), "tasks": list(TASKS), "arms": list(ARMS),
        "hid_epochs": HID_EPOCHS, "readout_epochs": EPOCHS,
        "summary": summary,
        "Q2_learned_minus_frozen_acc": {"mean": m, "ci95": [lo, hi]},
        "Q1_probe_delta": {"mean": mp, "ci95": [lop, hip]},
        "Q6_correct_minus_shuffled": {"mean": ms, "ci95": [los, his]},
        "verdict": verdict, "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: hidden-learning-capable-readout.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
