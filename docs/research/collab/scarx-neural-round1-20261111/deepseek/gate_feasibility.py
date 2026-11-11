"""Is the proposed gate satisfiable?

The second analysis proposes: "the test must be a task where a FROZEN head is
still below 65%". That condition is the whole ballgame, because it decides
whether adding a slow trace + per-neuron error can ever show anything.

This script measures, for every task in the campaign's suite, the BEST accuracy
achievable by a FROZEN hidden with a FIRST-ORDER readout (the class that every
proposal so far has used: single decay, decay sweep, multiscale bank, uniform
sum). If every task is already above 65% frozen, the proposed gate is
unsatisfiable and the ordering collapses.

Both samplers are measured, because grok-001 established that they differ.
Frozen weights, no synapse trained.
"""
import numpy as np

from scarx.evaluation.binding_audit21 import strict_binding
from scarx.evaluation.pair_binding import pair_binding
from scarx.evaluation.tasks21 import representation, ridge_probe
from scarx.neural.core import Config, SpikingCore
from scarx.neural.plasticity import PlasticCore

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
DECAYS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def cfg(seed=41001):
    return Config(seed=seed, **CFG)


def bernoulli_train(core, seq, seed=901):
    """Stochastic path: what features_many / PlasticCore use."""
    c = core.config
    x = np.asarray(seq, np.float32)
    b, t, _ = x.shape
    rng = np.random.default_rng(seed)
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy()
    out = []
    for s in range(t):
        v = c.leak * v + x[:, s] @ core.w_in.T + np.sum(core.w_rec[None, :, :] * z[:, core.indices], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        p = 1 / (1 + np.exp(-np.clip((v - lim) / .4, -20, 20)))
        z = (rng.random((b, c.hidden)) < p).astype(np.float32)
        v = v - z * lim
        a = c.adaptation_decay * a + z
        out.append(z.copy())
    return np.stack(out, 1)


def hard_train(core, seq):
    """Deterministic path: what SpikingCore.step / unified.py use."""
    c = core.config
    x = np.asarray(seq, np.float32)
    b, t, _ = x.shape
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy()
    out = []
    for s in range(t):
        v = c.leak * v + x[:, s] @ core.w_in.T + np.sum(core.w_rec[None, :, :] * z[:, core.indices], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        z = (v >= lim).astype(np.float32)
        v = v - z * lim
        a = c.adaptation_decay * a + z
        out.append(z.copy())
    return np.stack(out, 1)


def integrate(tr, d):
    n, t, h = tr.shape
    f = np.zeros((n, h), np.float32)
    for s in range(t):
        f = d * f + (1 - d) * tr[:, s]
    return f


def best_first_order(tr, tt, fy, ty):
    """Best over every first-order readout family the campaign has tried."""
    cands = {"uniform_sum": (tr.sum(1), tt.sum(1))}
    for d in DECAYS:
        cands[f"decay_{d}"] = (integrate(tr, d), integrate(tt, d))
    bank_f = np.concatenate([integrate(tr, d) for d in (0.1, 0.25, 0.5, 0.75, 0.9)], 1)
    bank_t = np.concatenate([integrate(tt, d) for d in (0.1, 0.25, 0.5, 0.75, 0.9)], 1)
    cands["bank5"] = (bank_f, bank_t)
    scores = {k: ridge_probe(a, fy, b, ty) for k, (a, b) in cands.items()}
    best = max(scores.values())
    winner = max(scores, key=scores.get)
    return best, winner, scores


def get_task(name, seed, count):
    if name == "order":
        return representation("order", seed, count)
    if name == "xor":
        return representation("xor", seed, count)
    if name == "pair_binding":
        return pair_binding(seed, count)
    if name == "strict_binding":
        return strict_binding(seed, count)
    raise ValueError(name)


def main():
    tasks = ("order", "xor", "pair_binding", "strict_binding")
    print("=" * 78)
    print("BEST FROZEN FIRST-ORDER READOUT per task (the gate every proposal must clear)")
    print("gate = 0.65.  If a task is already >0.65 frozen, no learning can be shown on it.")
    print("=" * 78)

    for sampler, fn in (("bernoulli", lambda c, s: bernoulli_train(c, s)),
                        ("hard-threshold", hard_train)):
        print(f"\n--- {sampler} sampler ---")
        print(f"  {'task':16s} {'best':>7s} {'winner':>14s} {'above65?':>9s}")
        for task in tasks:
            core = PlasticCore(cfg(), "frozen")
            fx, fy = get_task(task, 31001, 400)
            tx, ty = get_task(task, 31901, 400)
            tr = fn(core, fx)
            tt = fn(core, tx)
            best, winner, _ = best_first_order(tr, tt, fy, ty)
            flag = "YES" if best >= .65 else "no"
            print(f"  {task:16s} {best:7.4f} {winner:>14s} {flag:>9s}")

    print("\n" + "=" * 78)
    print("CONCLUSION INPUT")
    print("=" * 78)
    print("A task qualifies as a learning gate only if best_frozen < 0.65 AND some")
    print("local rule can push it above. Count how many rows say 'no'.")


if __name__ == "__main__":
    main()
