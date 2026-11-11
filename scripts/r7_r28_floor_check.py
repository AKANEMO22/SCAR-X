"""R7 round 28 — is the 6.41pp "irreducible floor" real, or is my JOINT under-trained?

THE SELF-CHECK ROUND 27 DID NOT RUN
-----------------------------------
Round 27 concluded the 3pp gate is unreachable because `solo - joint = 6.41pp`, and called it
irreducible. That conclusion is only valid if the JOINT head is at its optimum. My joint run:

    Fj = concat(all three tasks);  minibatch = half of the union, sampled each epoch
    JOINT_EPOCHS = 4800

Three ways that joint could be worse than achievable, each of which would inflate the floor:

  H1 UNDERTRAINED 4800 epochs of half-batch sampling may be too few for 1152 examples.
     Predicts: more joint epochs -> lower joint error -> SMALLER floor.
  H2 SAMPLING     随机 half-batches of the union give a noisy gradient. A full-batch or
     balanced-per-task sampler may converge better.
     Predicts: balanced full-union steps -> SMALLER floor.
  H3 GENUINE      The floor is a real co-representation cost of a 2-logit linear head.
     Predicts: the floor is stable across budgets and samplers.

DISCRIMINATION: sweep joint training from 4800 to 38400 epochs AND a full-batch arm. If the
floor moves materially, round 27's "unreachable" claim is WRONG and must be withdrawn.

This is the check that decides whether round 27's central claim stands. A floor claimed as
irreducible must be shown stable against the most obvious confound — optimisation effort.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = tuple(240001 + 7 * k for k in range(15))
ORDER = ("order_short", "parity3", "sim_xor")
N_TR, N_TE = 384, 256
SOLO_EPOCHS = 1600
JOINT_BUDGETS = (4800, 9600, 19200, 38400)


def sim_xor(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        a = int(rng.integers(2)); b = int(rng.integers(2)); y[i] = a ^ b
        t = int(rng.integers(1, 4))
        x[i, t, 0] += 1.5 * a; x[i, t, 1] += 1.5 * b; x[i, 10, 7] += 1.5
    return x, y


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


def order_short(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        y[i] = int(rng.integers(2))
        first, second = (4, 5) if y[i] == 0 else (5, 4)
        p = int(rng.integers(1, 4))
        x[i, p, first] += 1.5
        x[i, p + 2, second] += 1.5
        x[i, 10, 7] += 1.5
    return x, y


GENS = {"sim_xor": sim_xor, "parity3": parity3, "order_short": order_short}


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def feats(core, X):
    out = []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
        out.append(tot)
    return np.stack(out)


class Head:
    __slots__ = ("mu", "sd", "w", "b")

    def __init__(self, d):
        self.mu = None; self.sd = None
        self.w = np.zeros((2, d)); self.b = np.zeros(2)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 0.05)
        return (F - self.mu) / self.sd

    def step(self, A, y, lr=.5):
        p = softmax(A @ self.w.T + self.b)
        e = -p; e[np.arange(len(y)), y] += 1
        self.w += lr * (e.T @ A) / len(y)
        self.b += lr * e.mean(0)

    def acc(self, A, y):
        return float(np.mean(np.argmax(A @ self.w.T + self.b, 1) == y))


def train_joint(Fj, yj, epochs, mode, rng):
    h = Head(Fj.shape[1])
    Aj = h.z(Fj)
    n = len(yj)
    if mode == "full":
        for _ in range(epochs):
            h.step(Aj, yj)
    elif mode == "half":
        half = max(1, n // 2)
        for _ in range(epochs):
            idx = rng.choice(n, half, replace=False)
            h.step(Aj[idx], yj[idx])
    elif mode == "balanced":
        # draw equal counts from each task's slice
        per = n // 3
        k = max(1, per // 2)
        for _ in range(epochs):
            idx = np.concatenate([
                rng.choice(np.arange(t * per, (t + 1) * per), k, replace=False)
                for t in range(3)])
            h.step(Aj[idx], yj[idx])
    else:
        raise ValueError(mode)
    return h


def main():
    t0 = time.time()
    print("=" * 100)
    print("R7 ROUND 28 — is the irreducible floor stable, or is JOINT under-trained?")
    print("=" * 100)
    print(f"  solo {SOLO_EPOCHS}/task ; joint budgets {JOINT_BUDGETS} ; "
          f"samplers half/full/balanced\n")

    per_seed = []
    for k, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in ORDER:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = feats(core, trx); ytr[tk] = try_
            Fte[tk] = feats(core, tex); yte[tk] = tey

        solo = {}
        for tk in ORDER:
            h = Head(Ftr[tk].shape[1])
            A = h.z(Ftr[tk]); y = ytr[tk]
            for _ in range(SOLO_EPOCHS):
                h.step(A, y)
            solo[tk] = h.acc(h.z(Fte[tk]), yte[tk])

        Fj = np.concatenate([Ftr[tk] for tk in ORDER])
        yj = np.concatenate([ytr[tk] for tk in ORDER])

        rec = {"seed": seed, "solo": solo, "joint": {}}
        for budget in JOINT_BUDGETS:
            for mode in ("half", "full", "balanced"):
                rng = np.random.default_rng(seed + hash((budget, mode)) % 10000)
                h = train_joint(Fj, yj, budget, mode, rng)
                j = {tk: h.acc(h.z(Fte[tk]), yte[tk]) for tk in ORDER}
                irr = np.mean([max(0.0, (solo[tk] - j[tk]) * 100) for tk in ORDER])
                rec["joint"][f"{budget}|{mode}"] = {"per_task": j,
                                                    "irr_loss": float(irr)}
        per_seed.append(rec)
        base = rec["joint"][f"{JOINT_BUDGETS[0]}|half"]["irr_loss"]
        best = min(v["irr_loss"] for v in rec["joint"].values())
        print(f"  {k+1:2d}/15 {seed}  floor@4800half {base:6.2f}  "
              f"best {best:6.2f}", flush=True)

    print("\n" + "=" * 100)
    print("FLOOR vs JOINT TRAINING EFFORT  (loss-based mean over 15 seeds)")
    print("=" * 100)
    print(f"  {'budget':>8} {'half':>9} {'full':>9} {'balanced':>10}")
    table = {}
    for budget in JOINT_BUDGETS:
        row = {}
        for mode in ("half", "full", "balanced"):
            v = float(np.mean([r["joint"][f"{budget}|{mode}"]["irr_loss"]
                               for r in per_seed]))
            row[mode] = v
        table[budget] = row
        print(f"  {budget:>8} {row['half']:9.2f} {row['full']:9.2f} "
              f"{row['balanced']:10.2f}")

    base = table[JOINT_BUDGETS[0]]["half"]
    best = min(min(r.values()) for r in table.values())
    print(f"\n  round 27 reported floor : {base:.2f}pp  (4800, half-batch)")
    print(f"  best floor in this sweep: {best:.2f}pp")
    print(f"  reduction from effort   : {base - best:+.2f}pp")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    drop = base - best
    if drop >= 1.5:
        print(f"  => ROUND 27'S CLAIM IS WRONG. The floor falls {drop:.2f}pp with more joint")
        print(f"     training or a better sampler, so it was partly an UNDER-TRAINED JOINT,")
        print(f"     not an irreducible co-representation cost. The '3pp gate is")
        print(f"     unreachable' conclusion is WITHDRAWN.")
        verdict = "floor_was_undertrained"
    elif best >= 3.0:
        print(f"  => The floor is stable at {best:.2f}pp across {len(JOINT_BUDGETS)} budgets")
        print(f"     and 3 samplers, and remains above the 3pp gate. Round 27's")
        print(f"     unreachability claim STANDS.")
        verdict = "floor_stable_above_gate"
    else:
        print(f"  => The floor is stable but falls BELOW 3pp at best ({best:.2f}pp), so the")
        print(f"     gate is reachable in principle. Round 27's conclusion is revised.")
        verdict = "floor_below_gate"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "floor-stability.json").write_text(json.dumps({
        "seeds": list(SEEDS), "order": list(ORDER),
        "joint_budgets": list(JOINT_BUDGETS), "samplers": ["half", "full", "balanced"],
        "table": {str(b): r for b, r in table.items()},
        "round27_floor": base, "best_floor": best, "reduction_pp": drop,
        "per_seed": per_seed, "verdict": verdict,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: floor-stability.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
