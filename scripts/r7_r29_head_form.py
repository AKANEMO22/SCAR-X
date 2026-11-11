"""R7 round 29 — is the 6.23pp floor a property of the HEAD or of the FEATURES?

THE QUESTION THAT DECIDES THE SCOPE ISSUE WITH DATA
---------------------------------------------------
Round 28 established `solo - joint = 6.23pp` for a **2-logit linear head**, stable across 12
joint configurations. Round 27 called it "irreducible" and concluded the 3pp gate is
unreachable. But that word is doing too much work: irreducible *for which head?*

Round 5b measured a 32-unit tanh head at **4.30pp mean forgetting**, which is BELOW the 6.23pp
linear floor. That means the floor is not a property of the frozen features alone — a different
head has a different floor. If so, "unreachable" is a statement about my head, not about SCAR-X.

MEASUREMENT
    For each head form, measure the same decomposition on the same 15 seeds:
        solo_j   head trained on task j alone
        joint    head trained on the union, budget-matched (3x solo)
        floor_j  = max(0, solo_j - joint_j)   loss-based, as the mission's gate means

    HEADS (all trained on the SAME frozen spike-sum features; no new neurons, no new
    representation, only the readout's form changes)
        linear   2-logit, 48 params              <- round 27/28's floor
        wide     2 x 16 averaged blocks          linear but more parameters
        mlp32    one tanh layer, 32 units         nonlinear

    DIAGNOSTIC ONLY. None of these is proposed as a candidate. The brief forbids expanding
    capacity to mask a learning-rule failure, and a nonlinear head used to CLAIM a working
    learner would be exactly that. This measures where the floor comes from.

DISCRIMINATION
    If mlp32 floor << 6.23pp  -> the floor is HEAD-EXPRESSIVENESS, not features. The 3pp gate
       is unreachable only for a 2-logit linear readout, and the scope question becomes
       concrete: a nonlinear readout would need authorisation.
    If mlp32 floor ~ 6.23pp   -> the floor is in the FEATURES. No readout form can reach 3pp,
       and the gate is dead for this benchmark at H48/spike-sum, independently of the head.

Both outcomes are decisive and neither is tuned.
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
SOLO = 1600
JOINT = 3 * SOLO


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
    one = z.ndim == 1
    if one:
        z = z[None, :]
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    o = e / e.sum(1, keepdims=True)
    return o[0] if one else o


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


# --------------------------------------------------------------- heads
class Lin:
    """2-logit linear. The configuration rounds 27/28 measured."""
    name = "linear"

    def __init__(self, d, seed=0):
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


class Wide:
    """Wider LINEAR head: (2K, d) trained, logits average the K blocks per class."""
    name = "wide16"

    def __init__(self, d, seed=0, K=16):
        self.K = K
        self.mu = None; self.sd = None
        g = np.random.default_rng(seed + 11)
        self.w = g.normal(0, .01, (2 * K, d)); self.b = np.zeros(2 * K)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 0.05)
        return (F - self.mu) / self.sd

    def logits(self, A):
        lg = A @ self.w.T + self.b
        return lg.reshape(len(A), 2, self.K).mean(2)

    def step(self, A, y, lr=.5):
        n = len(y); K = self.K
        tgt = np.zeros((n, 2 * K))
        for k in range(K):
            tgt[np.arange(n), y * K + k] = 1.0
        lg = A @ self.w.T + self.b
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        e = tgt - p
        self.w += lr * (e.T @ A) / n
        self.b += lr * e.mean(0)

    def acc(self, A, y):
        return float(np.mean(np.argmax(self.logits(A), 1) == y))


class MLP:
    """One tanh hidden layer. DIAGNOSTIC ceiling proxy, never a candidate."""
    name = "mlp32"

    def __init__(self, d, seed=0, H=32):
        self.mu = None; self.sd = None
        g = np.random.default_rng(seed + 21)
        self.W1 = g.normal(0, 1 / np.sqrt(d), (H, d)); self.b1 = np.zeros(H)
        self.W2 = g.normal(0, 1 / np.sqrt(H), (2, H)); self.b2 = np.zeros(2)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 0.05)
        return (F - self.mu) / self.sd

    def fwd(self, A):
        Hd = np.tanh(A @ self.W1.T + self.b1)
        return Hd, Hd @ self.W2.T + self.b2

    def step(self, A, y, lr=.1):
        n = len(y)
        Hd, lg = self.fwd(A)
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        e = -p; e[np.arange(n), y] += 1
        gW2 = (e.T @ Hd) / n; gb2 = e.mean(0)
        eH = (e @ self.W2) * (1 - Hd ** 2)
        self.W2 += lr * gW2; self.b2 += lr * gb2
        self.W1 += lr * (eH.T @ A) / n; self.b1 += lr * eH.mean(0)

    def acc(self, A, y):
        _, lg = self.fwd(A)
        return float(np.mean(np.argmax(lg, 1) == y))


MAKERS = {"linear": lambda d, s: Lin(d, s),
          "wide16": lambda d, s: Wide(d, s, 16),
          "mlp32": lambda d, s: MLP(d, s, 32)}


def main():
    t0 = time.time()
    print("=" * 98)
    print("R7 ROUND 29 — is the floor HEAD-EXPRESSIVENESS or FEATURE-limited?")
    print("=" * 98)
    print(f"  same frozen features, same 15 seeds, same budgets ({SOLO} solo / {JOINT} joint)")
    print(f"  DIAGNOSTIC ONLY — no head here is proposed as a candidate\n")

    rows = []
    for k, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in ORDER:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = feats(core, trx); ytr[tk] = try_
            Fte[tk] = feats(core, tex); yte[tk] = tey
        Fj = np.concatenate([Ftr[tk] for tk in ORDER])
        yj = np.concatenate([ytr[tk] for tk in ORDER])

        rec = {"seed": seed, "heads": {}}
        for name, mk in MAKERS.items():
            solo = {}
            for tk in ORDER:
                h = mk(Ftr[tk].shape[1], seed)
                A = h.z(Ftr[tk])
                for _ in range(SOLO):
                    h.step(A, ytr[tk])
                solo[tk] = h.acc(h.z(Fte[tk]), yte[tk])

            hj = mk(Fj.shape[1], seed)
            Aj = hj.z(Fj)
            rng = np.random.default_rng(seed + 5)
            n = len(yj); half = max(1, n // 2)
            for _ in range(JOINT):
                idx = rng.choice(n, half, replace=False)
                hj.step(Aj[idx], yj[idx])
            joint = {tk: hj.acc(hj.z(Fte[tk]), yte[tk]) for tk in ORDER}

            per = {tk: max(0.0, (solo[tk] - joint[tk]) * 100) for tk in ORDER}
            rec["heads"][name] = {"solo": solo, "joint": joint, "floor_per_task": per,
                                  "floor": float(np.mean(list(per.values())))}
        rows.append(rec)
        print(f"  {k+1:2d}/15 {seed}  "
              + "  ".join(f"{nm} {rec['heads'][nm]['floor']:6.2f}"
                          for nm in MAKERS), flush=True)

    print("\n" + "=" * 98)
    print("FLOOR BY HEAD FORM (loss-based, mean over 15 seeds)")
    print("=" * 98)
    print(f"  {'head':10s} {'floor':>9} {'sd':>7} {'min':>8} {'max':>8} "
          f"{'seeds<=3pp':>11} {'params':>8}")
    summary = {}
    for name in MAKERS:
        v = np.array([r["heads"][name]["floor"] for r in rows])
        d = 48
        params = {"linear": 2 * d + 2, "wide16": 2 * 16 * (d + 1),
                  "mlp32": 32 * d + 32 + 2 * 32 + 2}[name]
        summary[name] = {"floor": float(v.mean()), "sd": float(v.std(ddof=1)),
                         "min": float(v.min()), "max": float(v.max()),
                         "n_at_3": int((v <= 3).sum()), "params": params}
        s = summary[name]
        print(f"  {name:10s} {s['floor']:9.2f} {s['sd']:7.2f} {s['min']:8.2f} "
              f"{s['max']:8.2f} {s['n_at_3']:8d}/15 {params:8d}")

    print("\n" + "=" * 98)
    print("VERDICT — where does the floor live?")
    print("=" * 98)
    lin = summary["linear"]["floor"]
    mlp = summary["mlp32"]["floor"]
    drop = lin - mlp
    print(f"  linear floor {lin:.2f}pp  |  mlp32 floor {mlp:.2f}pp  |  drop {drop:+.2f}pp")
    print(f"  gate 3.00pp")

    if drop >= 1.5 and mlp < 3.0:
        print(f"\n  => HEAD-EXPRESSIVENESS, and the gate IS reachable with capacity.")
        print(f"     A nonlinear readout cuts the floor {drop:.2f}pp and lands BELOW 3pp.")
        verdict = "head_limited_gate_reachable"
    elif drop >= 1.5:
        print(f"\n  => HEAD-EXPRESSIVENESS. The floor is a property of the 2-logit form,")
        print(f"     not of the frozen features: a nonlinear head cuts it {drop:.2f}pp.")
        print(f"     But it is still above 3pp, so capacity alone does not reach the gate.")
        verdict = "head_limited_still_above_gate"
    else:
        print(f"\n  => FEATURE-LIMITED. Every head form pays about the same floor, so the")
        print(f"     cost is in the frozen representation, not in the readout form. No")
        print(f"     readout change reaches 3pp and the gate is dead at this feature set.")
        verdict = "feature_limited"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "floor-by-head.json").write_text(json.dumps({
        "seeds": list(SEEDS), "order": list(ORDER),
        "solo_epochs": SOLO, "joint_epochs": JOINT,
        "summary": summary, "verdict": verdict, "per_seed": rows,
        "role": "DIAGNOSTIC ONLY - no head is a candidate",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: floor-by-head.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
