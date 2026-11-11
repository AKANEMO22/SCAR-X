"""R7 round 34 — at T=6, is the limit the HEAD again, or genuinely the FEATURES?

APPLYING ROUND 29'S LESSON TO ROUND 32'S FAILURE
-------------------------------------------------
Round 27 called a 6.41pp floor "irreducible" and round 29 overturned it: a nonlinear head's
floor was 0.36pp. The lesson was that a floor must name its configuration.

Round 32/33 now report a T=6 failure at 27.57pp and called it "a genuine scale limit". Before
that label is allowed to stand, it must pass the same test that destroyed the last one:

    is the T=6 floor a property of THIS head (32 tanh units, 1634 params), or of the frozen
    features at six tasks?

MEASUREMENT — the solo/joint decomposition at T=6, by head capacity
    For each head: solo_j (fit on task j alone) and joint (fit on all six, budget matched).
    floor_j = max(0, solo_j - joint_j), loss-based.

    Heads (all on the SAME frozen features, no new neurons in the core):
        mlp32     the round-30/31/32 head
        mlp128    4x wider nonlinear
        mlp32_deep one extra tanh layer

DISCRIMINATION
    If the T=6 floor collapses with more capacity -> round 32's "genuine scale limit" is
       again a HEAD limit, and must be withdrawn the way round 27 was.
    If the floor stays ~27pp across all three -> it is the frozen feature space at six tasks,
       and the T=6 failure is real for any readout over these features.

DIAGNOSTIC ONLY. No head here is proposed as a candidate; the brief forbids capacity
expansion to mask a learning-rule failure, and this measures where the limit is.

EFFICIENCY: features depend only on (seed, task), so they are computed once per seed and
reused by every head. 8 seeds x 6 tasks is enough for a 27pp effect.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scarx.evaluation import r7_secs as S  # noqa: E402
import r7_r32_tier3_six as R32  # noqa: E402

SEEDS = tuple(320001 + 6 * k for k in range(8))
TASKS = R32.TASKS
N_TR, N_TE = 320, 200
SOLO = 1200
JOINT = 6 * SOLO          # budget-matched: each task gets SOLO exposures
HEADS = ("mlp32", "mlp128", "mlp32_deep")


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class MLP:
    """One or two tanh layers, all trained online by delta rule (no BPTT)."""

    def __init__(self, d, seed=0, H=32, deep=False):
        self.mu = None; self.sd = None
        self.H = H; self.deep = deep
        g = np.random.default_rng(seed + 21)
        self.W1 = g.normal(0, 1 / np.sqrt(d), (H, d)); self.b1 = np.zeros(H)
        if deep:
            self.Wm = g.normal(0, 1 / np.sqrt(H), (H, H)); self.bm = np.zeros(H)
        self.W2 = g.normal(0, 1 / np.sqrt(H), (2, H)); self.b2 = np.zeros(2)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 0.05)
        return (F - self.mu) / self.sd

    def fwd(self, A):
        H1 = np.tanh(A @ self.W1.T + self.b1)
        if not self.deep:
            return H1, H1 @ self.W2.T + self.b2
        H2 = np.tanh(H1 @ self.Wm.T + self.bm)
        return (H2, H2 @ self.W2.T + self.b2)

    def step(self, A, y, lr=.1):
        n = len(y)
        out, lg = self.fwd(A)
        lg = lg - lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        e = -p; e[np.arange(n), y] += 1
        d_out = (e @ self.W2) * (1 - out ** 2)
        self.W2 += lr * (e.T @ out) / n; self.b2 += lr * e.mean(0)
        if not self.deep:
            self.W1 += lr * (d_out.T @ A) / n; self.b1 += lr * d_out.mean(0)
        else:
            H1 = np.tanh(A @ self.W1.T + self.b1)
            self.Wm += lr * (d_out.T @ H1) / n; self.bm += lr * d_out.mean(0)
            d_h = (d_out @ self.Wm) * (1 - H1 ** 2)
            self.W1 += lr * (d_h.T @ A) / n; self.b1 += lr * d_h.mean(0)

    def acc(self, A, y):
        _, lg = self.fwd(A)
        return float(np.mean(np.argmax(lg, 1) == y))


def mk(name, d, seed):
    if name == "mlp32":
        return MLP(d, seed, 32, False)
    if name == "mlp128":
        return MLP(d, seed, 128, False)
    return MLP(d, seed, 32, True)


def main():
    t0 = time.time()
    print("=" * 98)
    print("R7 ROUND 34 — at T=6, is the limit the HEAD or the FEATURES?")
    print("=" * 98)
    print(f"  6 tasks, 8 fresh seeds, solo {SOLO} / joint {JOINT} (budget-matched)")
    print(f"  DIAGNOSTIC ONLY — no head is a candidate\n")

    rows = []
    for s_i, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in TASKS:
            trx, try_ = R32.GENS[tk](seed + 200_000, N_TR)
            tex, tey = R32.GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = R32.feats(core, trx); ytr[tk] = try_
            Fte[tk] = R32.feats(core, tex); yte[tk] = tey
        Fj = np.concatenate([Ftr[tk] for tk in TASKS])
        yj = np.concatenate([ytr[tk] for tk in TASKS])

        rec = {"seed": seed, "heads": {}}
        for hn in HEADS:
            solo = {}
            for tk in TASKS:
                h = mk(hn, Ftr[tk].shape[1], seed)
                A = h.z(Ftr[tk])
                for _ in range(SOLO):
                    h.step(A, ytr[tk])
                solo[tk] = h.acc(h.z(Fte[tk]), yte[tk])

            hj = mk(hn, Fj.shape[1], seed)
            Aj = hj.z(Fj)
            rng = np.random.default_rng(seed + 5)
            n = len(yj); half = max(1, n // 2)
            for _ in range(JOINT):
                idx = rng.choice(n, half, replace=False)
                hj.step(Aj[idx], yj[idx])
            joint = {tk: hj.acc(hj.z(Fte[tk]), yte[tk]) for tk in TASKS}

            per = {tk: max(0.0, (solo[tk] - joint[tk]) * 100) for tk in TASKS}
            rec["heads"][hn] = {
                "floor": float(np.mean(list(per.values()))),
                "solo_mean": float(np.mean(list(solo.values()))),
                "joint_mean": float(np.mean(list(joint.values()))),
                "per_task": per}
        rows.append(rec)
        print(f"  {s_i+1}/8 {seed}  " + "  ".join(
            f"{hn} {rec['heads'][hn]['floor']:6.2f}" for hn in HEADS), flush=True)

    print("\n" + "=" * 98)
    print("T=6 FLOOR BY HEAD CAPACITY (loss-based, mean over 8 seeds)")
    print("=" * 98)
    print(f"  {'head':12s} {'floor':>9} {'sd':>7} {'solo':>8} {'joint':>8} {'params':>9}")
    summary = {}
    d = 48
    for hn in HEADS:
        v = np.array([r["heads"][hn]["floor"] for r in rows])
        so = np.mean([r["heads"][hn]["solo_mean"] for r in rows])
        jo = np.mean([r["heads"][hn]["joint_mean"] for r in rows])
        params = {"mlp32": 32 * d + 32 + 2 * 32 + 2,
                  "mlp128": 128 * d + 128 + 2 * 128 + 2,
                  "mlp32_deep": 32 * d + 32 + 32 * 32 + 32 + 2 * 32 + 2}[hn]
        summary[hn] = {"floor": float(v.mean()), "sd": float(v.std(ddof=1)),
                       "solo": float(so), "joint": float(jo), "params": params}
        s = summary[hn]
        print(f"  {hn:12s} {s['floor']:9.2f} {s['sd']:7.2f} {so:8.4f} {jo:8.4f} "
              f"{params:9d}")

    print("\n" + "=" * 98)
    print("COMPARISON WITH T=3 (round 29)")
    print("=" * 98)
    print(f"  T=3  linear 6.52pp   mlp32 0.36pp")
    print(f"  T=6  " + "   ".join(f"{hn} {summary[hn]['floor']:.2f}pp" for hn in HEADS))

    print("\n" + "=" * 98)
    print("VERDICT")
    print("=" * 98)
    base = summary["mlp32"]["floor"]
    best_hn = min(HEADS, key=lambda h: summary[h]["floor"])
    best = summary[best_hn]["floor"]
    drop = base - best
    print(f"  mlp32 floor {base:.2f}pp ; best ({best_hn}) {best:.2f}pp ; drop {drop:+.2f}pp")
    if drop >= 5.0 and best <= 3.0:
        print(f"\n  => HEAD-LIMITED AGAIN. The T=6 floor collapses from {base:.1f} to "
              f"{best:.1f}pp")
        print(f"     with more capacity and lands under the gate. Round 32/33's 'genuine")
        print(f"     scale limit' MUST BE WITHDRAWN, exactly as round 27's was.")
        verdict = "head_limited_again"
    elif drop >= 5.0:
        print(f"\n  => PARTLY HEAD-LIMITED. The floor falls {drop:.1f}pp with capacity but")
        print(f"     stays at {best:.1f}pp, above the 3pp gate. Both factors matter.")
        verdict = "partly_head_limited"
    else:
        print(f"\n  => FEATURE-LIMITED AT T=6. Every head pays about the same floor, so the")
        print(f"     cost is in the frozen six-task feature space, not the readout. Round 32's")
        print(f"     scale-limit conclusion STANDS and this time it has been tested against")
        print(f"     the confound that broke round 27.")
        verdict = "feature_limited_t6"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t6-floor-by-head.json").write_text(json.dumps({
        "tasks": list(TASKS), "seeds": list(SEEDS), "heads": list(HEADS),
        "solo_epochs": SOLO, "joint_epochs": JOINT,
        "summary": summary, "verdict": verdict,
        "t3_reference": {"linear": 6.52, "mlp32": 0.36},
        "role": "DIAGNOSTIC ONLY - no head is a candidate",
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: t6-floor-by-head.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
