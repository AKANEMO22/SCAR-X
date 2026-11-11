"""R7 round 35 — WHY is T=6 feature-limited? The effective-rank connection.

ROUND 34 RESULT
    T=6 floor: mlp32 31.33pp, mlp128 30.47pp, mlp32_deep 31.64pp
    A 4x wider head and a deeper head both fail identically.
    solo (one task) 0.97   ->   joint (all six) 0.66
So the frozen features carry each task well and cannot carry six at once, regardless of readout.

THE HYPOTHESIS THIS CONNECTS TO
    Round 12 measured the frozen population at **effective rank 4.60 of 48** (participation
    ratio). Round 12's *correlation* with accuracy was later refuted (+0.169 across a threshold
    sweep), but the rank measurement itself replicated and was never explained.

    Six 2-class decision boundaries need more than ~5 effective dimensions to coexist. The
    linear case needed 6.52pp at T=3 with rank ~4.6 and only 3 tasks.

TEST
    For each seed, measure at T=3 and at T=6:
        effective rank of the union features
        joint ceiling of an mlp32 head on the union
    Then correlate rank with the joint ceiling across seeds, and compare T=3 to T=6.

    Predicts, if the hypothesis holds: rank is LOW and joint is HIGH at T=3, and BOTH degrade
    together at T=6. If rank does not change from T=3 to T=6 while joint collapses, the
    hypothesis is wrong and the cause is elsewhere.

Cheap: features depend on (seed, task); reuse. 8 seeds.
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
import r7_r34_t6_head as R34  # noqa: E402

SEEDS = R34.SEEDS
ALL6 = R32.TASKS
FIRST3 = ALL6[:3]
N_TR, N_TE = 320, 200
JOINT = 6 * 1200


def eff_rank(F):
    C = np.cov(F - F.mean(0), rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def joint_ceiling(seed, Fte, yte, Ftr, ytr):
    Fj = np.concatenate([Ftr[tk] for tk in Ftr])
    yj = np.concatenate([ytr[tk] for tk in Ftr])
    h = R34.MLP(Fj.shape[1], seed, 32, False)
    Aj = h.z(Fj)
    rng = np.random.default_rng(seed + 5)
    n = len(yj); half = max(1, n // 2)
    for _ in range(JOINT):
        idx = rng.choice(n, half, replace=False)
        h.step(Aj[idx], yj[idx])
    return float(np.mean([h.acc(h.z(Fte[tk]), yte[tk]) for tk in Fte]))


def main():
    t0 = time.time()
    print("=" * 94)
    print("R7 ROUND 35 — is the T=6 ceiling explained by the effective rank of the features?")
    print("=" * 94)

    rows = []
    for s_i, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in ALL6:
            trx, try_ = R32.GENS[tk](seed + 200_000, N_TR)
            tex, tey = R32.GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = R32.feats(core, trx); ytr[tk] = try_
            Fte[tk] = R32.feats(core, tex); yte[tk] = tey

        # T=3 union
        u3 = np.concatenate([Ftr[tk] for tk in FIRST3])
        r3 = eff_rank(u3)
        j3 = joint_ceiling(seed, {tk: Fte[tk] for tk in FIRST3},
                           {tk: yte[tk] for tk in FIRST3},
                           {tk: Ftr[tk] for tk in FIRST3},
                           {tk: ytr[tk] for tk in FIRST3})
        # T=6 union
        u6 = np.concatenate([Ftr[tk] for tk in ALL6])
        r6 = eff_rank(u6)
        j6 = joint_ceiling(seed, Fte, yte, Ftr, ytr)

        rows.append({"seed": seed, "rank3": r3, "joint3": j3,
                     "rank6": r6, "joint6": j6})
        print(f"  {s_i+1}/8 {seed}  T=3 rank {r3:5.2f} joint {j3:.4f} | "
              f"T=6 rank {r6:5.2f} joint {j6:.4f}", flush=True)

    r3 = np.array([x["rank3"] for x in rows])
    r6 = np.array([x["rank6"] for x in rows])
    j3 = np.array([x["joint3"] for x in rows])
    j6 = np.array([x["joint6"] for x in rows])

    def corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print("\n" + "=" * 94)
    print("AGGREGATES")
    print("=" * 94)
    print(f"  T=3 union: effective rank {r3.mean():5.2f}/48   joint ceiling "
          f"{j3.mean():.4f}   floor {(1-j3.mean())*100:5.2f}pp")
    print(f"  T=6 union: effective rank {r6.mean():5.2f}/48   joint ceiling "
          f"{j6.mean():.4f}   floor {(1-j6.mean())*100:5.2f}pp")
    print(f"\n  rank change T3->T6 : {r6.mean()-r3.mean():+6.2f}")
    print(f"  joint change T3->T6: {j6.mean()-j3.mean():+.4f}")
    print(f"\n  corr(rank, joint) at T=3 = {corr(r3, j3):+.4f}")
    print(f"  corr(rank, joint) at T=6 = {corr(r6, j6):+.4f}")
    print(f"  corr over both pooled    = {corr(np.r_[r3, r6], np.r_[j3, j6]):+.4f}")

    # how many tasks per effective dimension
    print(f"\n  tasks per effective dimension: T=3 {3/r3.mean():.2f}, "
          f"T=6 {6/r6.mean():.2f}")

    print("\n" + "=" * 94)
    print("VERDICT")
    print("=" * 94)
    drop_rank = r6.mean() - r3.mean()
    drop_joint = j6.mean() - j3.mean()
    if drop_joint < -0.10 and drop_rank < -1.0:
        print(f"  => SUPPORTED. The rank falls {abs(drop_rank):.1f} dims and the joint")
        print(f"     ceiling falls {abs(drop_joint):.2f}. Six tasks need "
              f"{6/r6.mean():.1f} per effective")
        print(f"     dimension where three needed {3/r3.mean():.1f}. The feature space,")
        print(f"     not the readout, is the T=6 wall.")
        verdict = "rank_explains_t6"
    elif abs(drop_rank) < 1.0:
        print(f"  => NOT SUPPORTED. Rank barely moves ({drop_rank:+.2f}) while the joint")
        print(f"     ceiling falls {drop_joint:+.2f}. Something else explains T=6.")
        verdict = "rank_does_not_explain"
    else:
        print(f"  => PARTIAL. rank {drop_rank:+.2f}, joint {drop_joint:+.2f}.")
        verdict = "partial"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t6-rank-connection.json").write_text(json.dumps({
        "seeds": list(SEEDS), "first3": list(FIRST3), "all6": list(ALL6),
        "rank3": float(r3.mean()), "rank6": float(r6.mean()),
        "joint3": float(j3.mean()), "joint6": float(j6.mean()),
        "corr_rank_joint_t3": corr(r3, j3),
        "corr_rank_joint_t6": corr(r6, j6),
        "corr_pooled": corr(np.r_[r3, r6], np.r_[j3, j6]),
        "verdict": verdict, "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: t6-rank-connection.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
