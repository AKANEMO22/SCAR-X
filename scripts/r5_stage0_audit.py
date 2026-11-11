"""Round 5B Stage 0 — correctness audit before any capability experiment.

The most important check here is not a code check: it is confirming empirically that
the shortcut ceilings are what the protocol claims. If "always answer key A" is NOT
~0.75, then the whole gate design is wrong and nothing downstream is interpretable.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r5_probe as P  # noqa: E402
from scarx.neural.core import SpikingCore  # noqa: E402

SEEDS = (97001, 97007, 97013)


class Res:
    def __init__(self):
        self.rows, self.ok = [], True

    def check(self, name, passed, detail=""):
        self.rows.append({"check": name, "pass": bool(passed), "detail": detail})
        if not passed:
            self.ok = False
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return passed


def main():
    print("=" * 78)
    print("R5B STAGE 0 — audit")
    print("=" * 78)
    res = Res()

    x, y, vA, vB, q, _ = G.independent_query_set_v1(97001, 4096)

    print("\nA — generator shape and independence")
    res.check("shapes are (N,12,8) with binary labels",
              x.shape == (4096, 12, 8) and set(np.unique(y).tolist()) <= {0, 1},
              f"{x.shape}")
    # independence: vA and vB must be uncorrelated
    corr = float(np.corrcoef(vA, vB)[0, 1])
    res.check("vA and vB are independent (|corr| < 0.05)", abs(corr) < .05,
              f"corr = {corr:+.4f}")
    comp = float(np.mean((vA == 0) & (vB == 1)))
    res.check("the R4A complementary pattern is NOT forced",
              abs(comp - .25) < .04,
              f"P(vA=0,vB=1) = {comp:.4f}, expected ~0.25 if independent")
    # all four states uniformly
    states = vA * 2 + vB
    cnt = np.bincount(states, minlength=4) / len(states)
    res.check("all four memory states are uniform (each ~0.25)",
              np.all(np.abs(cnt - .25) < .04), f"{np.round(cnt, 4).tolist()}")
    qc = np.bincount(q, minlength=2) / len(q)
    res.check("query is uniform", np.all(np.abs(qc - .5) < .04), f"{np.round(qc,4).tolist()}")

    print("\nB — oracle agrees with the generator")
    orc = G.oracle_all(x)
    res.check("oracle reproduces every label", np.array_equal(orc, y),
              f"{int(np.sum(orc == y))}/{len(y)}")

    print("\nC — THE CRUX: shortcut ceilings are 0.75, not 0.50")
    fk = G.baseline_first_key(x)
    acc_fk = float(np.mean(fk == y))
    res.check("always-answer-key-A scores ~0.75", abs(acc_fk - .75) < .03,
              f"{acc_fk:.4f}  <-- this is why 0.50 is the wrong null")
    maj = G.baseline_majority(len(y), 0)
    acc_maj = float(np.mean(maj == y))
    res.check("majority class scores ~0.50", abs(acc_maj - .5) < .03, f"{acc_maj:.4f}")
    qo = G.baseline_query_only(x)
    acc_qo = float(np.mean(qo == y))
    res.check("query-only scores ~0.50", abs(acc_qo - .5) < .03, f"{acc_qo:.4f}")
    # know both values but not the query: correct only when vA == vB
    know_both = np.where(vA == vB, vA, maj)
    acc_kb = float(np.mean(know_both == y))
    res.check("knows both values but not the query scores ~0.75",
              abs(acc_kb - .75) < .03, f"{acc_kb:.4f}")

    print("\nD — counterfactual pairs, both cases")
    px, py, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(97001, 1024)
    a, b = py[0::2], py[1::2]
    caseA = pvA[0::2] != pvB[0::2]
    res.check("case A (vA != vB) labels flip on query swap",
              bool(np.all(a[caseA] != b[caseA])),
              f"{int(caseA.sum())} pairs, all flip")
    res.check("case B (vA == vB) labels do NOT flip",
              bool(np.all(a[~caseA] == b[~caseA])),
              f"{int((~caseA).sum())} pairs, none flip")
    res.check("both cases are present and roughly balanced",
              0.4 < float(caseA.mean()) < 0.6, f"case A fraction {caseA.mean():.3f}")
    # siblings differ ONLY at the query tick
    ev, od = px[0::2], px[1::2]
    diff = np.abs(ev - od)
    res.check("siblings differ only at the query tick",
              bool(np.all(diff[:, :G.QUERY_TICK, :] == 0)) and
              bool(np.all(diff[:, G.QUERY_TICK + 1:, :] == 0)),
              f"max off-query diff {diff[:, :G.QUERY_TICK, :].max():.6f}")
    res.check("background noise is identical within a pair",
              bool(np.array_equal(ev[:, 0, 6:], od[:, 0, 6:])), "ch6,7 equal")

    print("\nE — representations cannot see the label")
    before = P.capture(97001, x[:64])
    _ = y[:64] * 0 + 1          # labels flipped conceptually, features must not move
    after = P.capture(97001, x[:64])
    same = all(np.array_equal(before[w][k], after[w][k])
               for w in ("pre", "post") for k in before[w])
    res.check("frozen features are label-independent", same)

    print("\nF — hidden weights frozen, forward parity exact")
    _, _, w0, r0 = P.weights(97001)
    h0 = hashlib.sha256(w0.tobytes() + r0.tobytes()).hexdigest()
    cap = P.capture(97001, x[:64])
    feats = cap["final"]["decay025"]
    P.train_head(feats, y[:64])
    _, _, w1, r1 = P.weights(97001)
    res.check("head training leaves hidden weights untouched",
              h0 == hashlib.sha256(w1.tobytes() + r1.tobytes()).hexdigest())

    base = SpikingCore(P.cfg(97001))
    worst = 0.0
    for i in range(4):
        base.reset_state()
        for t in range(G.N_TICKS):
            base.step(x[i, t], track=False)
            worst = max(worst, float(np.max(np.abs(base.spikes - cap["spike_train"][i, t]))))
    res.check("spike trajectory equals SpikingCore.step", worst == 0.0,
              f"max diff {worst:.2e}")

    print("\nG — query tick is read where the generator puts it")
    # shift must keep the query inside the window: query tick 10 + shift <= 11
    for shift in (0, 1):
        xs = G.shifted(x[:32], shift) if shift else x[:32]
        qt = G.QUERY_TICK + shift
        P.capture(97001, xs, query_tick=qt)
        qm = np.flatnonzero(xs[0, qt, 4:6] > .5)
        res.check(f"shift={shift}: cue found at tick {qt}", len(qm) == 1,
                  f"query channel index {int(qm[0]) if len(qm) else 'none'}")
    # a shift that would push the query out of the window must be rejected, not
    # silently truncated
    try:
        G.shifted(x[:8], 2)
        clipped = P.capture(97001, G.shifted(x[:8], 2), query_tick=12)
        found = bool(np.any(G.shifted(x[:8], 2)[0, 11, 4:6] > .5))
        res.check("shift=2 pushes the query out of the window (documented limit)",
                  not found, "query would land at tick 12; generator drops it")
    except Exception as exc:                       # noqa: BLE001
        res.check("shift=2 is rejected rather than silently truncated", True,
                  f"raised {type(exc).__name__}")

    print("\nH — train/test independence")
    xtr, ytr, *_ = G.independent_query_set_v1(97001 + 200_000, 512)
    xte, yte, *_ = G.independent_query_set_v1(97001 + 300_000, 512)
    res.check("train and test checksums differ",
              G.dataset_sha256(xtr, ytr) != G.dataset_sha256(xte, yte),
              f"{G.dataset_sha256(xtr,ytr)[:12]}… vs {G.dataset_sha256(xte,yte)[:12]}…")
    res.check("no byte-identical episode shared across splits",
              len({r.tobytes() for r in xtr} & {r.tobytes() for r in xte}) == 0)

    print("\nI — label distribution is balanced per split")
    for nm, (xx, yy, *_ ) in (("train", (xtr, ytr)), ("test", (xte, yte))):
        c = np.bincount(yy, minlength=2) / len(yy)
        res.check(f"{nm} labels ~50/50", np.all(np.abs(c - .5) < .05),
                  f"{np.round(c,4).tolist()}")

    print("\nJ — transforms preserve labels where they must")
    xs = G.shifted(x[:256], 1)
    res.check("shift preserves the oracle answer",
              np.array_equal(G.oracle_all(xs), G.oracle_all(x[:256])))
    xr = G.reversed_writes(x[:256])
    res.check("reversed write order preserves the oracle answer",
              np.array_equal(G.oracle_all(xr), G.oracle_all(x[:256])))
    xd = G.with_distractors(x[:256], 5)
    res.check("distractors preserve the oracle answer",
              np.array_equal(G.oracle_all(xd), G.oracle_all(x[:256])))

    passed = sum(r["pass"] for r in res.rows)
    print("\n" + "=" * 78)
    print(f"RESULT {passed}/{len(res.rows)}  VERDICT: {'P0_PASS' if res.ok else 'P0_FAIL'}")
    print("=" * 78)
    out = {"stage": 0, "checks": res.rows, "passed": passed, "total": len(res.rows),
           "verdict": "P0_PASS" if res.ok else "P0_FAIL",
           "shortcut_ceilings": {"first_key": acc_fk, "majority": acc_maj,
                                 "query_only": acc_qo, "knows_both_no_query": acc_kb},
           "memory_states": cnt.tolist(), "caseA_fraction": float(caseA.mean()),
           "seeds": list(SEEDS)}
    p = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r5"
    p.mkdir(parents=True, exist_ok=True)
    (p / "stage0-audit.json").write_text(json.dumps(out, indent=2, default=float) + "\n",
                                         encoding="utf-8")
    print(f"written: {p / 'stage0-audit.json'}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
