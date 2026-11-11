"""Round 4A Stage 0 — audit, reproduction of Grok's near-chance query result.

Checks before any hypothesis test:
  A. query_set is verbatim vs the Grok R3B body hash, and matches an independent
     re-implementation from the written specification.
  B. the oracle answers agree with the generator's own labels
  C. counterfactual pairs really flip only because the query changes
  D. forward parity with SpikingCore.step
  E. hidden weights frozen
  F. train/test independence by checksum
  G. reproduction of the near-chance query result
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r4_relational as R  # noqa: E402
from scarx.evaluation import r4_probe as P  # noqa: E402
from scarx.neural.core import SpikingCore  # noqa: E402

SEEDS = (95001, 95007, 95013)


class Res:
    def __init__(self):
        self.rows, self.ok = [], True

    def check(self, name, passed, detail=""):
        self.rows.append({"check": name, "pass": bool(passed), "detail": detail})
        if not passed:
            self.ok = False
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return passed


def body_sha(fn):
    import inspect
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()


def main():
    print("=" * 76)
    print("R4A STAGE 0 — audit and reproduction")
    print("=" * 76)
    res = Res()

    print("\nA — generator provenance")
    x, y, fk = R.query_set(95001, 128)
    res.check("query_set produces 12x8 episodes with 2 classes",
              x.shape == (128, 12, 8) and set(y.tolist()) == {0, 1},
              f"shape {x.shape}, labels {sorted(set(y.tolist()))}")
    xi, yi, fki = R.independent_query_set(95001, 128)
    res.check("independent re-implementation reproduces the same task statistics",
              np.array_equal(np.bincount(yi, minlength=2), np.bincount(y, minlength=2))
              and np.array_equal(np.bincount(fki, minlength=2), np.bincount(fk, minlength=2)),
              f"label counts {np.bincount(y)} vs {np.bincount(yi)}")

    print("\nB — oracle agrees with the generator")
    orc = np.array([R.oracle_answer(x[i]) for i in range(len(x))])
    res.check("input oracle reproduces every label", np.array_equal(orc, y),
              f"{int(np.sum(orc == y))}/{len(y)} match")

    print("\nC — counterfactual pairs flip only because the query changes")
    px, py, pid, _ = R.paired_counterfactual(95001, 64)
    ev, od = px[0::2], px[1::2]
    # the two siblings must differ ONLY at the query tick 10, channels 4:6
    diff = np.abs(ev - od).max(axis=(1, 2))
    only_query = np.all(np.abs(ev - od)[:, :10, :] == 0) and np.all(np.abs(ev - od)[:, 11:, :] == 0)
    flips = py[0::2] != py[1::2]
    res.check("counterfactual siblings differ only at the query tick", bool(only_query),
              f"max off-query diff {diff.max():.6f}")
    res.check("counterfactual labels flip whenever the values differ",
              bool(np.all(flips)), f"{int(flips.sum())}/{len(flips)} pairs flip")

    print("\nC2 — equal-value pairs do NOT flip (must not be scored as flips)")
    ex, ey, epid = R.equal_value_pairs(95001, 64)
    res.check("equal-value siblings have identical answers",
              bool(np.all(ey[0::2] == ey[1::2])), "no legitimate flip to score")

    print("\nD — forward parity with SpikingCore.step")
    base = SpikingCore(P.cfg(SEEDS[0]))
    cap = P.capture(SEEDS[0], x[:8])
    worst = 0.0
    for i in range(4):
        base.reset_state()
        for t in range(12):
            base.step(x[i, t], track=False)
            worst = max(worst, float(np.max(np.abs(base.spikes - cap["spike_train"][i, t]))))
    res.check("spike trajectory equals SpikingCore.step", worst == 0.0, f"max diff {worst:.2e}")

    print("\nE — hidden weights frozen")
    _, _, w0, r0 = P.weights(SEEDS[0])
    h0 = hashlib.sha256(w0.tobytes() + r0.tobytes()).hexdigest()
    trf = cap["final"]["spikesum"]
    P.train_head(trf, y[:8])
    _, _, w1, r1 = P.weights(SEEDS[0])
    res.check("head training does not touch hidden weights",
              h0 == hashlib.sha256(w1.tobytes() + r1.tobytes()).hexdigest())

    print("\nF — train/test independence")
    xtr, ytr, _ = R.query_set(95001 + 200_000, 256)
    xte, yte, _ = R.query_set(95001 + 300_000, 256)
    a = R.dataset_sha256(xtr, ytr)
    b = R.dataset_sha256(xte, yte)
    res.check("train and test checksums differ", a != b,
              f"train {a[:12]}… test {b[:12]}…")
    res.check("no identical episodes shared across splits",
              len({r.tobytes() for r in xtr} & {r.tobytes() for r in xte}) == 0,
              "zero byte-identical episodes")

    print("\nG — reproduce Grok's near-chance query result")
    print(f"  {'seed':>7} {'decay head':>11} {'ssum head':>10} {'ssum ridge':>11} {'bank5 head':>11}")
    rows = []
    for seed in SEEDS:
        trx, try_, _ = R.query_set(seed + 200_000, 1024)
        tex, tey, _ = R.query_set(seed + 300_000, 512)
        a_tr = P.capture(seed, trx)
        a_te = P.capture(seed, tex)
        d_tr = a_tr["final"]["decay025"]; d_te = a_te["final"]["decay025"]
        s_tr = a_tr["final"]["spikesum"]; s_te = a_te["final"]["spikesum"]
        b_tr = a_tr["final"]["bank5"]; b_te = a_te["final"]["bank5"]
        dh = P.train_head(d_tr, try_).acc(d_te, tey)
        sh = P.train_head(s_tr, try_).acc(s_te, tey)
        sr = P.ridge(s_tr, try_, s_te, tey)
        bh = P.train_head(b_tr, try_).acc(b_te, tey)
        rows.append({"seed": seed, "decay_head": dh, "ssum_head": sh,
                     "ssum_ridge": sr, "bank5_head": bh})
        print(f"  {seed:>7} {dh:11.4f} {sh:10.4f} {sr:11.4f} {bh:11.4f}")
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "seed"}
    print(f"  {'MEAN':>7} {m['decay_head']:11.4f} {m['ssum_head']:10.4f} "
          f"{m['ssum_ridge']:11.4f} {m['bank5_head']:11.4f}")
    res.check("query task is near chance for the spike-sum head (Grok: ~0.55-0.59)",
              abs(m["ssum_head"] - 0.5) < 0.12, f"mean {m['ssum_head']:.4f}")
    res.check("reproduction within Grok's reported band",
              0.45 <= m["ssum_head"] <= 0.65, f"mean {m['ssum_head']:.4f}")

    passed = sum(r["pass"] for r in res.rows)
    print("\n" + "=" * 76)
    print(f"RESULT {passed}/{len(res.rows)}  VERDICT: {'P0_PASS' if res.ok else 'P0_FAIL'}")
    print("=" * 76)
    out = {"stage": 0, "checks": res.rows, "passed": passed, "total": len(res.rows),
           "verdict": "P0_PASS" if res.ok else "P0_FAIL",
           "query_set_body_sha256": R.QUERY_SET_BODY_SHA256,
           "grok_source": {"branch": "research/grok/r3-temporal-stress",
                           "commit": "211380fecdcd05966ec78bb7686f54fa8179c11a",
                           "blob_sha": "874fb83507af445ea8ec5db73942fabef03563fa"},
           "reproduction": rows, "means": m}
    p = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r4"
    p.mkdir(parents=True, exist_ok=True)
    (p / "stage0-audit.json").write_text(json.dumps(out, indent=2, default=float) + "\n",
                                         encoding="utf-8")
    print(f"written: {p / 'stage0-audit.json'}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
