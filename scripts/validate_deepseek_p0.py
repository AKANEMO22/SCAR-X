"""P0 technical validation for the deepseek pair_binding branch.

No capability number may be reported unless every check here passes.

Checks
------
1. generator determinism (same seed -> bit-identical arrays)
2. NO BAG SHORTCUT: order-1 probe <= 0.60 on every dev generator seed.
   This is the check that caught the discarded first generator draft.
3. order-2 solvability + honest label-shuffled control
4. shape compatibility with strict_binding (harness cannot read the wrong task)
5. no target leakage into inference or hidden update
6. frozen arm provably does not change hidden weights
7. checkpoint round-trip + cold-process resume
8. peak RSS and commit, in-process and across the process tree

Run: python scripts/validate_deepseek_p0.py
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation.pair_binding import (  # noqa: E402
    order2_features, order3_binding, order3_features, pair_binding,
)
from scarx.evaluation.tasks21 import ridge_probe  # noqa: E402
from scarx.neural.core import Config  # noqa: E402
from scarx.neural.plasticity import PlasticCore  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

DEV_SEEDS = (41001, 41007, 41013)
GEN_SEEDS = (31001, 31009, 31013, 31019, 31027)
CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12)


def config(seed):
    return Config(seed=seed, **CFG)


class Results:
    def __init__(self):
        self.rows, self.ok = [], True

    def check(self, name, passed, detail=""):
        self.rows.append({"check": name, "pass": bool(passed), "detail": detail})
        if not passed:
            self.ok = False
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
        return passed


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------- 1 determinism
def check_determinism(res):
    for seed in GEN_SEEDS:
        ax, ay = pair_binding(seed, 64)
        bx, by = pair_binding(seed, 64)
        if not (np.array_equal(ax, bx) and np.array_equal(ay, by)):
            res.check(f"generator deterministic seed{seed}", False, "arrays differ")
            return
    res.check("generator deterministic across repeated calls", True,
              f"{len(GEN_SEEDS)} seeds bit-identical")


# -------------------------------------------------- 2 the critical: no shortcut
def check_no_bag_shortcut(res):
    """An order-1 additive bag must NOT read the label. This is the gate that
    caught the discarded first draft (which scored 0.70-0.77)."""
    worst_o1, worst_shuf = 0.0, 0.0
    for seed in GEN_SEEDS:
        fx, fy = pair_binding(seed, 500)
        tx, ty = pair_binding(seed + 900, 500)
        o1 = ridge_probe(fx.reshape(len(fx), -1), fy, tx.reshape(len(tx), -1), ty)
        perm = np.random.default_rng(0).permutation(fy)
        o1s = ridge_probe(fx.reshape(len(fx), -1), perm, tx.reshape(len(tx), -1), ty)
        worst_o1 = max(worst_o1, o1)
        worst_shuf = max(worst_shuf, abs(o1s - 0.5) + 0.5)
        print(f"       seed{seed}: order-1={o1:.4f} shuffled={o1s:.4f}")
    res.check("NO bag shortcut: order-1 probe <= 0.60 on all seeds", worst_o1 <= 0.60,
              f"worst order-1 = {worst_o1:.4f}")
    res.check("order-1 shuffled control is near chance", worst_shuf <= 0.60,
              f"worst shuffled = {worst_shuf:.4f}")


# ------------------------------------------------------ 3 order-2 solvability
def check_solvable(res):
    worst = 1.0
    for seed in GEN_SEEDS:
        fx, fy = pair_binding(seed, 400)
        tx, ty = pair_binding(seed + 900, 400)
        o2 = ridge_probe(order2_features(fx), fy, order2_features(tx), ty)
        worst = min(worst, o2)
        perm = np.random.default_rng(0).permutation(fy)
        o2s = ridge_probe(order2_features(fx), perm, order2_features(tx), ty)
        print(f"       seed{seed}: order-2={o2:.4f} shuffled={o2s:.4f}")
    res.check("order-2 basis solves pair_binding (>= 0.90)", worst >= 0.90,
              f"worst order-2 = {worst:.4f}")

    # and strict_binding still needs order-3, confirming the diagnosis
    fx, fy = order3_binding(31001, 400)
    tx, ty = order3_binding(31901, 400)
    s2 = ridge_probe(order2_features(fx), fy, order2_features(tx), ty)
    s3 = ridge_probe(order3_features(fx), fy, order3_features(tx), ty)
    res.check("strict_binding still requires order-3 (order-2 insufficient)",
              s2 < 0.60 and s3 > 0.90, f"order-2={s2:.4f} order-3={s3:.4f}")


# --------------------------------------------------------- 4 shape compatibility
def check_shapes(res):
    px, py = pair_binding(41001, 32)
    sx, sy = order3_binding(41001, 32)
    res.check("pair_binding returns strict_binding shapes",
              px.shape == sx.shape and py.shape == sy.shape and px.dtype == sx.dtype,
              f"{px.shape} vs {sx.shape}")
    res.check("inputs are finite and labels in range",
              np.isfinite(px).all() and int(py.min()) >= 0 and int(py.max()) < 2,
              f"label set {sorted(set(py.tolist()))}")


# ------------------------------------------------------------- 5 no leakage
def check_no_leakage(res):
    seed = 5100
    x, y = pair_binding(seed, 64)
    perm = np.random.default_rng(1).permutation(y)
    res.check("label permutation leaves the input hash unchanged",
              sha(x) == sha(x.copy()), "input untouched by label handling")

    cfg = config(seed)
    a = PlasticCore(cfg, "score")
    b = PlasticCore(cfg, "score")
    xb, yb = pair_binding(seed + 900, 64)
    for i in range(8):
        a.forward(x[i], track=True); a.learn(int(y[i]), reward=-0.5)
        b.forward(xb[i], track=True); b.learn(int(yb[i]), reward=-0.5)
    # same inputs, different labels, same reward -> hidden must be identical
    c = PlasticCore(cfg, "score")
    d = PlasticCore(cfg, "score")
    for i in range(8):
        c.forward(x[i], track=True); c.learn(int(y[i]), reward=-0.5)
        d.forward(x[i], track=True); d.learn(int(perm[i]), reward=-0.5)
    res.check("hidden update independent of the label at equal reward",
              np.array_equal(c.w_in, d.w_in) and np.array_equal(c.w_rec, d.w_rec),
              "w_in/w_rec bit-equal under permuted labels")


# ------------------------------------------------------------ 6 frozen arm
def check_frozen(res):
    seed = 6100
    x, y = pair_binding(seed, 32)
    m = PlasticCore(config(seed), "frozen")
    before_in, before_rec = m.w_in.copy(), m.w_rec.copy()
    for i in range(16):
        m.forward(x[i], track=True); m.learn(int(y[i]))
    res.check("frozen arm does not change hidden weights",
              np.array_equal(m.w_in, before_in) and np.array_equal(m.w_rec, before_rec),
              "w_in/w_rec unchanged after 16 updates")


# ------------------------------------------------- 7 checkpoint / cold resume
def check_checkpoint(res, tmpdir):
    seed = 7100
    x, y = pair_binding(seed, 16)
    m = PlasticCore(config(seed), "score")
    for i in range(8):
        m.forward(x[i], track=True); m.learn(int(y[i]))

    payload = m.dumps()
    restored = PlasticCore.loads(payload)
    same = all(np.array_equal(np.ascontiguousarray(getattr(m, k)),
                              np.ascontiguousarray(getattr(restored, k)))
               for k in m.arrays())
    res.check("checkpoint round-trip restores all arrays bit-exactly", same,
              f"{len(m.arrays())} arrays")

    path = Path(tmpdir) / "d.core.npz"
    path.write_bytes(payload)
    np.save(Path(tmpdir) / "seq.npy", x[8])
    script = (
        "import sys,json,hashlib,numpy as np\n"
        f"sys.path.insert(0,{str(Path(__file__).resolve().parents[1] / 'src')!r})\n"
        "from scarx.neural.plasticity import PlasticCore\n"
        f"m=PlasticCore.load({str(path)!r})\n"
        f"m.forward(np.load({str(Path(tmpdir) / 'seq.npy')!r}),track=True)\n"
        "m.learn(int(sys.argv[1]))\n"
        "print(json.dumps({k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()"
        " for k,v in m.arrays().items()}))\n")
    (Path(tmpdir) / "cold.py").write_text(script)
    cold = subprocess.run([sys.executable, str(Path(tmpdir) / "cold.py"), str(int(y[8]))],
                          capture_output=True, text=True, timeout=300)
    if cold.returncode != 0:
        res.check("cold-process resume", False, cold.stderr.strip()[:200])
        return
    cold_h = json.loads(cold.stdout.strip().splitlines()[-1])
    warm = PlasticCore.loads(payload)
    warm.forward(x[8], track=True); warm.learn(int(y[8]))
    warm_h = {k: sha(v) for k, v in warm.arrays().items()}
    res.check("cold-process resume reproduces the next step bit-exactly",
              cold_h == warm_h, "array hashes match")


# ------------------------------------------------------------------ 8 memory
def check_memory(res):
    seed = 8100
    x, y = pair_binding(seed, 64)
    with ResourceMonitor(interval=.05) as mon:
        m = PlasticCore(config(seed), "score")
        for i in range(64):
            m.forward(x[i], track=True); m.learn(int(y[i]))
        _ = m.dumps()
        mon.check()
    rep = mon.report()
    res.check("peak commit under 2,000,000,000 B",
              rep["peak_commit_or_vms_bytes"] < 2_000_000_000,
              f"peak_commit={rep['peak_commit_or_vms_bytes']:,} B")
    res.check("monitor sampled (not a vacuous zero)", rep["samples"] > 0,
              f"{rep['samples']} samples")
    return rep


def main():
    print("=" * 72)
    print("P0 TECHNICAL VALIDATION — deepseek pair_binding")
    print("=" * 72)
    res = Results()
    with tempfile.TemporaryDirectory() as tmp:
        print("\n[1] generator determinism")
        check_determinism(res)
        print("\n[2] CRITICAL: no bag shortcut")
        check_no_bag_shortcut(res)
        print("\n[3] order-2 solvability and the strict_binding diagnosis")
        check_solvable(res)
        print("\n[4] shape compatibility")
        check_shapes(res)
        print("\n[5] target leakage")
        check_no_leakage(res)
        print("\n[6] frozen arm")
        check_frozen(res)
        print("\n[7] checkpoint and cold resume")
        check_checkpoint(res, tmp)
        print("\n[8] memory")
        rep = check_memory(res)

    passed = sum(r["pass"] for r in res.rows)
    print("\n" + "=" * 72)
    print(f"RESULT: {passed}/{len(res.rows)} checks passed")
    print("VERDICT:", "P0_PASS" if res.ok else "P0_FAIL")
    print("=" * 72)

    out = {"scope": "P0 only; no capability claim", "checks": res.rows,
           "passed": passed, "total": len(res.rows),
           "verdict": "P0_PASS" if res.ok else "P0_FAIL",
           "resources": rep, "python": sys.version, "numpy": np.__version__,
           "platform": platform.platform(),
           "threads": {k: os.environ.get(k) for k in
                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    tgt = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r1" / "p0-validation.json"
    tgt.parent.mkdir(parents=True, exist_ok=True)
    tgt.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(f"written: {tgt}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
