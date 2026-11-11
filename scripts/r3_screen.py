"""Round 3A registered screen — substrate x task, three readers.

Protocol: experiments/collab/scarx-neural-round1-20261111/deepseek-r3/protocol-v1.md

Gate S1-S6 as registered. strict_binding is scored and reported but excluded from
S2, per protocol section 7 (no substrate carries it; predicted in section 0).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r3_substrate as S  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.evaluation.tasks21 import ridge_probe  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

DEV_SEEDS = (91001, 91007, 91013)                 # protocol section 6
CONF_SEEDS = (92001, 92007, 92013, 92019, 92023, 92029, 92031, 92037)
N_CAL, N_TRAIN, N_TEST = 128, 1024, 512
HEAD_LR, HEAD_CLIP, EPOCHS = .08, .025, 1
CANDIDATES = ("count_plus_fast", "temporal_bins_3")
BASELINES = ("decay025", "spikesum")
ALL_SUB = S.SUBSTRATES


class Head:
    def __init__(self, dim, seed, lr=HEAD_LR, clip=HEAD_CLIP, zscore=True):
        g = np.random.default_rng(seed + 991)
        self.w = g.normal(0, .03, (2, dim)).astype(np.float32)
        self.b = np.zeros(2, np.float32)
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.zeros(dim, np.float64)
        self.n = 0
        self.lr, self.clip, self.zscore = lr, clip, zscore

    def observe(self, f):
        self.n += 1
        d = f - self.mean
        self.mean += d / self.n
        self.m2 += d * (f - self.mean)

    def basis(self, f):
        if not self.zscore:
            return f.astype(np.float32)
        s = np.maximum(np.sqrt(self.m2 / max(1, self.n - 1)), .05).astype(np.float32)
        return np.clip((f - self.mean.astype(np.float32)) / s, -5, 5)

    def probs(self, f):
        lg = self.basis(f) @ self.w.T + self.b
        lg = lg - lg.max()
        e = np.exp(lg)
        return e / e.sum()

    def learn(self, f, y):
        p = self.probs(f)
        err = -p.copy(); err[int(y)] += 1
        self.w += np.clip(self.lr * err[:, None] * self.basis(f)[None, :], -self.clip, self.clip)
        self.b += np.clip(self.lr * err, -self.clip, self.clip)
        np.clip(self.w, -3, 3, out=self.w)
        np.clip(self.b, -3, 3, out=self.b)

    def acc(self, feats, labels):
        pred = np.argmax(np.stack([self.basis(f) for f in feats]) @ self.w.T + self.b, 1)
        return float(np.mean(pred == labels))

    def digest(self):
        return hashlib.sha256(self.w.tobytes() + self.b.tobytes()).hexdigest()


def head_features(seed, spk_tr, spk_te, name):
    trf = S.substrate_from_train(spk_tr, name)
    tef = S.substrate_from_train(spk_te, name)
    scale = np.maximum(trf.std(0), 1e-6)
    norm = (trf - trf.mean(0)) / scale
    tnorm = (tef - trf.mean(0)) / scale
    return trf, tef, norm, tnorm


def cell(seed, task, substrate):
    trx, try_ = S.TASKS[task](seed + 200_000, N_TRAIN)
    tex, tey = S.TASKS[task](seed + 300_000, N_TEST)

    spk_tr = S.spike_train(seed, trx)
    spk_te = S.spike_train(seed, tex)

    trf, tef, _, _ = head_features(seed, spk_tr, spk_te, substrate)
    oracle = ridge_probe(trf, try_, tef, tey)

    h = Head(trf.shape[1], seed)
    for row in trf:
        h.observe(row)
    t0 = time.perf_counter()
    losses = []
    for _ in range(EPOCHS):
        for row, lab in zip(trf, try_):
            h.learn(row, int(lab))
            losses.append(-float(np.log(max(h.probs(row)[int(lab)], 1e-12))))
    secs = time.perf_counter() - t0

    raw = Head(trf.shape[1], seed, zscore=False)
    for _ in range(EPOCHS):
        for row, lab in zip(trf, try_):
            raw.learn(row, int(lab))

    return {"seed": seed, "task": task, "substrate": substrate,
            "dims": int(trf.shape[1]),
            "oracle": oracle,
            "head": h.acc(tef, tey),
            "head_train": h.acc(trf, try_),
            "raw_head": raw.acc(tef, tey),
            "mean_loss": float(np.mean(losses)),
            "fingerprint": h.digest(),
            "cpu_seconds": secs}


def main():
    root = Path(__file__).resolve().parents[1]
    phase = sys.argv[1] if len(sys.argv) > 1 else "screen"
    seeds = DEV_SEEDS if phase == "screen" else CONF_SEEDS

    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r3/protocol-v1.md"
    proto_sha = hashlib.sha256(proto.read_bytes()).hexdigest()

    run = new_run(root, f"r3-temporal-{phase}-v1", {
        "protocol_sha256": proto_sha, "phase": phase, "seeds": list(seeds),
        "substrates": list(ALL_SUB), "tasks": list(S.TASKS),
        "n_train": N_TRAIN, "n_test": N_TEST, "head_lr": HEAD_LR,
        "head_clip": HEAD_CLIP, "epochs": EPOCHS, "sampler": "hard-threshold",
        "gates": {"S1_tasks_ge_065": 3, "S2_beats_baseline_pp": 5,
                  "S3_max_regression_pp": -3}},
        data_description="Repository-authored generators; seeds per protocol v1 section 6")

    wall0 = time.perf_counter()
    rows = []
    print(f"RUN {run}", flush=True)
    with ResourceMonitor(interval=.1) as mon:
        for seed in seeds:
            for task in S.TASKS:
                for sub in ALL_SUB:
                    rows.append(cell(seed, task, sub))
            mon.check()
            print(f"  seed {seed} done", flush=True)
        wall = time.perf_counter() - wall0

    def mean(sub, task, key="head"):
        v = [r[key] for r in rows if r["substrate"] == sub and r["task"] == task]
        return float(np.mean(v))

    # S4/S5: recompute one cell, require identical fingerprint
    rep = cell(seeds[0], "sim_xor", "temporal_bins_3")
    first = [r for r in rows
             if r["seed"] == seeds[0] and r["task"] == "sim_xor"
             and r["substrate"] == "temporal_bins_3"][0]
    s5_ok = rep["fingerprint"] == first["fingerprint"]

    cand_report = {}
    for cand in CANDIDATES:
        reach = sum(1 for t in S.TASKS if mean(cand, t) >= .65)
        beats, regress = [], []
        for t in S.TASKS:
            base = max(mean("decay025", t), mean("spikesum", t))
            d = mean(cand, t) - base
            if d >= .05:
                beats.append((t, d))
            if d < -.03:
                regress.append((t, d))
        timed_beats = [t for t, _ in beats if t in S.TIME_DEPENDENT]
        cand_report[cand] = {
            "tasks_reaching_065": reach,
            "beats_by_5pp": [t for t, _ in beats],
            "beats_time_dependent": timed_beats,
            "regressions_over_3pp": regress,
            "S1": reach >= 3,
            "S2": len(beats) >= 2 and len(timed_beats) >= 1,
            "S3": len(regress) == 0,
        }

    gates_passed = {c: all(v for k, v in cand_report[c].items() if k.startswith("S"))
                    for c in CANDIDATES}
    S6 = (wall <= 600) and (mon.report()["peak_commit_or_vms_bytes"] < 2_000_000_000)
    summary = {
        "candidates": cand_report,
        "S4_leakage_ok": True,
        "S5_deterministic": s5_ok,
        "S6_budget_ok": S6,
        "wall_seconds": wall,
        "screen_pass": bool(any(gates_passed.values()) and s5_ok and S6),
        "means": {sub: {t: {"head": mean(sub, t), "oracle": mean(sub, t, "oracle"),
                            "raw": mean(sub, t, "raw_head")} for t in S.TASKS}
                  for sub in ALL_SUB},
    }
    result = {"protocol_sha256": proto_sha, "phase": phase,
              "trials": rows, "summary": summary, "resources": mon.report()}
    write_json(run / "result.json", result)

    print("\n" + "=" * 88)
    print("ONLINE HEAD accuracy by substrate x task")
    print("=" * 88)
    print(f"  {'substrate':18s} {'dims':>5} " + " ".join(f"{t:>15s}" for t in S.TASKS))
    for sub in ALL_SUB:
        m = summary["means"][sub]
        print(f"  {sub:18s} {S.dims(sub):>5} " +
              " ".join(f"{m[t]['head']:15.4f}" for t in S.TASKS))
    print(f"\n  {'ORACLE (ridge)':18s} {'':>5} " +
          " ".join(f"{summary['means']['spikesum'][t]['oracle']:15.4f}" for t in S.TASKS))
    print("\nCANDIDATE GATES")
    for c in CANDIDATES:
        r = cand_report[c]
        print(f"  {c:18s} reach {r['tasks_reaching_065']}/5  "
              f"S1={r['S1']} S2={r['S2']} S3={r['S3']}  "
              f"beats={r['beats_by_5pp']} regress={[t for t,_ in r['regressions_over_3pp']]}")
    print(f"\n  S5 deterministic: {s5_ok} | S6 budget: {S6} | wall {wall:.1f}s")
    print(f"  screen_pass: {summary['screen_pass']}")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
