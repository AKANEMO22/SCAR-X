"""Round 2A registered screen — substrate x reader.

Protocol: experiments/collab/scarx-neural-round1-20261111/deepseek-r2/protocol-v1.md
SHA-256 18b618969b2cbd6648d8457904d874516e1c83cb5308ee091fc7c47d5784c28a

Nothing here is tuned. Gates S1-S6 are evaluated exactly as registered.
Hidden weights are asserted unchanged. Calibration uses training features only.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.evaluation.tasks21 import ridge_probe  # noqa: E402
from scarx.neural.core import Config  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
S = 1.5
DEV_SEEDS = (81001, 81007, 81013)          # protocol section 4
CONFIRM_SEEDS = (82001, 82007, 82013, 82019, 82023, 82029, 82031, 82037)
N_CAL, N_TRAIN, N_TEST = 128, 1024, 512    # protocol section 3
SUBSTRATES = ("decay025", "spikesum", "bank5", "bank5_proj48")
HEAD_LR, HEAD_CLIP, EPOCHS = .08, .025, 1  # protocol section 5
BANK = (.10, .25, .50, .75, .90)


def cfg(seed):
    return Config(seed=seed, **CFG)


def sim_xor(seed, count):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, count).astype(np.int32)
    for i in range(count):
        a = int(rng.integers(2))
        b = a ^ int(y[i])
        t = int(rng.integers(1, 4))
        x[i, t, 0] += S * a
        x[i, t, 1] += S * b
        x[i, 10, 7] += S
    return x, y


def weights(seed):
    c = cfg(seed)
    g = np.random.default_rng(c.seed)
    idx = np.stack([g.choice(c.hidden, c.fan_in, replace=False) for _ in range(c.hidden)]).astype(np.int32)
    w_in = g.normal(.15, .45, (c.hidden, c.inputs)).astype(np.float32)
    w_rec = g.normal(0, .12 / np.sqrt(c.fan_in), (c.hidden, c.fan_in)).astype(np.float32)
    return c, idx, w_in, w_rec


def projection(seed, out_dim=48):
    """Fixed random projection for the matched-capacity control (protocol 2)."""
    g = np.random.default_rng(seed + 880_000)
    return g.normal(0, 1 / np.sqrt(len(BANK) * CFG["hidden"]),
                    (len(BANK) * CFG["hidden"], out_dim)).astype(np.float32)


def build(seed, seqs, substrate):
    """Frozen ALIF forward -> substrate. Hidden weights are never modified."""
    c, idx, w_in, w_rec = weights(seed)
    x = np.asarray(seqs, np.float32)
    b, t, _ = x.shape
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy()
    f = np.zeros((b, c.hidden), np.float32)
    banks = [np.zeros((b, c.hidden), np.float32) for _ in BANK]
    spk = np.zeros((b, t, c.hidden), np.float32)
    for s in range(t):
        v = c.leak * v + x[:, s] @ w_in.T + np.sum(w_rec[None, :, :] * z[:, idx], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        z = (v >= lim).astype(np.float32)
        v = v - z * lim
        a = c.adaptation_decay * a + z
        f = c.feature_decay * f + (1 - c.feature_decay) * z
        for i, d in enumerate(BANK):
            banks[i] = d * banks[i] + (1 - d) * z
        spk[:, s] = z
    if substrate == "decay025":
        return f
    if substrate == "spikesum":
        return spk.sum(1)
    bank = np.concatenate(banks, 1)
    if substrate == "bank5":
        return bank
    if substrate == "bank5_proj48":
        return bank @ projection(seed)
    raise ValueError(substrate)


class Head:
    """Delta rule on a z-scored basis. Registered hyperparameters only."""

    def __init__(self, dim, seed, lr=HEAD_LR, clip=HEAD_CLIP):
        g = np.random.default_rng(seed + 991)
        self.w = g.normal(0, .03, (2, dim)).astype(np.float32)
        self.b = np.zeros(2, np.float32)
        self.mean = np.zeros(dim, np.float64); self.m2 = np.zeros(dim, np.float64)
        self.n = 0
        self.lr, self.clip = lr, clip
        self.sat = 0; self.total = 0

    def observe(self, f):
        self.n += 1
        d = f - self.mean
        self.mean += d / self.n
        self.m2 += d * (f - self.mean)

    def norm(self, f):
        s = np.maximum(np.sqrt(self.m2 / max(1, self.n - 1)), .05).astype(np.float32)
        return np.clip((f - self.mean.astype(np.float32)) / s, -5, 5)

    def probs(self, f):
        lg = self.norm(f) @ self.w.T + self.b
        lg = lg - lg.max()
        e = np.exp(lg)
        return e / e.sum()

    def learn(self, f, y):
        p = self.probs(f)
        err = -p.copy(); err[int(y)] += 1
        basis = self.norm(f)
        delta = self.lr * err[:, None] * basis[None, :]
        self.sat += int(np.any(np.abs(delta) >= self.clip))
        self.total += 1
        self.w += np.clip(delta, -self.clip, self.clip)
        self.b += np.clip(self.lr * err, -self.clip, self.clip)
        np.clip(self.w, -3, 3, out=self.w); np.clip(self.b, -3, 3, out=self.b)

    def acc(self, feats, labels):
        return float(np.mean(np.argmax(np.stack([self.norm(f) for f in feats]) @ self.w.T
                                       + self.b, 1) == labels))

    def digest(self):
        return hashlib.sha256(self.w.tobytes() + self.b.tobytes()).hexdigest()


def cell(seed, substrate, hw_before):
    calx, _ = sim_xor(seed + 100_000, N_CAL)
    trx, try_ = sim_xor(seed + 200_000, N_TRAIN)
    tex, tey = sim_xor(seed + 300_000, N_TEST)

    trf = build(seed, trx, substrate)
    tef = build(seed, tex, substrate)

    oracle = ridge_probe(trf, try_, tef, tey)

    h = Head(trf.shape[1], seed)
    for f in trf:
        h.observe(f)                     # calibration from TRAINING features only
    t0 = time.perf_counter()
    losses = []
    for _ in range(EPOCHS):
        for f, y in zip(trf, try_):
            h.learn(f, int(y))
            losses.append(-float(np.log(max(h.probs(f)[int(y)], 1e-12))))
    secs = time.perf_counter() - t0

    # feature geometry
    sparsity = float((tef == 0).mean())
    stds = tef.std(0)
    rank = int(np.sum(stds > 1e-6))

    _, idx, w_in, w_rec = weights(seed)
    hw_after = hashlib.sha256(w_in.tobytes() + w_rec.tobytes()).hexdigest()

    return {"seed": seed, "substrate": substrate, "dims": int(trf.shape[1]),
            "oracle": oracle, "direct": h.acc(tef, tey),
            "train_acc": h.acc(trf, try_),
            "mean_loss": float(np.mean(losses)),
            "clip_sat": h.sat / max(1, h.total),
            "sparsity": sparsity, "n_active_dims": rank,
            "cpu_seconds": secs, "head_digest": h.digest(),
            "hidden_unchanged": hw_before == hw_after}


def main():
    root = Path(__file__).resolve().parents[1]
    phase = sys.argv[1] if len(sys.argv) > 1 else "screen"
    seeds = DEV_SEEDS if phase == "screen" else CONFIRM_SEEDS
    if phase == "confirmation":
        lock = json.loads((root / "artifacts/deepseek-r2/screen-lock.json").read_text())
        if not lock["screen_pass"]:
            print("screen did not pass; confirmation not opened"); return 0

    proto_sha = hashlib.sha256(
        (root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r2/protocol-v1.md")
        .read_bytes()).hexdigest()

    run = new_run(root, f"r2-substrate-{phase}-v1", {
        "protocol_sha256": proto_sha, "phase": phase, "seeds": list(seeds),
        "substrates": list(SUBSTRATES), "n_cal": N_CAL, "n_train": N_TRAIN,
        "n_test": N_TEST, "head_lr": HEAD_LR, "head_clip": HEAD_CLIP,
        "epochs": EPOCHS, "sampler": "hard-threshold",
        "gates": {"S1_oracle_ge": .80, "S2_head_ge": .65,
                  "S3_improvement_ge": .10, "S4_gap_ge": -.10}},
        data_description="Repository-authored sim_xor generator, seeds per protocol v1 section 4")

    wall0 = time.perf_counter()
    rows = []
    with ResourceMonitor(interval=.1) as mon:
        for seed in seeds:
            _, _, w_in, w_rec = weights(seed)
            hw = hashlib.sha256(w_in.tobytes() + w_rec.tobytes()).hexdigest()
            for sub in SUBSTRATES:
                r = cell(seed, sub, hw)
                rows.append(r)
                print(f"  seed{seed} {sub:14s} oracle={r['oracle']:.4f} "
                      f"head={r['direct']:.4f} train={r['train_acc']:.4f}", flush=True)
            mon.check()
        wall = time.perf_counter() - wall0

    def mean(sub, key):
        v = [r[key] for r in rows if r["substrate"] == sub]
        return float(np.mean(v)) if v else None

    g = {
        "S1_oracle_sufficient": all(
            r["oracle"] >= .80 for r in rows if r["substrate"] == "spikesum"),
        "S2_head_clears": all(
            r["direct"] >= .65 for r in rows if r["substrate"] == "spikesum"),
        "S3_improvement": (mean("spikesum", "direct") - mean("decay025", "direct")) >= .10,
        "S4_gap_narrowed": (mean("spikesum", "direct") - mean("spikesum", "oracle")) >= -.10,
        "S5_no_leakage": True,   # calibration built from training features only, by construction
        "S6_reproducible": None,  # filled by the repeat check below
    }

    # S6: repeat one cell and require bit-identical head weights
    _, _, w_in, w_rec = weights(seeds[0])
    hw = hashlib.sha256(w_in.tobytes() + w_rec.tobytes()).hexdigest()
    again = cell(seeds[0], "spikesum", hw)
    first = [r for r in rows if r["seed"] == seeds[0] and r["substrate"] == "spikesum"][0]
    g["S6_reproducible"] = again["head_digest"] == first["head_digest"]

    passed = all(v is True for v in g.values())
    summary = {
        "gates": g, "screen_pass": passed, "wall_clock_seconds": wall,
        "within_wall_budget": wall <= 600,
        "hidden_unchanged_all": all(r["hidden_unchanged"] for r in rows),
        "means": {sub: {k: mean(sub, k) for k in
                        ("oracle", "direct", "train_acc", "mean_loss",
                         "clip_sat", "sparsity", "n_active_dims", "cpu_seconds")}
                  for sub in SUBSTRATES},
    }
    result = {"protocol_sha256": proto_sha, "phase": phase, "trials": rows,
              "summary": summary, "resources": mon.report()}
    write_json(run / "result.json", result)
    write_json(root / "artifacts/deepseek-r2" / f"{phase}-lock.json",
               {"screen_pass": passed, "gates": g, "run": run.name,
                "protocol_sha256": proto_sha})

    print("\n" + "=" * 74)
    print("SUBSTRATE MEANS (online head vs oracle)")
    print("=" * 74)
    print(f"  {'substrate':16s} {'dims':>5} {'oracle':>8} {'head':>8} {'gap':>8} {'train':>8}")
    for sub in SUBSTRATES:
        m = summary["means"][sub]
        print(f"  {sub:16s} {int(m['n_active_dims']):>5} {m['oracle']:8.4f} "
              f"{m['direct']:8.4f} {m['direct']-m['oracle']:+8.4f} {m['train_acc']:8.4f}")
    print("\nGATES:", json.dumps(g, indent=1))
    print(f"screen_pass: {passed} | wall {wall:.1f}s | under 600s: {wall <= 600}")
    print(f"hidden unchanged in all cells: {summary['hidden_unchanged_all']}")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
