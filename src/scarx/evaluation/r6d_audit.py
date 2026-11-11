"""Round 6D — audited, corrected implementations.

Every function here exists because of a specific defect found in R6A. The defect id
links to `docs/.../deepseek-r6d/audit.md`.

A1  `Scaler` reproduces R5B exactly by default (floor .05, clip ±5) and offers the R6A
    form (floor 1e-6, no clip) as an explicitly named alternative. Neither is the
    default-by-accident.
A2  `apply_features` is the ONLY way to get features for evaluation, and it always
    applies a scaler. There is no raw path a caller can stumble into.
A3  `fixed_calibration` derives statistics from a pinned, seed-independent episode set
    that is identical for every budget, so the scaler cannot drift with data quantity.
A4  `both_a` is computed and exposed as the primary relational metric; gates read it.
A5  No gate is a literal `True`.
A6  Every registered gate is computed.
"""
import hashlib

import numpy as np

from scarx.evaluation import r5_independent as G
from scarx.evaluation import r5_probe as P

QUERY_TICK = G.QUERY_TICK

DEV_SEEDS = (120011, 120017, 120023)          # R6A dev, reused for CORRECTED_REPLAY
CONF_SEEDS = (121011, 121017, 121023, 121029, 121031, 121037, 121041, 121043)
FRESH_SEEDS = (130001, 130007, 130013, 130019,
               130023, 130029, 130031, 130037)   # never used in R5B, R6A or R6D

TRAIN_OFFSET, HELD_OFFSET, PAIRED_OFFSET = 200_000, 300_000, 400_000
CALIB_OFFSET = 700_000        # A3: a SEPARATE pinned pool for calibration statistics
CALIB_SIZE = 256

BUDGETS = (1024, 16384)
MATCHED_UPDATES = 16384
CEILING = 0.7522


# ------------------------------------------------------------------ A1
class Scaler:
    """Feature scaler with an explicitly named convention.

    `r5b`  : floor .05, clip +-5     (R5B's `Head.basis`, reproduced exactly)
    `r6a`  : floor 1e-6, no clip     (R6A's `zscale`)

    Passing `convention` is mandatory. R6A's defect was that two conventions existed
    and neither caller recorded which one it used.
    """

    CONVENTIONS = ("r5b", "r6a")

    def __init__(self, mean, sd, convention):
        if convention not in self.CONVENTIONS:
            raise ValueError(f"convention must be one of {self.CONVENTIONS}")
        self.mean = np.asarray(mean, np.float64)
        self.sd = np.asarray(sd, np.float64)
        self.convention = convention

    @classmethod
    def fit(cls, feats, convention="r5b"):
        std = feats.std(0)
        floor = .05 if convention == "r5b" else 1e-6
        return cls(feats.mean(0), np.maximum(std, floor), convention)

    def __call__(self, feats):
        z = (np.asarray(feats, np.float64) - self.mean) / self.sd
        if self.convention == "r5b":
            z = np.clip(z, -5, 5)
        return np.asarray(z, np.float32)

    def digest(self):
        h = hashlib.sha256()
        h.update(self.convention.encode())
        h.update(self.mean.tobytes())
        h.update(self.sd.tobytes())
        return h.hexdigest()

    def __repr__(self):
        return f"Scaler({self.convention}, dim={len(self.mean)}, {self.digest()[:8]})"


# ------------------------------------------------------------------ A2 / A3
def raw_features(seed, x, query_tick=QUERY_TICK):
    """Unscaled bilinear map. Used ONLY to fit a Scaler, never for evaluation."""
    cap = P.capture(seed, x, query_tick=query_tick)
    state = P.state_vector(cap, "post")
    cue = G.query_cue(x, tick=query_tick)
    return P.bilinear_query_map(state, cue)


def apply_features(seed, x, scaler, query_tick=QUERY_TICK):
    """The single evaluation path: extract then ALWAYS scale (defect A2)."""
    return scaler(raw_features(seed, x, query_tick))


def fixed_calibration(seed, convention="r5b"):
    """A3: calibration drawn from a SEPARATE pinned pool, independent of budget.

    R6A fitted the scaler on `trx[:256]`, which differed between budgets because the
    generator reshuffles. Here the pool comes from its own offset and count, so every
    budget in the round sees a bit-identical scaler.
    """
    cx, _, *_ = G.independent_query_set_v1(seed + CALIB_OFFSET, CALIB_SIZE)
    return Scaler.fit(raw_features(seed, cx), convention)


# ------------------------------------------------------------------ head
class Head:
    """R5B per-example delta rule, unchanged arithmetic.

    Provenance: `r5_probe.Head.learn`, verified bit-exact by
    `test_head_update_matches_r5b` in the R6A suite (reused here).
    """

    def __init__(self, dim, seed, lr=.08, clip=.025, wmax=3.0):
        g = np.random.default_rng(seed + 991)
        self.w = g.normal(0, .03, (2, dim)).astype(np.float32)
        self.b = np.zeros(2, np.float32)
        self.lr, self.clip, self.wmax = lr, clip, wmax
        self.updates = 0
        self.clipped = 0

    def logits(self, f):
        return np.asarray(f, np.float32) @ self.w.T + self.b

    def probs(self, f):
        lg = self.logits(f)
        lg = lg - lg.max(axis=-1, keepdims=True)
        e = np.exp(lg)
        return e / e.sum(axis=-1, keepdims=True)

    def learn(self, f, y):
        f = np.asarray(f, np.float32)
        p = self.probs(f)
        err = -p.copy()
        err[int(y)] += 1
        dw = self.lr * err[:, None] * f[None, :]
        self.clipped += int(np.any(np.abs(dw) >= self.clip))
        self.updates += 1
        self.w += np.clip(dw, -self.clip, self.clip).astype(np.float32)
        self.b += np.clip(self.lr * err, -self.clip, self.clip).astype(np.float32)
        np.clip(self.w, -self.wmax, self.wmax, out=self.w)
        np.clip(self.b, -self.wmax, self.wmax, out=self.b)

    def predict(self, f):
        return np.argmax(self.logits(f), 1)

    def acc(self, f, y):
        return float(np.mean(self.predict(f) == y))

    def digest(self):
        return hashlib.sha256(self.w.tobytes() + self.b.tobytes()).hexdigest()


def train_head(f_tr, y_tr, seed, order, epochs=1):
    h = Head(f_tr.shape[1], seed)
    for _ in range(epochs):
        for i in order:
            h.learn(f_tr[i], int(y_tr[i]))
    return h


# ------------------------------------------------------------------ metrics
def paired_metrics(pred, y, vA, vB, flips):
    a, b = pred[0::2], pred[1::2]
    ya, yb = y[0::2], y[1::2]
    fA, fB = flips, ~flips
    return {
        "both_A": float(np.mean((a[fA] == ya[fA]) & (b[fA] == yb[fA]))) if fA.any() else float("nan"),
        "both_B": float(np.mean((a[fB] == ya[fB]) & (b[fB] == yb[fB]))) if fB.any() else float("nan"),
        "both_all": float(np.mean((a == ya) & (b == yb))),
    }


def evaluate(h, d, scaler=None, query_tick=QUERY_TICK):
    """A4: returns both_A prominently; every gate reads from here.

    `d["f_te"]` / `d["f_pt"]` are already scaled inside `build_cell`, so the common
    path needs no scaler argument. When `scaler` is given it is applied to the RAW
    tensors instead, which is what the stress path must do (defect A2).
    """
    if scaler is None:
        f_te, f_pt = d["f_te"], d["f_pt"]
    else:
        f_te = apply_features(d["seed"], d["tex"], scaler, query_tick)
        f_pt = apply_features(d["seed"], d["ptx"], scaler, query_tick)
    pred = h.predict(f_te)
    pm = paired_metrics(h.predict(f_pt), d["pty"], d["pvA"], d["pvB"], d["pflips"])
    need_b = (d["q"] == 1) & (d["vA"] != d["vB"])
    return {
        "direct": float(np.mean(pred == d["y_te"])),
        "both_A": pm["both_A"], "both_B": pm["both_B"], "both_all": pm["both_all"],
        "acc_qA": float(np.mean(pred[d["q"] == 0] == d["y_te"][d["q"] == 0])),
        "acc_qB": float(np.mean(pred[d["q"] == 1] == d["y_te"][d["q"] == 1])),
        "acc_needB": float(np.mean(pred[need_b] == d["y_te"][need_b])) if need_b.any() else float("nan"),
        "train": h.acc(d["f_tr"], d["y_tr"]),
        "clip_fraction": h.clipped / max(1, h.updates),
        "updates": h.updates,
        "head_digest": h.digest(),
    }


def build_cell(seed, n_unique, scaler, held=1024, pairs=256):
    """One (seed, budget, scaler) cell. Features are scaled exactly once, here."""
    trx, try_, *_ = G.independent_query_set_v1(seed + TRAIN_OFFSET, n_unique)
    tex, tey, evA, evB, eq, _ = G.independent_query_set_v1(seed + HELD_OFFSET, held)
    ptx, pty, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(
        seed + PAIRED_OFFSET, pairs)
    return {
        "seed": seed, "n_unique": n_unique,
        "f_tr": scaler(raw_features(seed, trx)),
        "f_te": scaler(raw_features(seed, tex)),
        "f_pt": scaler(raw_features(seed, ptx)),
        "y_tr": try_, "y_te": tey, "vA": evA, "vB": evB, "q": eq,
        "pty": pty, "pvA": pvA, "pvB": pvB, "pflips": flips,
        "tex": tex, "ptx": ptx, "trx": trx,
        "digests": {"train": G.dataset_sha256(trx, try_),
                    "held": G.dataset_sha256(tex, tey),
                    "paired": G.dataset_sha256(ptx, pty),
                    "scaler": scaler.digest()},
    }
