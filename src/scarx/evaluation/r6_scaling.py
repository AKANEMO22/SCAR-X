"""Round 6A — learning-scaling harness.

Separates the three causes the brief demands:

  1. more UNIQUE examples            -> vary `n_unique`
  2. more UPDATE steps               -> vary `updates_per_example` / epochs
  3. representation / dynamics change -> frozen vs permitted hidden plasticity

The decisive control is the matched-update design: hold total updates fixed at U while
varying the number of unique examples. If repeated data gives the same held-out gain,
the improvement is not about new knowledge.

Provenance of the copied R5B pieces
-----------------------------------
  source branch : research/deepseek/r5-independent-relational-binding
  source commit : d6c609df6f90fc8df96fb3de424dd54388f8dc59
  r5_independent.py sha256 : 74188f9e6a46a500a209a11d00c25ee2...
  r5_probe.py       sha256 : 38e025d27b359c5519987b5077054c9d...
The functions `independent_query_set_v1` / `independent_paired_set_v1` are used
verbatim; `tests/test_r6_scaling.py` asserts behavioural equivalence.
"""
import hashlib

import numpy as np

from scarx.evaluation import r5_independent as G
from scarx.evaluation import r5_probe as P

QUERY_TICK = G.QUERY_TICK

# ---- registered seed bank (protocol section D) -----------------------------
DEV_SEEDS = (120011, 120017, 120023)
CONF_SEEDS = (121011, 121017, 121023, 121029, 121031, 121037, 121041, 121043)
TRAIN_OFFSET = 200_000
HELD_OFFSET = 300_000
PAIRED_OFFSET = 400_000

BUDGETS = (256, 1024, 4096, 16384)
CEILING = 0.7522          # always-answer-key-A, measured in R5B stage 0


# ---------------------------------------------------------------- features
def project(seed, dim=32):
    """The R5 fixed projection, same construction and same seed convention."""
    return None      # placeholder to make the dependency explicit; see bilinear()


def capture_features(seed, x, query_tick=QUERY_TICK):
    """Frozen ALIF state -> R5 bilinear map. The only feature path used here."""
    cap = P.capture(seed, x, query_tick=query_tick)
    state = P.state_vector(cap, "post")
    cue = G.query_cue(x, tick=query_tick)
    return P.bilinear_query_map(state, cue)


def zscale(train, apply_to):
    """Registered feature scaling. Statistics come from TRAIN ONLY."""
    mu = train.mean(0)
    sd = np.maximum(train.std(0), 1e-6)
    return (apply_to - mu) / sd


# ------------------------------------------------------------- delta head
class Head:
    """The registered R5 local delta rule, VERBATIM update, instrumented for scaling.

    The update is PER EXAMPLE, exactly as R5B's `r5_probe.Head.learn`:

        p  = softmax(f @ W.T + b)          for ONE example
        err = onehot(y) - p
        W += clip(lr * err[:,None] * f[None,:], -clip, clip)
        b += clip(lr * err, -clip, clip)
        W, b = clip(W, -3, 3), clip(b, -3, 3)

    I first wrote this as a BATCH update (averaging over the whole array). That is a
    different learner, and the brief forbids changing the readout function partway
    through a scaling study. The per-example form is retained so that every budget in
    this round trains the SAME rule R5B used; only the number of examples and the
    number of passes change.

    Extra fields are instrumentation only (update norm, clip fraction, loss, margins).
    They never influence the update.
    """

    def __init__(self, dim, seed, lr=.08, clip=.025, wmax=3.0):
        g = np.random.default_rng(seed + 991)
        self.w = g.normal(0, .03, (2, dim)).astype(np.float32)
        self.b = np.zeros(2, np.float32)
        self.lr, self.clip, self.wmax = lr, clip, wmax
        self.updates = 0
        self.clipped = 0
        self.norm_sum = 0.0
        self.loss_sum = 0.0
        self.loss_trace = []

    def logits(self, f):
        return np.asarray(f, np.float32) @ self.w.T + self.b

    def probs(self, f):
        lg = self.logits(f)
        lg = lg - lg.max(axis=-1, keepdims=True)
        e = np.exp(lg)
        return e / e.sum(axis=-1, keepdims=True)

    def learn(self, f, y):
        """One per-example delta update. Identical arithmetic to R5B."""
        f = np.asarray(f, np.float32)
        p = self.probs(f)
        err = -p.copy()
        err[int(y)] += 1
        delta_w = self.lr * err[:, None] * f[None, :]
        delta_b = self.lr * err
        self.clipped += int(np.any(np.abs(delta_w) >= self.clip))
        self.updates += 1
        self.w += np.clip(delta_w, -self.clip, self.clip).astype(np.float32)
        self.b += np.clip(delta_b, -self.clip, self.clip).astype(np.float32)
        np.clip(self.w, -self.wmax, self.wmax, out=self.w)
        np.clip(self.b, -self.wmax, self.wmax, out=self.b)
        self.norm_sum += float(np.linalg.norm(delta_w))
        loss = float(-np.log(max(p[int(y)], 1e-12)))
        self.loss_sum += loss
        if self.updates % 64 == 0:
            self.loss_trace.append(loss)

    def predict(self, f):
        return np.argmax(self.logits(f), 1)

    def acc(self, f, y):
        return float(np.mean(self.predict(f) == y))

    def margins(self, f):
        lg = self.logits(f)
        return float(np.mean(np.abs(lg[:, 1] - lg[:, 0])))

    def digest(self):
        return hashlib.sha256(self.w.tobytes() + self.b.tobytes()).hexdigest()


def train_head(f_tr, y_tr, seed, order, epochs=1, lr=.08, clip=.025):
    """One locked training pass over `order`, one epoch (or more if `epochs` > 1).

    `order` is a permutation of example indices; applying it `epochs` times is the
    repeated-data arm. Total updates = len(order) * epochs.
    """
    h = Head(f_tr.shape[1], seed, lr=lr, clip=clip)
    for _ in range(epochs):
        for i in order:
            h.learn(f_tr[i], int(y_tr[i]))
    return h


# ----------------------------------------------------------------- metrics
def paired_metrics(pred, y, vA, vB, flips):
    a, b = pred[0::2], pred[1::2]
    ya, yb = y[0::2], y[1::2]
    fA, fB = flips, ~flips
    out = {
        "both_A": float(np.mean((a[fA] == ya[fA]) & (b[fA] == yb[fA]))) if fA.any() else float("nan"),
        "both_B": float(np.mean((a[fB] == ya[fB]) & (b[fB] == yb[fB]))) if fB.any() else float("nan"),
        "both_all": float(np.mean((a == ya) & (b == yb))),
    }
    return out


def full_report(h, f_te, y_te, vA, vB, q, f_pt, pty, pvA, pvB, pflips,
                f_tr, y_tr, ridge_acc):
    """Every metric the brief lists for a budget checkpoint."""
    pred = h.predict(f_te)
    pm = paired_metrics(h.predict(f_pt), pty, pvA, pvB, pflips)
    need_b = (q == 1) & (vA != vB)
    return {
        "direct": float(np.mean(pred == y_te)),
        "train": h.acc(f_tr, y_tr),
        "both_A": pm["both_A"], "both_B": pm["both_B"], "both_all": pm["both_all"],
        "acc_qA": float(np.mean(pred[q == 0] == y_te[q == 0])),
        "acc_qB": float(np.mean(pred[q == 1] == y_te[q == 1])),
        "acc_needB": float(np.mean(pred[need_b] == y_te[need_b])) if need_b.any() else float("nan"),
        "loss": h.loss_sum / max(1, h.updates),
        "margin": h.margins(f_te),
        "update_norm": h.norm_sum / max(1, h.updates),
        "clip_fraction": h.clipped / max(1, h.updates),
        "ridge": ridge_acc,
        "ridge_gap": ridge_acc - float(np.mean(pred == y_te)),
        "updates": h.updates,
        "head_digest": h.digest(),
        "weight_norm": float(np.linalg.norm(h.w)),
    }
