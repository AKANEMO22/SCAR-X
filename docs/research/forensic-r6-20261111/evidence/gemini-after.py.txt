"""SCAR-X Round 6C evaluation-only module: Context-Keyed Competitive Eligibility Readout (CKCER).

Implements:
- CKCERHead (Mode A: explicit query cue; Mode B: 100% neural-derived query via unsupervised local Oja gate;
  plus un-gated protected-task mode)
- CKCERStreamRunner (tick-by-tick bounded local eligibility state runner on SpikingCore)
- capture_r6 (batch-vectorized equivalent of CKCERStreamRunner + R5 baseline features)
- All 10 Stage 2 mandatory baselines and controls
- Single-factor ablation ladder from R5 to CKCER
- Auxiliary generalization benchmarks (structured third write & 3-write key overwrite)
- Hardest-subset counterfactual diagnostics (Section IX.A) and paired bootstrap statistics
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from scarx.evaluation import r5_independent as G
from scarx.evaluation import r5_probe as P
from scarx.neural.core import SpikingCore


def enforce_single_thread_env() -> dict[str, str]:
    """Ensure deterministic single-thread CPU execution metadata."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    return {k: os.environ.get(k, "") for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}


@dataclass
class CKCERConfig:
    """Locked hyperparameters for CKCER (pre-registered in protocol-v1.md)."""

    lr: float = 0.08
    clip_step: float = 0.025
    weight_clip: float = 3.0
    dead_std: float = 1e-4
    gamma_c: float = 1.0
    gamma_a: float = 0.95
    lambda_v: float = 0.65
    oja_lr: float = 0.02
    oja_epochs: int = 10
    obs_steps: int = 1024
    train_passes: int = 1


class CKCERStreamRunner:
    """Tick-by-tick streaming eligibility runner on a frozen SpikingCore(P.cfg(seed)).

    Maintains only bounded local eligibility state per neuron:
    - c_j(t) = gamma_c * c_j(t-1) + z_j(t)  (48 floats)
    - a_j(t) = core.adaptation              (48 floats, already inside SpikingCore)
    - v_prev = v(t-1)                       (48 floats, for somatic transient at t_q)
    """

    def __init__(self, core: SpikingCore, cfg: CKCERConfig | None = None) -> None:
        self.core = core
        self.cfg = cfg or CKCERConfig(lambda_v=float(core.config.leak))
        h = self.core.config.hidden
        self.c_trace = np.zeros(h, dtype=np.float32)
        self.v_prev = np.zeros(h, dtype=np.float32)
        self.m_pre_latched = np.zeros(2 * h, dtype=np.float32)
        self.delta_q_latched = np.zeros(2 * h, dtype=np.float32)

    def reset_episode(self) -> None:
        self.core.reset_state()
        self.c_trace.fill(0.0)
        self.v_prev.fill(0.0)
        self.m_pre_latched.fill(0.0)
        self.delta_q_latched.fill(0.0)

    def run_episode(self, x_seq: np.ndarray, query_tick: int = G.QUERY_TICK) -> dict[str, np.ndarray]:
        """Step a single sequence x_seq of shape (T, C) tick by tick."""
        self.reset_episode()
        t_steps = x_seq.shape[0]
        for t in range(t_steps):
            if t == query_tick:
                self.m_pre_latched = np.concatenate(
                    [self.c_trace.copy(), self.core.adaptation.copy()], axis=0
                ).astype(np.float32)
                self.v_prev = self.core.voltage.copy()
            self.core.step(x_seq[t], track=False)
            self.c_trace = self.cfg.gamma_c * self.c_trace + self.core.spikes
            if t == query_tick:
                dv_q = self.core.voltage - np.float32(self.core.config.leak) * self.v_prev
                self.delta_q_latched = np.concatenate(
                    [self.core.spikes.copy(), dv_q.copy()], axis=0
                ).astype(np.float32)
        return {
            "m_pre": self.m_pre_latched.copy(),
            "delta_q": self.delta_q_latched.copy(),
            "sum_s": self.c_trace.copy(),
        }


def capture_r6(
    seed: int,
    x: np.ndarray,
    query_tick: int | None = None,
    cfg: CKCERConfig | None = None,
) -> dict[str, np.ndarray]:
    """Run frozen ALIF core across batch x (N, T, C) and extract CKCER + baseline states."""
    cfg = cfg or CKCERConfig()
    x_arr = np.asarray(x, dtype=np.float32)
    qt = G.locate_query_tick(x_arr[0]) if query_tick is None else int(query_tick)
    cap = P.capture(seed, x_arr, query_tick=qt)

    spk = cap["spike_train"]  # (N, T, 48)
    c_pre = spk[:, :qt, :].sum(axis=1).astype(np.float32)
    a_pre = cap["pre"]["adaptation"].astype(np.float32)
    m_pre = np.concatenate([c_pre, a_pre], axis=1).astype(np.float32)

    leak = np.float32(P.CFG["leak"])
    z_q = cap["post"]["spikes"].astype(np.float32)
    dv_q = (cap["post"]["voltage"] - leak * cap["pre"]["voltage"]).astype(np.float32)
    delta_q = np.concatenate([z_q, dv_q], axis=1).astype(np.float32)

    pre_full = P.state_vector(cap, "pre").astype(np.float32)
    post_full = P.state_vector(cap, "post").astype(np.float32)
    sum_s = cap["final"]["spikesum"].astype(np.float32)
    bank5 = cap["final"]["bank5"].astype(np.float32)

    mid = max(1, spk.shape[1] // 2)
    c_early = spk[:, :mid, :].sum(axis=1).astype(np.float32)
    c_late = spk[:, mid:, :].sum(axis=1).astype(np.float32)
    early_late = np.concatenate([c_early, c_late, c_early - c_late], axis=1).astype(np.float32)

    try:
        cue = G.query_cue(x_arr, tick=qt).astype(np.float32)
    except Exception:
        cue = np.zeros((x_arr.shape[0], 2), dtype=np.float32)

    return {
        "m_pre": m_pre,
        "delta_q": delta_q,
        "cue": cue,
        "pre": pre_full,
        "post": post_full,
        "sum_s": sum_s,
        "bank5": bank5,
        "early_late": early_late,
        "query_tick": qt,
    }


class CKCERHead:
    """Context-Keyed Competitive Eligibility Readout (CKCER).

    Supports:
    - mode="A": Explicit legal 2-channel query cue g(t_q) = [x_ch4(t_q), x_ch5(t_q)].
                390 learnable parameters (2 x 194 weights + 2 biases), 0 random projection bytes.
    - mode="B": 100% neural-derived query gate learned unsupervised via Oja's local rule
                from somatic transient delta_q(t_q) = [z(t_q), v(t_q) - leak * v(t_q-1)].
                Never accesses external query cue channels (4, 5) during observe, train, or predict.
                486 learnable parameters (390 readout + 96 Oja gate), 0 random projection bytes.
    - mode="ungated": Un-gated standardized readout for query-free protected tasks (sim_xor, xor, order).
    """

    def __init__(
        self,
        dim: int = 96,
        mode: str = "A",
        seed: int = 0,
        cfg: CKCERConfig | None = None,
    ) -> None:
        if mode not in ("A", "B", "ungated"):
            raise ValueError(f"Unsupported CKCER mode: {mode}")
        self.dim = dim
        self.mode = mode
        self.seed = seed
        self.cfg = cfg or CKCERConfig()

        # Pre-gating homeostatic calibration statistics on m_pre
        self.mean_m = np.zeros((1, dim), dtype=np.float32)
        self.std_m = np.ones((1, dim), dtype=np.float32)
        self.active_m = np.ones((1, dim), dtype=bool)
        self.calibrated = False

        # Mode B unsupervised Oja dendritic gate state
        self.mean_d = np.zeros((1, dim), dtype=np.float32)
        self.std_d = np.ones((1, dim), dtype=np.float32)
        self.active_d = np.ones((1, dim), dtype=bool)
        rng = np.random.default_rng(seed + 701)
        w_q_init = rng.standard_normal(dim).astype(np.float32)
        self.w_q = w_q_init / max(float(np.linalg.norm(w_q_init)), 1e-8)
        self.thresh_q: float = 0.0

        self.feat_dim = (2 * dim + 2) if mode in ("A", "B") else dim
        self.w = np.zeros((2, self.feat_dim), dtype=np.float32)
        self.b = np.zeros((2,), dtype=np.float32)

    @property
    def learnable_parameter_count(self) -> int:
        base = int(self.w.size + self.b.size)
        return base + int(self.w_q.size) if self.mode == "B" else base

    @property
    def random_projection_bytes(self) -> int:
        return 0

    def _standardize_m(self, m: np.ndarray) -> np.ndarray:
        z = np.zeros_like(m, dtype=np.float32)
        mask = self.active_m[0]
        if np.any(mask):
            z[:, mask] = np.clip((m[:, mask] - self.mean_m[:, mask]) / self.std_m[:, mask], -5.0, 5.0)
        return z

    def _standardize_d(self, d: np.ndarray) -> np.ndarray:
        z = np.zeros_like(d, dtype=np.float32)
        mask = self.active_d[0]
        if np.any(mask):
            z[:, mask] = np.clip((d[:, mask] - self.mean_d[:, mask]) / self.std_d[:, mask], -5.0, 5.0)
        return z

    def observe(self, m: np.ndarray, delta_q: np.ndarray | None = None) -> None:
        """Unsupervised pre-gating homeostatic calibration (and unsupervised Oja gate in Mode B).

        Labels y and external cues are NEVER passed to observe().
        """
        m_cal = m[: self.cfg.obs_steps]
        ddof = 1 if m_cal.shape[0] > 1 else 0
        self.mean_m = m_cal.mean(axis=0, keepdims=True).astype(np.float32)
        raw_std = m_cal.std(axis=0, ddof=ddof, keepdims=True).astype(np.float32)
        self.active_m = raw_std > self.cfg.dead_std
        self.std_m = np.where(self.active_m, raw_std, 1.0).astype(np.float32)

        if self.mode == "B":
            if delta_q is None:
                raise ValueError("Mode B requires somatic transient delta_q during observe().")
            d_cal = delta_q[: self.cfg.obs_steps]
            ddof_d = 1 if d_cal.shape[0] > 1 else 0
            self.mean_d = d_cal.mean(axis=0, keepdims=True).astype(np.float32)
            d_std = d_cal.std(axis=0, ddof=ddof_d, keepdims=True).astype(np.float32)
            self.active_d = d_std > self.cfg.dead_std
            self.std_d = np.where(self.active_d, d_std, 1.0).astype(np.float32)
            d_hat = self._standardize_d(d_cal)

            # Unsupervised local Oja plasticity: Δw_q = η_oja * u * (d_hat - u * w_q)
            w_q = self.w_q.copy()
            eta = np.float32(self.cfg.oja_lr)
            for _ in range(self.cfg.oja_epochs):
                for i in range(d_hat.shape[0]):
                    x_i = d_hat[i]
                    u = float(np.dot(w_q, x_i))
                    w_q = w_q + eta * np.float32(u) * (x_i - np.float32(u) * w_q)
                    norm = float(np.linalg.norm(w_q))
                    if norm > 1e-8:
                        w_q = w_q / np.float32(norm)
            self.w_q = w_q.astype(np.float32)
            self.thresh_q = 0.0

        self.calibrated = True

    def compute_gate(
        self,
        cue: np.ndarray | None = None,
        delta_q: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return binary context gate g(t_q) in {0, 1}^{N x 2}."""
        if self.mode == "A":
            if cue is None:
                raise ValueError("CKCER Mode A requires explicit 2-channel cue.")
            return cue.astype(np.float32)
        if self.mode == "B":
            if delta_q is None:
                raise ValueError("CKCER Mode B requires somatic transient delta_q.")
            d_hat = self._standardize_d(delta_q)
            proj = d_hat @ self.w_q
            g1 = (proj >= self.thresh_q).astype(np.float32)
            g0 = 1.0 - g1
            return np.stack([g0, g1], axis=1).astype(np.float32)
        raise ValueError("compute_gate is only valid for gated modes A and B.")

    def build_features(
        self,
        m: np.ndarray,
        cue: np.ndarray | None = None,
        delta_q: np.ndarray | None = None,
    ) -> np.ndarray:
        """Construct pre-gated standardized representation psi(hat_m, g)."""
        if not self.calibrated:
            self.observe(m, delta_q=delta_q)
        m_hat = self._standardize_m(m)
        if self.mode == "ungated":
            return m_hat
        g = self.compute_gate(cue=cue, delta_q=delta_q)
        g0 = g[:, 0:1]
        g1 = g[:, 1:2]
        # Purely context-keyed representation: inactive branch is identically 0.0
        return np.concatenate([m_hat * g0, m_hat * g1, g0, g1], axis=1).astype(np.float32)

    def scores_from_features(self, psi: np.ndarray) -> np.ndarray:
        return (psi @ self.w.T + self.b[None, :]).astype(np.float32)

    def scores(
        self,
        m: np.ndarray,
        cue: np.ndarray | None = None,
        delta_q: np.ndarray | None = None,
    ) -> np.ndarray:
        psi = self.build_features(m, cue=cue, delta_q=delta_q)
        return self.scores_from_features(psi)

    def predict(
        self,
        m: np.ndarray,
        cue: np.ndarray | None = None,
        delta_q: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.argmax(self.scores(m, cue=cue, delta_q=delta_q), axis=1).astype(np.int64)

    def step_update(self, psi_i: np.ndarray, y_i: int) -> tuple[np.ndarray, np.ndarray]:
        """Single-example local associative update using SpikingCore's native output-local delta rule."""
        sc = self.w @ psi_i + self.b
        sc = sc - np.max(sc)
        exp_sc = np.exp(sc)
        probs = (exp_sc / np.sum(exp_sc)).astype(np.float32)
        err = -probs
        err[int(y_i)] += 1.0
        dw = np.clip(self.cfg.lr * np.outer(err, psi_i), -self.cfg.clip_step, self.cfg.clip_step).astype(np.float32)
        db = np.clip(self.cfg.lr * err, -self.cfg.clip_step, self.cfg.clip_step).astype(np.float32)
        self.w = np.clip(self.w + dw, -self.cfg.weight_clip, self.cfg.weight_clip).astype(np.float32)
        self.b = np.clip(self.b + db, -self.cfg.weight_clip, self.cfg.weight_clip).astype(np.float32)
        return dw, db

    def train_online(
        self,
        m: np.ndarray,
        y: np.ndarray,
        cue: np.ndarray | None = None,
        delta_q: np.ndarray | None = None,
        passes: int | None = None,
    ) -> None:
        if not self.calibrated:
            self.observe(m, delta_q=delta_q)
        psi = self.build_features(m, cue=cue, delta_q=delta_q)
        n_passes = self.cfg.train_passes if passes is None else passes
        for _ in range(n_passes):
            for i in range(psi.shape[0]):
                self.step_update(psi[i], int(y[i]))


def evaluate_mode_b_gate_purity(head: CKCERHead, delta_q: np.ndarray, true_cue: np.ndarray) -> float:
    """External diagnostic measuring unsupervised Oja gate clustering purity against true query identity."""
    g = head.compute_gate(delta_q=delta_q)
    pred_q = g[:, 1].astype(np.int64)
    true_q = np.argmax(true_cue, axis=1).astype(np.int64)
    acc = float(np.mean(pred_q == true_q))
    return float(max(acc, 1.0 - acc))


# ---------------------------------------------------------------------------
# Stage 2 Mandatory Baselines & Controls
# ---------------------------------------------------------------------------


class RawServingReadoutBaseline:
    """Single-layer online readout matching SpikingCore.learn (lr=0.025, clip=0.025, un-normalized)."""

    def __init__(self, dim: int = 194, lr: float = 0.025, clip_step: float = 0.025) -> None:
        self.dim = dim
        self.lr = lr
        self.clip_step = clip_step
        self.w = np.zeros((2, dim), dtype=np.float32)
        self.b = np.zeros((2,), dtype=np.float32)

    @property
    def learnable_parameter_count(self) -> int:
        return int(self.w.size + self.b.size)

    def _feat(self, post: np.ndarray, cue: np.ndarray) -> np.ndarray:
        return np.concatenate([post, cue], axis=1).astype(np.float32)

    def train_online(self, post: np.ndarray, y: np.ndarray, cue: np.ndarray) -> None:
        f = self._feat(post, cue)
        for i in range(f.shape[0]):
            sc = self.w @ f[i] + self.b
            pred = int(np.argmax(sc))
            t = np.zeros((2,), dtype=np.float32)
            t[int(y[i])] = 1.0
            p = np.zeros((2,), dtype=np.float32)
            p[pred] = 1.0
            err = t - p
            self.w = np.clip(
                self.w + np.clip(self.lr * np.outer(err, f[i]), -self.clip_step, self.clip_step), -3.0, 3.0
            ).astype(np.float32)
            self.b = np.clip(self.b + np.clip(self.lr * err, -self.clip_step, self.clip_step), -3.0, 3.0).astype(
                np.float32
            )

    def scores(self, post: np.ndarray, cue: np.ndarray) -> np.ndarray:
        f = self._feat(post, cue)
        return (f @ self.w.T + self.b[None, :]).astype(np.float32)

    def predict(self, post: np.ndarray, cue: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(post, cue), axis=1).astype(np.int64)


class MatchedCapacityUngatedControl:
    """Un-gated linear head with 194 features (390 learnable params, matching CKCER-ModeA exactly).

    Uses [hat_m (96), R @ hat_m (96), cue (2)] without multiplicative query gating, trained with
    the exact same local perceptron rule (lr=0.08, dead_std=1e-4).
    """

    def __init__(self, dim: int = 96, seed: int = 0, cfg: CKCERConfig | None = None) -> None:
        self.dim = dim
        self.cfg = cfg or CKCERConfig()
        rng = np.random.default_rng(seed + 809)
        self.proj = (rng.standard_normal((dim, dim)) / np.sqrt(dim)).astype(np.float32)
        self.mean_m = np.zeros((1, dim), dtype=np.float32)
        self.std_m = np.ones((1, dim), dtype=np.float32)
        self.active_m = np.ones((1, dim), dtype=bool)
        self.calibrated = False
        self.feat_dim = 2 * dim + 2  # 194 features -> 390 learnable params
        self.w = np.zeros((2, self.feat_dim), dtype=np.float32)
        self.b = np.zeros((2,), dtype=np.float32)

    @property
    def learnable_parameter_count(self) -> int:
        return int(self.w.size + self.b.size)

    def observe(self, m: np.ndarray) -> None:
        m_cal = m[: self.cfg.obs_steps]
        self.mean_m = m_cal.mean(axis=0, keepdims=True).astype(np.float32)
        raw_std = m_cal.std(axis=0, keepdims=True).astype(np.float32)
        self.active_m = raw_std > self.cfg.dead_std
        self.std_m = np.where(self.active_m, raw_std, 1.0).astype(np.float32)
        self.calibrated = True

    def build_features(self, m: np.ndarray, cue: np.ndarray) -> np.ndarray:
        if not self.calibrated:
            self.observe(m)
        m_hat = np.zeros_like(m, dtype=np.float32)
        mask = self.active_m[0]
        if np.any(mask):
            m_hat[:, mask] = (m[:, mask] - self.mean_m[:, mask]) / self.std_m[:, mask]
        rp = (m_hat @ self.proj).astype(np.float32)
        return np.concatenate([m_hat, rp, cue.astype(np.float32)], axis=1).astype(np.float32)

    def train_online(self, m: np.ndarray, y: np.ndarray, cue: np.ndarray) -> None:
        if not self.calibrated:
            self.observe(m)
        psi = self.build_features(m, cue)
        for i in range(psi.shape[0]):
            sc = self.w @ psi[i] + self.b
            pred = int(np.argmax(sc))
            t = np.zeros((2,), dtype=np.float32)
            t[int(y[i])] = 1.0
            p = np.zeros((2,), dtype=np.float32)
            p[pred] = 1.0
            err = t - p
            dw = np.clip(self.cfg.lr * np.outer(err, psi[i]), -self.cfg.clip_step, self.cfg.clip_step)
            db = np.clip(self.cfg.lr * err, -self.cfg.clip_step, self.cfg.clip_step)
            self.w = np.clip(self.w + dw, -self.cfg.weight_clip, self.cfg.weight_clip).astype(np.float32)
            self.b = np.clip(self.b + db, -self.cfg.weight_clip, self.cfg.weight_clip).astype(np.float32)

    def scores(self, m: np.ndarray, cue: np.ndarray) -> np.ndarray:
        psi = self.build_features(m, cue)
        return (psi @ self.w.T + self.b[None, :]).astype(np.float32)

    def predict(self, m: np.ndarray, cue: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(m, cue), axis=1).astype(np.int64)


class OfflineRidgeDiagnostic:
    """DIAGNOSTIC-ONLY closed-form linear Ridge upper bound on CKCER features."""

    DIAGNOSTIC_ONLY = True

    def __init__(self, dim: int = 96, ridge: float = 1e-2, cfg: CKCERConfig | None = None) -> None:
        self.base = CKCERHead(dim=dim, mode="A", cfg=cfg)
        self.ridge = ridge
        self.w_ridge = np.zeros((195, 2), dtype=np.float64)

    def fit(self, m: np.ndarray, y: np.ndarray, cue: np.ndarray) -> None:
        self.base.observe(m)
        psi = self.base.build_features(m, cue=cue).astype(np.float64)
        design = np.concatenate([psi, np.ones((psi.shape[0], 1), dtype=np.float64)], axis=1)
        targets = np.stack([1.0 - y.astype(np.float64), y.astype(np.float64)], axis=1)
        reg = self.ridge * np.eye(design.shape[1], dtype=np.float64)
        self.w_ridge = np.linalg.solve(design.T @ design + reg, design.T @ targets)

    def scores(self, m: np.ndarray, cue: np.ndarray) -> np.ndarray:
        psi = self.base.build_features(m, cue=cue).astype(np.float64)
        design = np.concatenate([psi, np.ones((psi.shape[0], 1), dtype=np.float64)], axis=1)
        return (design @ self.w_ridge).astype(np.float32)

    def predict(self, m: np.ndarray, cue: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(m, cue), axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Hardest-Subset & Counterfactual Evaluation (Section IX.A)
# ---------------------------------------------------------------------------


def evaluate_counterfactual_detailed(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    v_first: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute full Section IX.A hardest-subset breakdown for counterfactual paired queries."""
    pred_a = np.argmax(scores_a, axis=1).astype(np.int64)
    pred_b = np.argmax(scores_b, axis=1).astype(np.int64)

    ok_a = pred_a == y_a
    ok_b = pred_b == y_b
    both = ok_a & ok_b

    diff_mask = y_a != y_b
    same_mask = y_a == y_b

    raw_margin_a = scores_a[:, 1] - scores_a[:, 0]
    raw_margin_b = scores_b[:, 1] - scores_b[:, 0]
    true_margin_a = (2.0 * y_a.astype(np.float32) - 1.0) * raw_margin_a
    true_margin_b = (2.0 * y_b.astype(np.float32) - 1.0) * raw_margin_b

    fk_ref = v_first if v_first is not None else y_a
    wrong_fk_b_diff = float(np.mean(pred_b[diff_mask] == fk_ref[diff_mask])) if np.any(diff_mask) else 0.0

    return {
        "pair_direct": float(0.5 * (np.mean(ok_a) + np.mean(ok_b))),
        "both_all": float(np.mean(both)),
        "both_A": float(np.mean(both[diff_mask])) if np.any(diff_mask) else 0.0,
        "both_B": float(np.mean(both[same_mask])) if np.any(same_mask) else 0.0,
        "acc_qA_diff": float(np.mean(ok_a[diff_mask])) if np.any(diff_mask) else 0.0,
        "acc_qB_diff": float(np.mean(ok_b[diff_mask])) if np.any(diff_mask) else 0.0,
        "acc_qA_same": float(np.mean(ok_a[same_mask])) if np.any(same_mask) else 0.0,
        "acc_qB_same": float(np.mean(ok_b[same_mask])) if np.any(same_mask) else 0.0,
        "margin_qA_diff": float(np.mean(true_margin_a[diff_mask])) if np.any(diff_mask) else 0.0,
        "margin_qB_diff": float(np.mean(true_margin_b[diff_mask])) if np.any(diff_mask) else 0.0,
        "pos_margin_both_A": float(
            np.mean((true_margin_a[diff_mask] > 0.0) & (true_margin_b[diff_mask] > 0.0))
        )
        if np.any(diff_mask)
        else 0.0,
        "flip_rate_A": float(np.mean(pred_a[diff_mask] != pred_b[diff_mask])) if np.any(diff_mask) else 0.0,
        "wrong_first_key_qB_diff": wrong_fk_b_diff,
        "n_case_A": int(np.sum(diff_mask)),
        "n_case_B": int(np.sum(same_mask)),
    }


def r5_scores_paired(head: P.Head, f_pt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (scores_a, scores_b) from an R5 P.Head on a 2*N_pairs paired feature array."""
    basis = np.stack([head.basis(row) for row in f_pt])
    logits = (basis @ head.w.T + head.b[None, :]).astype(np.float32)
    return logits[0::2], logits[1::2]


# ---------------------------------------------------------------------------
# Same-Sample Distractors & Auxiliary Generalization Tasks
# ---------------------------------------------------------------------------


def paired_with_same_sample_distractors(ptx: np.ndarray, seed: int, n: int = 2) -> np.ndarray:
    """Apply identical pre-query distractor pulses to both counterfactual siblings (2*p, 2*p+1)."""
    rng = np.random.default_rng(seed)
    out = np.asarray(ptx, np.float32).copy()
    used = set(G.WRITE_TICKS) | {G.QUERY_TICK}
    free = [t for t in range(G.N_TICKS) if t not in used]
    n_pairs = len(out) // 2
    for p in range(n_pairs):
        ticks = rng.choice(free, min(n, len(free)), replace=False)
        for t in ticks:
            ch = int(rng.integers(6, 8))
            out[2 * p, int(t), ch] += G.PULSE
            out[2 * p + 1, int(t), ch] += G.PULSE
    return out


def move_query(x: np.ndarray, delta: int = 1) -> np.ndarray:
    """Move query pulse later by `delta` ticks while keeping writes fixed (exact R5B Stage 4 transform)."""
    out = np.asarray(x, np.float32).copy()
    qmask = out[:, :, 4:6] > 0.5
    moved = np.zeros_like(out)
    for t in range(out.shape[1]):
        nt = t + delta if np.any(qmask[:, t]) else t
        if nt < out.shape[1]:
            moved[:, nt] += out[:, t]
    return moved


def irrelevant_third_write_windowed_set_v1(
    count: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Auxiliary Task Part 1: Variable write windows ({1,2,3}, {5,6,7}) + structured 3rd write at tick 8.

    At tick 8, a full-amplitude (PULSE=1.5) structured third write occurs on channels (6, 7)
    encoding an independent random value v_C in {0, 1}. The query at tick 10 asks for Key A or Key B.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.08, (count, G.N_TICKS, G.N_CHANNELS)).astype(np.float32)
    vA = rng.integers(0, 2, count).astype(np.int32)
    vB = rng.integers(0, 2, count).astype(np.int32)
    vC = rng.integers(0, 2, count).astype(np.int32)
    q = rng.integers(0, 2, count).astype(np.int32)
    t1_arr = np.zeros(count, dtype=np.int32)
    t2_arr = np.zeros(count, dtype=np.int32)

    for i in range(count):
        t1 = int(rng.integers(1, 4))
        t2 = int(rng.integers(5, 8))
        t1_arr[i] = t1
        t2_arr[i] = t2
        vals = {0: int(vA[i]), 1: int(vB[i])}
        for slot, key in zip((t1, t2), rng.permutation(2)):
            k = int(key)
            x[i, slot, k] += G.PULSE
            x[i, slot, 2 + vals[k]] += G.PULSE
        # Structured 3rd write at tick 8 on channels (6, 7)
        x[i, 8, 6 + int(vC[i])] += G.PULSE
        x[i, G.QUERY_TICK, 4 + int(q[i])] += G.PULSE

    y = np.where(q == 0, vA, vB).astype(np.int32)
    return x, y, {"va": vA, "vb": vB, "vc": vC, "q": q, "t1": t1_arr, "t2": t2_arr, "v_first": vA}


def irrelevant_third_write_windowed_paired_set_v1(
    n_pairs: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counterfactual paired set for Auxiliary Task Part 1 (returns ptx, pty, flips)."""
    rng = np.random.default_rng(seed)
    ptx = np.zeros((2 * n_pairs, G.N_TICKS, G.N_CHANNELS), np.float32)
    pty = np.zeros(2 * n_pairs, np.int32)
    flips = np.zeros(n_pairs, bool)

    for p in range(n_pairs):
        a = int(rng.integers(2))
        b = int(rng.integers(2))
        c = int(rng.integers(2))
        t1 = int(rng.integers(1, 4))
        t2 = int(rng.integers(5, 8))
        bg = rng.normal(0.0, 0.08, (G.N_TICKS, G.N_CHANNELS)).astype(np.float32)
        order = rng.permutation(2)
        flips[p] = a != b
        for slot, query in ((0, 0), (1, 1)):
            idx = 2 * p + slot
            ptx[idx] = bg
            vals = {0: a, 1: b}
            for place, key in zip((t1, t2), order):
                k = int(key)
                ptx[idx, place, k] += G.PULSE
                ptx[idx, place, 2 + vals[k]] += G.PULSE
            ptx[idx, 8, 6 + c] += G.PULSE
            ptx[idx, G.QUERY_TICK, 4 + query] += G.PULSE
            pty[idx] = vals[query]

    return ptx, pty, flips


def key_overwrite_query_set_v1(
    count: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Auxiliary Task Part 2: 3-write key overwrite task.

    - Tick 2: Key A written with v_A0 in {0, 1}
    - Tick 5: Key B written with v_B0 in {0, 1}
    - Tick 8: Overwrite key k_ov in {0 (Key A), 1 (Key B)} with v_ov in {0, 1}
    - Tick 10: Query q in {0, 1}; target is the latest value of the queried key.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.08, (count, G.N_TICKS, G.N_CHANNELS)).astype(np.float32)
    y = np.zeros(count, dtype=np.int32)
    va_final = np.zeros(count, dtype=np.int32)
    vb_final = np.zeros(count, dtype=np.int32)
    q_arr = np.zeros(count, dtype=np.int32)
    v_first = np.zeros(count, dtype=np.int32)

    for i in range(count):
        va0 = int(rng.integers(2))
        vb0 = int(rng.integers(2))
        k_ov = int(rng.integers(2))
        v_ov = int(rng.integers(2))
        q = int(rng.integers(2))

        x[i, 2, 0] += G.PULSE
        x[i, 2, 2 + va0] += G.PULSE
        x[i, 5, 1] += G.PULSE
        x[i, 5, 2 + vb0] += G.PULSE
        x[i, 8, k_ov] += G.PULSE
        x[i, 8, 2 + v_ov] += G.PULSE
        x[i, G.QUERY_TICK, 4 + q] += G.PULSE

        va = v_ov if k_ov == 0 else va0
        vb = v_ov if k_ov == 1 else vb0
        va_final[i] = va
        vb_final[i] = vb
        q_arr[i] = q
        v_first[i] = va0
        y[i] = va if q == 0 else vb

    return x, y, {"va": va_final, "vb": vb_final, "q": q_arr, "v_first": v_first}


def key_overwrite_paired_set_v1(
    n_pairs: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counterfactual paired set for Auxiliary Task Part 2 (key overwrite)."""
    rng = np.random.default_rng(seed)
    ptx = np.zeros((2 * n_pairs, G.N_TICKS, G.N_CHANNELS), np.float32)
    pty = np.zeros(2 * n_pairs, np.int32)
    flips = np.zeros(n_pairs, bool)

    for p in range(n_pairs):
        va0 = int(rng.integers(2))
        vb0 = int(rng.integers(2))
        k_ov = int(rng.integers(2))
        v_ov = int(rng.integers(2))
        bg = rng.normal(0.0, 0.08, (G.N_TICKS, G.N_CHANNELS)).astype(np.float32)
        va = v_ov if k_ov == 0 else va0
        vb = v_ov if k_ov == 1 else vb0
        flips[p] = va != vb

        for slot, query in ((0, 0), (1, 1)):
            idx = 2 * p + slot
            ptx[idx] = bg
            ptx[idx, 2, 0] += G.PULSE
            ptx[idx, 2, 2 + va0] += G.PULSE
            ptx[idx, 5, 1] += G.PULSE
            ptx[idx, 5, 2 + vb0] += G.PULSE
            ptx[idx, 8, k_ov] += G.PULSE
            ptx[idx, 8, 2 + v_ov] += G.PULSE
            ptx[idx, G.QUERY_TICK, 4 + query] += G.PULSE
            pty[idx] = va if query == 0 else vb

    return ptx, pty, flips


# ---------------------------------------------------------------------------
# Protected Benchmark Tasks (from R2A / R4B / R5A)
# ---------------------------------------------------------------------------


def sim_xor_set_v1(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    labels = rng.integers(0, 2, count).astype(np.int32)
    for index in range(count):
        first = int(rng.integers(2))
        second = first ^ int(labels[index])
        tick = int(rng.integers(1, 4))
        values[index, tick, 0] += G.PULSE * first
        values[index, tick, 1] += G.PULSE * second
        values[index, 10, 7] += G.PULSE
    return values, labels


def separated_xor_set_v1(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    from scarx.evaluation.tasks21 import representation
    return representation("xor", seed, count)


def order_task_set_v1(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    from scarx.evaluation.tasks21 import representation
    return representation("order", seed, count)


def run_protected_benchmarks(seed: int) -> dict[str, dict[str, float]]:
    """Run protected benchmarks (sim_xor, xor_separated, order_task) for R5 Head vs CKCER."""
    out: dict[str, dict[str, float]] = {"r5": {}, "ckcer": {}}
    # 1. simultaneous_xor
    x_tr, y_tr = sim_xor_set_v1(seed + 101, 1024)
    x_te, y_te = sim_xor_set_v1(seed + 151, 512)
    c_tr = capture_r6(seed, x_tr, query_tick=10)
    c_te = capture_r6(seed, x_te, query_tick=10)

    h_r5 = P.train_head(c_tr["m_pre"], y_tr, seed=seed)
    out["r5"]["sim_xor"] = h_r5.acc(c_te["m_pre"], y_te)

    h_ck = CKCERHead(dim=96, mode="ungated", seed=seed)
    h_ck.train_online(c_tr["m_pre"], y_tr)
    out["ckcer"]["sim_xor"] = float(np.mean(h_ck.predict(c_te["m_pre"]) == y_te))

    # 2. separated_xor (canonical tasks21 temporal XOR)
    x_tr, y_tr = separated_xor_set_v1(seed + 201, 1024)
    x_te, y_te = separated_xor_set_v1(seed + 251, 512)
    c_tr = capture_r6(seed, x_tr, query_tick=11)
    c_te = capture_r6(seed, x_te, query_tick=11)

    h_r5 = P.train_head(c_tr["m_pre"], y_tr, seed=seed)
    out["r5"]["xor_separated"] = h_r5.acc(c_te["m_pre"], y_te)

    h_ck = CKCERHead(dim=96, mode="ungated", seed=seed)
    h_ck.train_online(c_tr["m_pre"], y_tr)
    out["ckcer"]["xor_separated"] = float(np.mean(h_ck.predict(c_te["m_pre"]) == y_te))

    # 3. temporal_order (canonical tasks21 temporal order)
    x_tr, y_tr = order_task_set_v1(seed + 301, 1024)
    x_te, y_te = order_task_set_v1(seed + 351, 512)
    c_tr = capture_r6(seed, x_tr, query_tick=11)
    c_te = capture_r6(seed, x_te, query_tick=11)

    h_r5 = P.train_head(c_tr["m_pre"], y_tr, seed=seed)
    out["r5"]["order_task"] = h_r5.acc(c_te["m_pre"], y_te)

    h_ck = CKCERHead(dim=96, mode="ungated", seed=seed)
    h_ck.train_online(c_tr["m_pre"], y_tr)
    out["ckcer"]["order_task"] = float(np.mean(h_ck.predict(c_te["m_pre"]) == y_te))

    return out


# ---------------------------------------------------------------------------
# Single-Factor Ablation Ladder (R5 -> CKCER)
# ---------------------------------------------------------------------------


class R5AblationHead:
    """Configurable head isolating each architectural step from R5 to CKCER-ModeA."""

    def __init__(
        self,
        dim: int,
        seed: int,
        pre_gating_norm: bool = False,
        dead_std: float = 1e-4,
        lr: float = 0.08,
        proj: int = 32,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self.pre_gating_norm = pre_gating_norm
        self.dead_std = dead_std
        self.lr = lr
        rng = np.random.default_rng(1234)
        self.p = rng.normal(0, 1.0 / np.sqrt(dim), (dim, proj)).astype(np.float32)
        self.fdim = dim + 2 + 2 * proj
        self.mean_s = np.zeros((1, dim), dtype=np.float32)
        self.std_s = np.ones((1, dim), dtype=np.float32)
        self.active_s = np.ones((1, dim), dtype=bool)
        self.r5_head: P.Head | None = None
        self.w = np.zeros((2, self.fdim), dtype=np.float32)
        self.b = np.zeros((2,), dtype=np.float32)

    def train_online(self, s: np.ndarray, y: np.ndarray, cue: np.ndarray) -> None:
        if not self.pre_gating_norm:
            f = P.bilinear_query_map(s, cue, n_proj=32, seed=1234)
            self.r5_head = P.train_head(f, y, seed=self.seed)
            return
        s_cal = s[:1024]
        self.mean_s = s_cal.mean(axis=0, keepdims=True).astype(np.float32)
        raw_std = s_cal.std(axis=0, keepdims=True).astype(np.float32)
        self.active_s = raw_std > self.dead_std
        self.std_s = np.where(self.active_s, raw_std, 1.0).astype(np.float32)
        f = self._build_pre_norm(s, cue)
        for i in range(f.shape[0]):
            sc = self.w @ f[i] + self.b
            pred = int(np.argmax(sc))
            t = np.zeros((2,), dtype=np.float32)
            t[int(y[i])] = 1.0
            p = np.zeros((2,), dtype=np.float32)
            p[pred] = 1.0
            err = t - p
            self.w = np.clip(self.w + np.clip(self.lr * np.outer(err, f[i]), -0.025, 0.025), -3.0, 3.0).astype(
                np.float32
            )
            self.b = np.clip(self.b + np.clip(self.lr * err, -0.025, 0.025), -3.0, 3.0).astype(np.float32)

    def _build_pre_norm(self, s: np.ndarray, cue: np.ndarray) -> np.ndarray:
        s_hat = np.zeros_like(s, dtype=np.float32)
        mask = self.active_s[0]
        if np.any(mask):
            s_hat[:, mask] = (s[:, mask] - self.mean_s[:, mask]) / self.std_s[:, mask]
        z = s_hat @ self.p
        return np.concatenate([s_hat, cue, z * cue[:, :1], z * cue[:, 1:]], axis=1).astype(np.float32)

    def scores(self, s: np.ndarray, cue: np.ndarray) -> np.ndarray:
        if not self.pre_gating_norm:
            assert self.r5_head is not None
            f = P.bilinear_query_map(s, cue, n_proj=32, seed=1234)
            basis = np.stack([self.r5_head.basis(row) for row in f])
            return (basis @ self.r5_head.w.T + self.r5_head.b[None, :]).astype(np.float32)
        f = self._build_pre_norm(s, cue)
        return (f @ self.w.T + self.b[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Full Per-Seed Stage 2 + Stage 3 Evaluation Suite
# ---------------------------------------------------------------------------


def evaluate_seed_full(seed: int) -> dict[str, Any]:
    """Run all Stage 2 baselines, ablations, and Stage 3 adversarial benchmarks for one seed."""
    # 1. Clean train / test / paired datasets (offsets locked in protocol-v1.md, matching R5B Stage 4)
    x_tr, y_tr, vA_tr, vB_tr, q_tr, _ = G.independent_query_set_v1(seed + 200_000, 1024)
    x_te, y_te, vA_te, vB_te, q_te, _ = G.independent_query_set_v1(seed + 300_000, 512)
    ptx_clean, pty_clean, pvA_clean, pvB_clean, pq_clean, _, flips_clean = G.independent_paired_set_v1(
        seed + 400_000, 256
    )
    ya_clean, yb_clean = pty_clean[0::2], pty_clean[1::2]
    fk_pa_clean = G.baseline_first_key(ptx_clean[0::2])

    cap_tr = capture_r6(seed, x_tr, query_tick=10)
    cap_te = capture_r6(seed, x_te, query_tick=10)
    cap_pt = capture_r6(seed, ptx_clean, query_tick=10)

    # --- Train CKCER Mode A & Mode B on clean ---
    ck_a = CKCERHead(dim=96, mode="A", seed=seed)
    ck_a.train_online(cap_tr["m_pre"], y_tr, cue=cap_tr["cue"])

    ck_b = CKCERHead(dim=96, mode="B", seed=seed)
    ck_b.train_online(cap_tr["m_pre"], y_tr, delta_q=cap_tr["delta_q"])

    # --- Train Original R5 Head on clean ---
    f_tr_r5 = P.bilinear_query_map(cap_tr["post"], cap_tr["cue"])
    f_te_r5 = P.bilinear_query_map(cap_te["post"], cap_te["cue"])
    f_pt_r5 = P.bilinear_query_map(cap_pt["post"], cap_pt["cue"])
    r5_clean = P.train_head(f_tr_r5, y_tr, seed=seed)

    # --- Stage 2 Baselines (all 10) on Clean ---
    baselines: dict[str, dict[str, Any]] = {}

    # B1. first_key_shortcut
    fk_te_pred = G.baseline_first_key(x_te)
    fk_direct = float(np.mean(fk_te_pred == y_te))
    fk_scores_a = np.stack([1.0 - fk_pa_clean, fk_pa_clean], axis=1).astype(np.float32)
    fk_cf = evaluate_counterfactual_detailed(fk_scores_a, fk_scores_a, ya_clean, yb_clean, v_first=fk_pa_clean)
    baselines["first_key_shortcut"] = {"direct": fk_direct, **fk_cf}

    # B2. original_r5_head
    r5_direct = r5_clean.acc(f_te_r5, y_te)
    r5_sa, r5_sb = r5_scores_paired(r5_clean, f_pt_r5)
    r5_cf = evaluate_counterfactual_detailed(r5_sa, r5_sb, ya_clean, yb_clean, v_first=fk_pa_clean)
    baselines["original_r5_head"] = {"direct": r5_direct, **r5_cf}

    # B3. raw_serving_readout
    raw_srv = RawServingReadoutBaseline(dim=194, lr=0.025, clip_step=0.025)
    raw_srv.train_online(cap_tr["post"], y_tr, cap_tr["cue"])
    raw_direct = float(np.mean(raw_srv.predict(cap_te["post"], cap_te["cue"]) == y_te))
    raw_sc = raw_srv.scores(cap_pt["post"], cap_pt["cue"])
    raw_cf = evaluate_counterfactual_detailed(raw_sc[0::2], raw_sc[1::2], ya_clean, yb_clean, v_first=fk_pa_clean)
    baselines["raw_serving_readout"] = {"direct": raw_direct, **raw_cf}

    # B4. offline_ridge_diagnostic (DIAGNOSTIC ONLY)
    ridge = OfflineRidgeDiagnostic(dim=96, ridge=1e-2)
    ridge.fit(cap_tr["m_pre"], y_tr, cap_tr["cue"])
    ridge_direct = float(np.mean(ridge.predict(cap_te["m_pre"], cap_te["cue"]) == y_te))
    ridge_sc = ridge.scores(cap_pt["m_pre"], cap_pt["cue"])
    ridge_cf = evaluate_counterfactual_detailed(
        ridge_sc[0::2], ridge_sc[1::2], ya_clean, yb_clean, v_first=fk_pa_clean
    )
    baselines["offline_ridge_diagnostic"] = {"direct": ridge_direct, "diagnostic_only": True, **ridge_cf}

    # B5. matched_capacity_control (un-gated 194 features, 390 params)
    mc_ctrl = MatchedCapacityUngatedControl(dim=96, seed=seed)
    mc_ctrl.train_online(cap_tr["m_pre"], y_tr, cap_tr["cue"])
    mc_direct = float(np.mean(mc_ctrl.predict(cap_te["m_pre"], cap_te["cue"]) == y_te))
    mc_sc = mc_ctrl.scores(cap_pt["m_pre"], cap_pt["cue"])
    mc_cf = evaluate_counterfactual_detailed(mc_sc[0::2], mc_sc[1::2], ya_clean, yb_clean, v_first=fk_pa_clean)
    baselines["matched_capacity_control"] = {"direct": mc_direct, **mc_cf}

    # B6. query_only_control
    qo_ctrl = CKCERHead(dim=96, mode="A", seed=seed)
    qo_ctrl.train_online(np.zeros_like(cap_tr["m_pre"]), y_tr, cue=cap_tr["cue"])
    qo_direct = float(np.mean(qo_ctrl.predict(np.zeros_like(cap_te["m_pre"]), cue=cap_te["cue"]) == y_te))
    qo_sc = qo_ctrl.scores(np.zeros_like(cap_pt["m_pre"]), cue=cap_pt["cue"])
    qo_cf = evaluate_counterfactual_detailed(qo_sc[0::2], qo_sc[1::2], ya_clean, yb_clean, v_first=fk_pa_clean)
    baselines["query_only_control"] = {"direct": qo_direct, **qo_cf}

    # B7. query_free_state_control
    qf_cue_tr = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (cap_tr["m_pre"].shape[0], 1))
    qf_cue_te = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (cap_te["m_pre"].shape[0], 1))
    qf_cue_pt = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (cap_pt["m_pre"].shape[0], 1))
    qf_ctrl = CKCERHead(dim=96, mode="A", seed=seed)
    qf_ctrl.train_online(cap_tr["m_pre"], y_tr, cue=qf_cue_tr)
    qf_direct = float(np.mean(qf_ctrl.predict(cap_te["m_pre"], cue=qf_cue_te) == y_te))
    qf_sc = qf_ctrl.scores(cap_pt["m_pre"], cue=qf_cue_pt)
    qf_cf = evaluate_counterfactual_detailed(qf_sc[0::2], qf_sc[1::2], ya_clean, yb_clean, v_first=fk_pa_clean)
    baselines["query_free_state_control"] = {"direct": qf_direct, **qf_cf}

    # B8. random_label_control
    rng_rl = np.random.default_rng(seed + 910)
    y_perm = rng_rl.permutation(y_tr)
    rl_ctrl = CKCERHead(dim=96, mode="A", seed=seed)
    rl_ctrl.train_online(cap_tr["m_pre"], y_perm, cue=cap_tr["cue"])
    rl_direct = float(np.mean(rl_ctrl.predict(cap_te["m_pre"], cue=cap_te["cue"]) == y_te))
    rl_sc = rl_ctrl.scores(cap_pt["m_pre"], cue=cap_pt["cue"])
    rl_cf = evaluate_counterfactual_detailed(rl_sc[0::2], rl_sc[1::2], ya_clean, yb_clean, v_first=fk_pa_clean)
    baselines["random_label_control"] = {"direct": rl_direct, **rl_cf}

    # B9. fixed_nonlearning_candidate
    fn_ctrl = CKCERHead(dim=96, mode="A", seed=seed)
    fn_ctrl.observe(cap_tr["m_pre"])
    fn_direct = float(np.mean(fn_ctrl.predict(cap_te["m_pre"], cue=cap_te["cue"]) == y_te))
    fn_sc = fn_ctrl.scores(cap_pt["m_pre"], cue=cap_pt["cue"])
    fn_cf = evaluate_counterfactual_detailed(fn_sc[0::2], fn_sc[1::2], ya_clean, yb_clean, v_first=fk_pa_clean)
    baselines["fixed_nonlearning_candidate"] = {"direct": fn_direct, **fn_cf}

    # B10. cue_shuffling_intervention (train-time and test-time)
    rng_cs = np.random.default_rng(seed + 920)
    cue_tr_shuf = cap_tr["cue"][rng_cs.permutation(cap_tr["cue"].shape[0])]
    cst_ctrl = CKCERHead(dim=96, mode="A", seed=seed)
    cst_ctrl.train_online(cap_tr["m_pre"], y_tr, cue=cue_tr_shuf)
    cst_direct = float(np.mean(cst_ctrl.predict(cap_te["m_pre"], cue=cap_te["cue"]) == y_te))
    cst_sc = cst_ctrl.scores(cap_pt["m_pre"], cue=cap_pt["cue"])
    cst_cf = evaluate_counterfactual_detailed(cst_sc[0::2], cst_sc[1::2], ya_clean, yb_clean, v_first=fk_pa_clean)
    baselines["cue_shuf_train"] = {"direct": cst_direct, **cst_cf}

    cue_te_shuf = cap_te["cue"][rng_cs.permutation(cap_te["cue"].shape[0])]
    cue_pt_shuf = cap_pt["cue"][rng_cs.permutation(cap_pt["cue"].shape[0])]
    cse_direct = float(np.mean(ck_a.predict(cap_te["m_pre"], cue=cue_te_shuf) == y_te))
    cse_sc = ck_a.scores(cap_pt["m_pre"], cue=cue_pt_shuf)
    cse_cf = evaluate_counterfactual_detailed(cse_sc[0::2], cse_sc[1::2], ya_clean, yb_clean, v_first=fk_pa_clean)
    baselines["cue_shuf_test"] = {"direct": cse_direct, **cse_cf}

    # --- Single-Factor Ablation Ladder ---
    ab1 = R5AblationHead(dim=96, seed=seed, pre_gating_norm=False)
    ab1.train_online(cap_tr["m_pre"], y_tr, cap_tr["cue"])
    ab1_sc_pt = ab1.scores(cap_pt["m_pre"], cap_pt["cue"])
    ab1_cf = evaluate_counterfactual_detailed(
        ab1_sc_pt[0::2], ab1_sc_pt[1::2], ya_clean, yb_clean, v_first=fk_pa_clean
    )
    ab1_dir = float(np.mean(np.argmax(ab1.scores(cap_te["m_pre"], cap_te["cue"]), axis=1) == y_te))

    ab2 = R5AblationHead(dim=96, seed=seed, pre_gating_norm=True, dead_std=1e-4, lr=0.08)
    ab2.train_online(cap_tr["m_pre"], y_tr, cap_tr["cue"])
    ab2_sc_pt = ab2.scores(cap_pt["m_pre"], cap_pt["cue"])
    ab2_cf = evaluate_counterfactual_detailed(
        ab2_sc_pt[0::2], ab2_sc_pt[1::2], ya_clean, yb_clean, v_first=fk_pa_clean
    )
    ab2_dir = float(np.mean(np.argmax(ab2.scores(cap_te["m_pre"], cap_te["cue"]), axis=1) == y_te))

    ablations = {
        "step0_original_r5": baselines["original_r5_head"],
        "step1_pre_query_trace_only": {"direct": ab1_dir, **ab1_cf},
        "step2_pre_gating_norm_plus_trunk": {"direct": ab2_dir, **ab2_cf},
    }

    # --- Stage 3 Adversarial Conditions for R5, CKCER-ModeA, CKCER-ModeB ---
    conditions: dict[str, dict[str, dict[str, float]]] = {"r5": {}, "ckcer_mode_a": {}, "ckcer_mode_b": {}}

    def _eval_condition(
        cond_name: str,
        r5_model: P.Head,
        cka_model: CKCERHead,
        ckb_model: CKCERHead,
        c_te_d: dict[str, np.ndarray],
        y_te_d: np.ndarray,
        c_pt_d: dict[str, np.ndarray],
        pty_d: np.ndarray,
        fk_pa_d: np.ndarray,
    ) -> None:
        ya_d, yb_d = pty_d[0::2], pty_d[1::2]
        f_te_d = P.bilinear_query_map(c_te_d["post"], c_te_d["cue"])
        f_pt_d = P.bilinear_query_map(c_pt_d["post"], c_pt_d["cue"])
        r5_d = r5_model.acc(f_te_d, y_te_d)
        r5_sa_d, r5_sb_d = r5_scores_paired(r5_model, f_pt_d)
        r5_c = evaluate_counterfactual_detailed(r5_sa_d, r5_sb_d, ya_d, yb_d, v_first=fk_pa_d)
        conditions["r5"][cond_name] = {"direct": r5_d, **r5_c}

        cka_d = float(np.mean(cka_model.predict(c_te_d["m_pre"], cue=c_te_d["cue"]) == y_te_d))
        cka_sc = cka_model.scores(c_pt_d["m_pre"], cue=c_pt_d["cue"])
        cka_c = evaluate_counterfactual_detailed(cka_sc[0::2], cka_sc[1::2], ya_d, yb_d, v_first=fk_pa_d)
        conditions["ckcer_mode_a"][cond_name] = {"direct": cka_d, **cka_c}

        ckb_d = float(np.mean(ckb_model.predict(c_te_d["m_pre"], delta_q=c_te_d["delta_q"]) == y_te_d))
        ckb_sc = ckb_model.scores(c_pt_d["m_pre"], delta_q=c_pt_d["delta_q"])
        ckb_c = evaluate_counterfactual_detailed(ckb_sc[0::2], ckb_sc[1::2], ya_d, yb_d, v_first=fk_pa_d)
        ckb_c["oja_gate_purity"] = evaluate_mode_b_gate_purity(ckb_model, c_te_d["delta_q"], c_te_d["cue"])
        conditions["ckcer_mode_b"][cond_name] = {"direct": ckb_d, **ckb_c}

    def _fit_and_eval(
        cond_name: str,
        trx_d: np.ndarray,
        try_d: np.ndarray,
        tex_d: np.ndarray,
        tey_d: np.ndarray,
        ptx_d: np.ndarray,
        pty_d: np.ndarray,
    ) -> None:
        c_tr_d = capture_r6(seed, trx_d)
        c_te_d = capture_r6(seed, tex_d)
        c_pt_d = capture_r6(seed, ptx_d)
        fk_pa_d = G.baseline_first_key(ptx_d[0::2])
        r5_m = P.train_head(P.bilinear_query_map(c_tr_d["post"], c_tr_d["cue"]), try_d, seed=seed)
        cka_m = CKCERHead(dim=96, mode="A", seed=seed)
        cka_m.train_online(c_tr_d["m_pre"], try_d, cue=c_tr_d["cue"])
        ckb_m = CKCERHead(dim=96, mode="B", seed=seed)
        ckb_m.train_online(c_tr_d["m_pre"], try_d, delta_q=c_tr_d["delta_q"])
        _eval_condition(cond_name, r5_m, cka_m, ckb_m, c_te_d, tey_d, c_pt_d, pty_d, fk_pa_d)

    # Condition 1: clean
    _eval_condition("clean", r5_clean, ck_a, ck_b, cap_te, y_te, cap_pt, pty_clean, fk_pa_clean)
    ablations["step3_full_ckcer_mode_a"] = conditions["ckcer_mode_a"]["clean"]

    # Condition 2 & 3: train_shift_test_shift and ood_clean_to_shift (exact R5B Stage 4 transform)
    s_tr = G.shifted(x_tr, 1, seed=seed)
    s_te = G.shifted(x_te, 1, seed=seed)
    s_pt = G.shifted(ptx_clean, 1, seed=seed)
    cap_s_te = capture_r6(seed, s_te)
    cap_s_pt = capture_r6(seed, s_pt)
    fk_pa_shift = G.baseline_first_key(s_pt[0::2])

    # OOD clean -> shift uses clean-trained models without retraining
    _eval_condition("ood_clean_to_shift", r5_clean, ck_a, ck_b, cap_s_te, y_te, cap_s_pt, pty_clean, fk_pa_shift)
    # In-distribution shifted learning
    _fit_and_eval("train_shift_test_shift", s_tr, y_tr, s_te, y_te, s_pt, pty_clean)

    # Condition 4: reversed_writes
    _fit_and_eval(
        "reversed_writes",
        G.reversed_writes(x_tr),
        y_tr,
        G.reversed_writes(x_te),
        y_te,
        G.reversed_writes(ptx_clean),
        pty_clean,
    )

    # Condition 5 & 6: distractors (exact R5B Stage 4) + paired_same_sample_distractors
    d_tr = G.with_distractors(x_tr, seed + 11)
    d_te = G.with_distractors(x_te, seed + 12)
    d_pt = G.with_distractors(ptx_clean, seed + 13)
    d_pt_same = paired_with_same_sample_distractors(ptx_clean, seed + 13)

    c_tr_di = capture_r6(seed, d_tr)
    c_te_di = capture_r6(seed, d_te)
    c_pt_di = capture_r6(seed, d_pt)
    c_pt_sdi = capture_r6(seed, d_pt_same)
    r5_di = P.train_head(P.bilinear_query_map(c_tr_di["post"], c_tr_di["cue"]), y_tr, seed=seed)
    cka_di = CKCERHead(dim=96, mode="A", seed=seed)
    cka_di.train_online(c_tr_di["m_pre"], y_tr, cue=c_tr_di["cue"])
    ckb_di = CKCERHead(dim=96, mode="B", seed=seed)
    ckb_di.train_online(c_tr_di["m_pre"], y_tr, delta_q=c_tr_di["delta_q"])

    _eval_condition("distractors", r5_di, cka_di, ckb_di, c_te_di, y_te, c_pt_di, pty_clean, fk_pa_clean)
    _eval_condition(
        "paired_same_sample_distractors", r5_di, cka_di, ckb_di, c_te_di, y_te, c_pt_sdi, pty_clean, fk_pa_clean
    )

    # Condition 7: noise_change (sigma=0.16, exact R5B Stage 4)
    _fit_and_eval(
        "noise_change",
        G.noisier(x_tr, seed + 21, sigma=0.16),
        y_tr,
        G.noisier(x_te, seed + 22, sigma=0.16),
        y_te,
        G.noisier(ptx_clean, seed + 23, sigma=0.16),
        pty_clean,
    )

    # Condition 8: query_tick_variation (move_query delta=1, exact R5B Stage 4)
    _fit_and_eval(
        "query_tick_variation",
        move_query(x_tr, 1),
        y_tr,
        move_query(x_te, 1),
        y_te,
        move_query(ptx_clean, 1),
        pty_clean,
    )

    # Condition 9: Auxiliary Generalization Tasks
    # Part 1: Structured 3rd write at tick 8 + variable write windows
    x_tr_aux1, y_tr_aux1, _ = irrelevant_third_write_windowed_set_v1(1024, seed=seed + 500_000)
    x_te_aux1, y_te_aux1, _ = irrelevant_third_write_windowed_set_v1(512, seed=seed + 510_000)
    ptx_aux1, pty_aux1, _ = irrelevant_third_write_windowed_paired_set_v1(256, seed=seed + 520_000)
    _fit_and_eval("aux_structured_third_write_windowed", x_tr_aux1, y_tr_aux1, x_te_aux1, y_te_aux1, ptx_aux1, pty_aux1)

    # Part 2: 3-write key overwrite task
    x_tr_aux2, y_tr_aux2, m_tr_aux2 = key_overwrite_query_set_v1(1024, seed=seed + 530_000)
    x_te_aux2, y_te_aux2, m_te_aux2 = key_overwrite_query_set_v1(512, seed=seed + 540_000)
    ptx_aux2, pty_aux2, _ = key_overwrite_paired_set_v1(256, seed=seed + 550_000)
    c_tr_aux2 = capture_r6(seed, x_tr_aux2, query_tick=10)
    c_te_aux2 = capture_r6(seed, x_te_aux2, query_tick=10)
    c_pt_aux2 = capture_r6(seed, ptx_aux2, query_tick=10)
    r5_aux2 = P.train_head(P.bilinear_query_map(c_tr_aux2["post"], c_tr_aux2["cue"]), y_tr_aux2, seed=seed)
    cka_aux2 = CKCERHead(dim=96, mode="A", seed=seed)
    cka_aux2.train_online(c_tr_aux2["m_pre"], y_tr_aux2, cue=c_tr_aux2["cue"])
    ckb_aux2 = CKCERHead(dim=96, mode="B", seed=seed)
    ckb_aux2.train_online(c_tr_aux2["m_pre"], y_tr_aux2, delta_q=c_tr_aux2["delta_q"])
    _eval_condition(
        "aux_key_overwrite",
        r5_aux2,
        cka_aux2,
        ckb_aux2,
        c_te_aux2,
        y_te_aux2,
        c_pt_aux2,
        pty_aux2,
        pty_aux2[0::2],
    )

    # Protected tasks
    protected = run_protected_benchmarks(seed)

    return {
        "seed": seed,
        "baselines_clean": baselines,
        "ablations_clean": ablations,
        "conditions": conditions,
        "protected": protected,
        "parameter_audit": {
            "r5_learnable_params": int(r5_clean.w.size + r5_clean.b.size),
            "r5_random_proj_bytes": int(192 * 32 * 4),
            "ckcer_mode_a_learnable_params": ck_a.learnable_parameter_count,
            "ckcer_mode_a_random_proj_bytes": ck_a.random_projection_bytes,
            "ckcer_mode_b_learnable_params": ck_b.learnable_parameter_count,
            "ckcer_mode_b_random_proj_bytes": ck_b.random_projection_bytes,
        },
    }


def summarize_across_seeds(
    seed_results: list[dict[str, Any]],
    bootstrap_seed: int = 2101,
) -> dict[str, Any]:
    """Compute mean, SD, median, IQR, and 10,000-sample paired bootstrap CI95% across seeds."""
    rng = np.random.default_rng(bootstrap_seed)

    def _stats(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        return {
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(q50),
            "iqr": float(q75 - q25),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def _paired_ci(cand: list[float], ref: list[float]) -> dict[str, Any]:
        d = np.asarray(cand, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
        idx = rng.integers(0, d.size, size=(10000, d.size))
        means = d[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return {
            "mean_delta": float(np.mean(d)),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "wins": int(np.sum(d > 0.0)),
            "ties": int(np.sum(d == 0.0)),
            "losses": int(np.sum(d < 0.0)),
            "total": int(d.size),
        }

    base_keys = list(seed_results[0]["baselines_clean"].keys())
    base_summary: dict[str, Any] = {}
    for bk in base_keys:
        base_summary[bk] = {}
        for m in (
            "direct",
            "both_all",
            "both_A",
            "both_B",
            "acc_qA_diff",
            "acc_qB_diff",
            "flip_rate_A",
            "wrong_first_key_qB_diff",
        ):
            base_summary[bk][m] = _stats([sr["baselines_clean"][bk][m] for sr in seed_results])

    ab_keys = list(seed_results[0]["ablations_clean"].keys())
    ab_summary: dict[str, Any] = {}
    for ak in ab_keys:
        ab_summary[ak] = {}
        for m in (
            "direct",
            "both_all",
            "both_A",
            "both_B",
            "acc_qA_diff",
            "acc_qB_diff",
            "margin_qA_diff",
            "margin_qB_diff",
            "flip_rate_A",
            "wrong_first_key_qB_diff",
        ):
            ab_summary[ak][m] = _stats([sr["ablations_clean"][ak][m] for sr in seed_results])

    cond_keys = list(seed_results[0]["conditions"]["r5"].keys())
    cond_summary: dict[str, Any] = {
        "r5": {},
        "ckcer_mode_a": {},
        "ckcer_mode_b": {},
        "paired_deltas_mode_a_vs_r5": {},
        "paired_deltas_mode_b_vs_r5": {},
    }
    metric_list = (
        "direct",
        "both_all",
        "both_A",
        "both_B",
        "acc_qA_diff",
        "acc_qB_diff",
        "margin_qA_diff",
        "margin_qB_diff",
        "pos_margin_both_A",
        "flip_rate_A",
        "wrong_first_key_qB_diff",
    )
    for ck in cond_keys:
        for mk in ("r5", "ckcer_mode_a", "ckcer_mode_b"):
            cond_summary[mk][ck] = {}
            avail_metrics = [m for m in metric_list if m in seed_results[0]["conditions"][mk][ck]]
            if "oja_gate_purity" in seed_results[0]["conditions"][mk][ck]:
                avail_metrics.append("oja_gate_purity")
            for m in avail_metrics:
                cond_summary[mk][ck][m] = _stats([sr["conditions"][mk][ck][m] for sr in seed_results])
        cond_summary["paired_deltas_mode_a_vs_r5"][ck] = {}
        cond_summary["paired_deltas_mode_b_vs_r5"][ck] = {}
        for m in ("direct", "both_all", "both_A", "both_B", "flip_rate_A"):
            r5_vals = [sr["conditions"]["r5"][ck][m] for sr in seed_results]
            cka_vals = [sr["conditions"]["ckcer_mode_a"][ck][m] for sr in seed_results]
            ckb_vals = [sr["conditions"]["ckcer_mode_b"][ck][m] for sr in seed_results]
            cond_summary["paired_deltas_mode_a_vs_r5"][ck][m] = _paired_ci(cka_vals, r5_vals)
            cond_summary["paired_deltas_mode_b_vs_r5"][ck][m] = _paired_ci(ckb_vals, r5_vals)

    fk_dir = [sr["baselines_clean"]["first_key_shortcut"]["direct"] for sr in seed_results]
    fk_ba = [sr["baselines_clean"]["first_key_shortcut"]["both_A"] for sr in seed_results]
    cka_clean_dir = [sr["conditions"]["ckcer_mode_a"]["clean"]["direct"] for sr in seed_results]
    cka_clean_ba = [sr["conditions"]["ckcer_mode_a"]["clean"]["both_A"] for sr in seed_results]
    cond_summary["paired_deltas_mode_a_vs_first_key_clean"] = {
        "direct": _paired_ci(cka_clean_dir, fk_dir),
        "both_A": _paired_ci(cka_clean_ba, fk_ba),
    }

    prot_summary: dict[str, Any] = {"r5": {}, "ckcer": {}, "paired_deltas": {}}
    for pk in ("sim_xor", "xor_separated", "order_task"):
        r5_p = [sr["protected"]["r5"][pk] for sr in seed_results]
        ck_p = [sr["protected"]["ckcer"][pk] for sr in seed_results]
        prot_summary["r5"][pk] = _stats(r5_p)
        prot_summary["ckcer"][pk] = _stats(ck_p)
        prot_summary["paired_deltas"][pk] = _paired_ci(ck_p, r5_p)

    return {
        "baselines_clean": base_summary,
        "ablations_clean": ab_summary,
        "conditions": cond_summary,
        "protected": prot_summary,
        "parameter_audit": seed_results[0]["parameter_audit"],
    }


def evaluate_protocol_gates(summary: dict[str, Any], peak_commit_bytes: int) -> dict[str, Any]:
    """Evaluate all 10 pre-registered GO/STOP gates from protocol-v1.md."""
    c_a = summary["conditions"]["ckcer_mode_a"]
    d_a = summary["conditions"]["paired_deltas_mode_a_vs_r5"]
    fk = summary["baselines_clean"]["first_key_shortcut"]
    prot = summary["protected"]["ckcer"]
    prot_r5 = summary["protected"]["r5"]
    b_clean = summary["baselines_clean"]

    gates = {
        "G1_direct_beats_fk_plus_10pp_and_r5": bool(
            c_a["clean"]["direct"]["mean"] > fk["direct"]["mean"] + 0.10
            and d_a["clean"]["direct"]["ci95_low"] > 0.0
        ),
        "G2_clean_caseA_both_correct_ge_075_and_beats_r5": bool(
            c_a["clean"]["both_A"]["mean"] >= 0.75
            and c_a["clean"]["both_A"]["mean"] > 0.6676
            and d_a["clean"]["both_A"]["ci95_low"] > 0.0
        ),
        "G3_clean_caseB_both_correct_ge_090": bool(c_a["clean"]["both_B"]["mean"] >= 0.90),
        "G4_timing_shift_direct_ge_080_caseA_ge_065": bool(
            c_a["train_shift_test_shift"]["direct"]["mean"] >= 0.80
            and c_a["train_shift_test_shift"]["both_A"]["mean"] >= 0.65
            and c_a["ood_clean_to_shift"]["direct"]["mean"] >= 0.80
            and c_a["ood_clean_to_shift"]["both_A"]["mean"] >= 0.65
        ),
        "G5_reversed_writes_direct_ge_080_caseA_ge_065": bool(
            c_a["reversed_writes"]["direct"]["mean"] >= 0.80
            and c_a["reversed_writes"]["both_A"]["mean"] >= 0.65
        ),
        "G6_distractors_caseA_ge_060": bool(
            c_a["distractors"]["both_A"]["mean"] >= 0.60
            and c_a["paired_same_sample_distractors"]["both_A"]["mean"] >= 0.60
        ),
        "G7_noise_change_caseA_ge_060": bool(c_a["noise_change"]["both_A"]["mean"] >= 0.60),
        "G8_protected_tasks_no_regression_gt_1pp": bool(
            prot["sim_xor"]["mean"] >= prot_r5["sim_xor"]["mean"] - 0.01
            and prot["xor_separated"]["mean"] >= prot_r5["xor_separated"]["mean"] - 0.01
            and prot["order_task"]["mean"] >= prot_r5["order_task"]["mean"] - 0.01
        ),
        "G9_integrity_and_falsification_controls": bool(
            0.40 <= b_clean["random_label_control"]["direct"]["mean"] <= 0.60
            and b_clean["cue_shuf_test"]["both_A"]["mean"] <= 0.25
            and b_clean["fixed_nonlearning_candidate"]["both_A"]["mean"] == 0.0
        ),
        "G10_resource_limit_le_2gb": bool(0 < peak_commit_bytes <= 2_000_000_000),
    }
    return {
        "gates": gates,
        "all_passed": bool(all(gates.values())),
        "verdict": "GO_CONFIRMATION" if all(gates.values()) else "STOP",
    }
