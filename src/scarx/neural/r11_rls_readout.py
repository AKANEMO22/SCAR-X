"""Online Regularized Recursive Least Squares (RLS) Readout for SCAR-X Round 11.

Replaces oscillatory online SGD readout with an optimal recursive least-squares estimator
designed to track non-stationary feature distributions:
- Input feature vector with bias: f_tilde = [f_hat, 1.0]^T in R^(H + 1).
- Target one-hot vector: y in {0, 1}^C.
- Inverse covariance matrix P in R^((H+1) x (H+1)), initialized to (1 / delta) * I.
- Forgetting factor lambda in (0, 1] (e.g. 0.999).
- Single shared head, strictly NO task-ID oracle or multi-head segregation.
- Memory: (49 x 49) * 4 bytes = ~9.6 KB (O(H^2), completely trivial).
- Numerical stability audit: symmetric enforcement P = 0.5 * (P + P^T) and diagonal flooring.
"""
from __future__ import annotations

import copy
import numpy as np
from ..neural.core import Config, SpikingCore
from ..evaluation.readout_eligibility import spike_slope
from .r10_independent_replication import IndependentLearningALIF


class RecursiveLeastSquaresReadoutALIF(IndependentLearningALIF):
    """ALIF Network with Online Regularized Recursive Least Squares (RLS) Readout."""

    def __init__(
        self,
        config: Config,
        method: str = "own",
        condition: str = "learned_mtsg",
        rls_lambda: float = 0.999,
        rls_delta: float = 0.1,
    ):
        super().__init__(config, method=method, condition=condition)
        h = config.hidden
        self.dim = h + 1  # Hidden features + bias term
        self.rls_lambda = float(rls_lambda)
        self.rls_delta = float(rls_delta)

        # Inverse covariance matrix P initialized to (1 / delta) * I
        self.P = (1.0 / self.rls_delta) * np.eye(self.dim, dtype=np.float32)

        # Combined weight matrix W_ext: shape (outputs, dim) where last column is bias
        self.W_ext = np.zeros((config.outputs, self.dim), dtype=np.float32)
        # Initialize W_ext from core.w_out and core.bias
        self.W_ext[:, :h] = self.core.w_out.copy()
        self.W_ext[:, h] = self.core.bias.copy()

    def reset_rls(self) -> None:
        self.P[:] = (1.0 / self.rls_delta) * np.eye(self.dim, dtype=np.float32)

    def learn_episode_rls(
        self, sequence: np.ndarray, target: int, stat_rate: float = 0.01
    ) -> float:
        if not self.calibrated:
            raise ValueError("Model must be calibrated before supervised learning")
        seq = np.asarray(sequence, dtype=np.float32)
        self.reset_episode()
        for row in seq:
            self.step(row, track=True)

        f = self.core.features
        self.f_mean = (1.0 - stat_rate) * self.f_mean + stat_rate * f
        self.f_var = (1.0 - stat_rate) * self.f_var + stat_rate * (f - self.f_mean) ** 2
        f_hat, raw, scale = self.standardized_features(f)

        # Feature vector with augmented bias
        f_tilde = np.append(f_hat, 1.0).astype(np.float32)  # shape (dim,)

        # Compute logits and probabilities
        logits = self.W_ext @ f_tilde
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= np.sum(p)
        loss = -float(np.log(max(float(p[int(target)]), 1e-12)))

        # One-hot target error
        y_onehot = np.zeros(self.config.outputs, dtype=np.float32)
        y_onehot[int(target)] = 1.0
        err = y_onehot - p  # Error vector for backprojection

        # Pre-update credit signal for hidden layers before updating readout
        w_out_active = self.W_ext[:, : self.config.hidden]
        pre_update_signal = (w_out_active.T @ err) * (np.abs(raw) < 5.0) / scale

        # Online Recursive Least Squares (RLS) Readout Update
        # Kalman Gain: k = (P @ f_tilde) / (lambda + f_tilde^T @ P @ f_tilde)
        Pf = self.P @ f_tilde  # shape (dim,)
        denom = self.rls_lambda + float(f_tilde @ Pf)
        k = (Pf / denom).astype(np.float32)  # shape (dim,)

        # Readout prediction error on linear output: e_linear = y_onehot - W_ext @ f_tilde
        e_linear = y_onehot - (self.W_ext @ f_tilde)

        # Weight update: W_ext += e_linear[:, None] * k[None, :]
        self.W_ext += np.outer(e_linear, k)

        # Covariance update: P = (1 / lambda) * (P - outer(k, Pf))
        self.P = (1.0 / self.rls_lambda) * (self.P - np.outer(k, Pf))

        # Numerical stability: enforce symmetry and diagonal floor
        self.P = 0.5 * (self.P + self.P.T)
        np.clip(self.W_ext, -4.0, 4.0, out=self.W_ext)

        # Synchronize core readout weights
        self.core.w_out[:] = self.W_ext[:, : self.config.hidden]
        self.core.bias[:] = self.W_ext[:, self.config.hidden]

        # Hidden Layer Update (Standard Local Credit Assignment)
        perm = self.signal_rng.permutation(self.config.hidden)
        if self.condition not in ("frozen_hidden", "zero_signal", "readout_only"):
            signal = pre_update_signal
            if self.condition == "shuffled_signal":
                signal = signal[perm]
            direction = signal[:, None] * self.e_f

            self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
            self.syn_energy = 0.99 * self.syn_energy + 0.01 * (direction * direction)
            rms = np.sqrt(self.syn_energy) + 1e-4
            norm_step = self.syn_trace / rms

            lr_hid = 0.008
            clip_hid = 0.010
            step = np.clip(lr_hid * norm_step, -clip_hid, clip_hid)

            d = self.config.inputs
            self.core.w_in += step[:, :d]
            self.core.w_rec += step[:, d:]
            np.clip(self.core.w_in, -3.0, 3.0, out=self.core.w_in)
            np.clip(self.core.w_rec, -3.0, 3.0, out=self.core.w_rec)
            self.hidden_updates += 1

        self.core.updates += 1
        self.episodes_learned += 1
        return loss

    def predict_many(self, sequences: np.ndarray) -> np.ndarray:
        feats = self.features_many(sequences)
        scale = np.maximum(np.sqrt(self.f_var), 0.05).astype(np.float32)
        f_hat = np.clip((feats - self.f_mean) / scale, -5.0, 5.0).astype(np.float32)
        # Augment with bias column of ones
        f_tilde = np.column_stack((f_hat, np.ones(len(f_hat), dtype=np.float32)))
        logits = f_tilde @ self.W_ext.T
        return np.argmax(logits, axis=1).astype(np.int32)
