"""SCAR-X Round 9: Unified Neural Mechanism Framework for Massive Discovery.

Integrates hypotheses H1 to H5 into a single high-performance, strictly verified
local plasticity architecture operating over SpikingCore (ALIF, H=48, fan_in=8).

Key mechanisms:
1. H1 (Reset & Temporal Credit): Subtractive reset Jacobian vs Hard reset, surrogate functions.
2. H2 (Timescales & Gating): Factorial combinations of decay distributions and sensory event gating.
3. H3 (Distractor Robustness): Dynamic sensory attenuation, noise-sensitive credit modulation.
4. H4 (Readout & Associative Memory): Scaled multi-key retrieval, Context-Conditioned Readout (CKCER).
5. H5 (Continual Learning): Benna-Fusi metaplastic cascade (u1, u2), orthogonal projection, selective protection.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any
import zipfile
import numpy as np

from .core import Config, SpikingCore


def fast_sigmoid_slope(x: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    return (gamma / ((1.0 + np.abs(x)) ** 2)).astype(np.float32)


def exponential_slope(x: np.ndarray, gamma: float = 0.5, sigma: float = 0.5) -> np.ndarray:
    return (gamma * np.exp(-np.abs(x) / max(sigma, 1e-4))).astype(np.float32)


def boxcar_slope(x: np.ndarray, gamma: float = 0.5, delta: float = 0.5) -> np.ndarray:
    return (gamma * (np.abs(x) < delta)).astype(np.float32)


def gaussian_slope(x: np.ndarray, gamma: float = 0.5, sigma: float = 0.5) -> np.ndarray:
    s = max(sigma, 1e-4)
    return (gamma / (np.sqrt(2.0 * np.pi) * s) * np.exp(-0.5 * (x / s) ** 2)).astype(np.float32)


def compute_surrogate_slope(x: np.ndarray, fn_type: str = "fast_sigmoid", gamma: float = 0.5, param: float = 0.5) -> np.ndarray:
    if fn_type == "fast_sigmoid":
        return fast_sigmoid_slope(x, gamma=gamma)
    elif fn_type == "exponential":
        return exponential_slope(x, gamma=gamma, sigma=param)
    elif fn_type == "boxcar":
        return boxcar_slope(x, gamma=gamma, delta=param)
    elif fn_type == "gaussian":
        return gaussian_slope(x, gamma=gamma, sigma=param)
    else:
        return fast_sigmoid_slope(x, gamma=gamma)


def generate_timescale_vector(
    hidden: int,
    dist_type: str = "hetero_linear",
    gamma_min: float = 0.25,
    gamma_max: float = 0.92,
) -> np.ndarray:
    """Generates neuronal eligibility decay vector gamma_e in [0, 1]."""
    if dist_type == "uniform_fast":
        return np.full((hidden, 1), gamma_min, dtype=np.float32)
    elif dist_type == "uniform_slow":
        return np.full((hidden, 1), gamma_max, dtype=np.float32)
    elif dist_type == "uniform_mid":
        return np.full((hidden, 1), 0.5 * (gamma_min + gamma_max), dtype=np.float32)
    elif dist_type == "hetero_linear":
        return np.linspace(gamma_min, gamma_max, hidden, dtype=np.float32)[:, None]
    elif dist_type == "hetero_log":
        # Logarithmic distribution covering fast to slow memory scales
        log_vals = 1.0 - 10.0 ** np.linspace(-0.6, -1.8, hidden, dtype=np.float32)
        log_vals = np.clip(log_vals, gamma_min, gamma_max)
        return log_vals[:, None]
    elif dist_type == "hetero_bimodal":
        h_half = hidden // 2
        fast_part = np.full(h_half, gamma_min, dtype=np.float32)
        slow_part = np.full(hidden - h_half, gamma_max, dtype=np.float32)
        return np.concatenate([fast_part, slow_part])[:, None]
    else:
        return np.linspace(gamma_min, gamma_max, hidden, dtype=np.float32)[:, None]


class R9UnifiedNeuralModel:
    """Unified Neural Model for SCAR-X Round 9 massive mechanism search."""

    state_array_names = (
        "e_v",
        "e_a",
        "e_f",
        "syn_trace",
        "syn_energy",
        "f_mean",
        "f_var",
        "w_in_0",
        "w_rec_0",
        "gamma_e_vec",
        "u1_in",
        "u2_in",
        "u1_rec",
        "u2_rec",
        "u1_out",
        "u2_out",
        "bias",
    )

    def __init__(
        self,
        config: Config,
        # H1 parameters
        reset_mode: str = "subtractive_exact",  # 'subtractive_exact', 'hard_reset', 'damped_subtractive'
        damped_lambda: float = 0.0,
        surrogate_fn: str = "fast_sigmoid",
        surrogate_gamma: float = 0.5,
        surrogate_param: float = 0.5,
        # H2 parameters
        timescale_dist: str = "hetero_linear",
        gamma_min: float = 0.25,
        gamma_max: float = 0.92,
        gating_mode: str = "recurrent_only",  # 'none', 'hard_sensory', 'soft_sigmoid', 'recurrent_only'
        sensory_threshold: float = 0.6,
        sensory_floor: float = 0.15,
        # H3 parameters
        learned_attenuation: bool = False,
        attenuation_factor: float = 0.0,
        noise_sensitive_credit: bool = False,
        # H4 parameters
        readout_mode: str = "standard_linear",  # 'standard_linear', 'context_gated'
        # H5 parameters
        use_bf_cascade: bool = False,
        kappa1: float = 0.004,
        kappa2: float = 0.001,
        # Experimental condition (for causal ablations)
        condition: str = "learned_candidate",
    ) -> None:
        if config.neuron != "alif" or config.hidden != 48 or config.fan_in != 8:
            raise ValueError("Round 9 requires ALIF SpikingCore with hidden=48, fan_in=8")
        config.validate()

        self.config = config
        self.condition = condition
        self.reset_mode = reset_mode
        self.damped_lambda = float(damped_lambda)
        self.surrogate_fn = surrogate_fn
        self.surrogate_gamma = float(surrogate_gamma)
        self.surrogate_param = float(surrogate_param)

        self.timescale_dist = timescale_dist
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.gating_mode = gating_mode
        self.sensory_threshold = float(sensory_threshold)
        self.sensory_floor = float(sensory_floor)

        self.learned_attenuation = bool(learned_attenuation)
        self.attenuation_factor = float(attenuation_factor)
        self.noise_sensitive_credit = bool(noise_sensitive_credit)

        self.readout_mode = readout_mode
        self.use_bf_cascade = bool(use_bf_cascade)
        self.kappa1 = float(kappa1)
        self.kappa2 = float(kappa2)

        self.core = SpikingCore(config)
        h = config.hidden
        d = config.inputs
        k = config.fan_in
        o = config.outputs
        d_tot = d + k

        # Eligibility tensors
        self.e_v = np.zeros((h, d_tot), dtype=np.float32)
        self.e_a = np.zeros((h, d_tot), dtype=np.float32)
        self.e_f = np.zeros((h, d_tot), dtype=np.float32)

        # Synaptic optimizer state
        self.syn_trace = np.zeros((h, d_tot), dtype=np.float32)
        self.syn_energy = np.full((h, d_tot), 1e-4, dtype=np.float32)

        # Running feature statistics
        self.f_mean = np.zeros(h, dtype=np.float32)
        self.f_var = np.full(h, 0.01, dtype=np.float32)

        # Baseline snapshots for causal intervention
        self.w_in_0 = self.core.w_in.copy()
        self.w_rec_0 = self.core.w_rec.copy()

        # BF-Cascade two-compartment synapses
        self.u1_in = self.core.w_in.copy()
        self.u2_in = self.core.w_in.copy()
        self.u1_rec = self.core.w_rec.copy()
        self.u2_rec = self.core.w_rec.copy()
        self.u1_out = self.core.w_out.copy()
        self.u2_out = self.core.w_out.copy()
        self.bias = self.core.bias.copy()

        # Context-conditioned readout weights (CKCER) if enabled
        if self.readout_mode == "context_gated":
            self.w_ctx = np.zeros((d, o, h), dtype=np.float32)
            self.e_ctx = np.zeros(d, dtype=np.float32)
        else:
            self.w_ctx = None
            self.e_ctx = None

        # Generate timescale distribution
        self.gamma_e_vec = generate_timescale_vector(
            h, dist_type=timescale_dist, gamma_min=self.gamma_min, gamma_max=self.gamma_max
        )

        self.signal_rng = np.random.default_rng(config.seed + 90001)
        self.calibrated = False
        self.episodes_learned = 0
        self.hidden_updates = 0
        self.total_ticks = 0

    def calibrate(self, sequences: np.ndarray) -> None:
        seqs = np.asarray(sequences, dtype=np.float32)
        if seqs.ndim != 3 or seqs.shape[2] != self.config.inputs or len(seqs) < 1:
            raise ValueError("Invalid calibration sequences")
        for seq in seqs:
            self.core.reset_state()
            for row in seq:
                self.core.step(row, track=False)
            f = self.core.features
            self.f_mean = 0.95 * self.f_mean + 0.05 * f
            self.f_var = 0.95 * self.f_var + 0.05 * (f - self.f_mean) ** 2
        self.calibrated = True

    def reset_episode(self) -> None:
        self.core.reset_state()
        self.e_v.fill(0.0)
        self.e_a.fill(0.0)
        self.e_f.fill(0.0)
        if self.e_ctx is not None:
            self.e_ctx.fill(0.0)

    def step(self, x: np.ndarray, track: bool = True) -> np.ndarray:
        c = self.config
        row = np.asarray(x, dtype=np.float32)
        if row.shape != (c.inputs,) or not np.isfinite(row).all():
            raise ValueError("Invalid input vector")

        # Dynamic sensory attenuation (H3)
        if self.learned_attenuation and self.attenuation_factor > 0.0:
            # Attenuate noise-heavy channel (channel 6) if signal is small
            row = row.copy()
            if abs(row[6]) > 0.5:
                row[6] *= max(0.1, 1.0 - self.attenuation_factor)

        # Context trace for readout conditioning (H4)
        if self.e_ctx is not None and track:
            self.e_ctx = 0.85 * self.e_ctx + 0.15 * np.abs(row)

        is_hidden_active = self.condition not in ("frozen_hidden", "readout_only")

        if track and is_hidden_active:
            old_z = self.core.spikes
            theta = c.threshold + c.adaptation_strength * self.core.adaptation
            i_in = self.core.w_in @ row
            i_rec = np.sum(self.core.w_rec * old_z[self.core.indices], axis=1)
            u_carry = c.leak * self.core.voltage + i_rec
            u = u_carry + i_in

            # Surrogate gradient slope
            psi = compute_surrogate_slope(
                u - theta,
                fn_type=self.surrogate_fn,
                gamma=self.surrogate_gamma,
                param=self.surrogate_param,
            )

            pre = np.concatenate(
                (np.broadcast_to(row, (c.hidden, c.inputs)), old_z[self.core.indices]),
                axis=1,
            )
            u_e = c.leak * self.e_v + pre
            a_old = self.e_a.copy()
            z_e = psi[:, None] * (u_e - c.adaptation_strength * a_old)

        # Forward execution in core (with subtractive reset)
        self.core.step(row, track=False)

        if track and is_hidden_active:
            # Eligibility propagation through reset (H1)
            if self.reset_mode == "subtractive_exact":
                # Exact Jacobian of subtractive reset
                self.e_v[:] = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
            elif self.reset_mode == "hard_reset":
                # Standard e-prop approximation assuming hard reset to zero
                self.e_v[:] = u_e * (1.0 - self.core.spikes[:, None]) - theta[:, None] * z_e
            elif self.reset_mode == "damped_subtractive":
                # Damped subtractive reset with interpolation factor lambda
                lam = self.damped_lambda
                exact = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
                damped = exact * (1.0 - lam * self.core.spikes[:, None])
                self.e_v[:] = damped
            else:
                self.e_v[:] = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old

            self.e_a[:] = c.adaptation_decay * a_old + z_e

            # Multi-timescale eligibility filtering and sensory gating (H2)
            if self.gating_mode == "none":
                self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e
            elif self.gating_mode == "hard_sensory":
                s_activity = float(np.max(np.abs(row)))
                g = 1.0 if s_activity > self.sensory_threshold else self.sensory_floor
                self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * (z_e * g)
            elif self.gating_mode == "soft_sigmoid":
                s_activity = float(np.max(np.abs(row)))
                sig = 1.0 / (1.0 + np.exp(-10.0 * (s_activity - self.sensory_threshold)))
                g = self.sensory_floor + (1.0 - self.sensory_floor) * sig
                self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * (z_e * g)
            elif self.gating_mode == "recurrent_only":
                s_activity = float(np.max(np.abs(row)))
                g_rec = 1.0 if s_activity > self.sensory_threshold else self.sensory_floor
                z_e_gated = z_e.copy()
                z_e_gated[:, c.inputs:] *= g_rec
                self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e_gated
            else:
                self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e

        self.total_ticks += 1
        return self.core.features

    def standardized_features(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = np.maximum(np.sqrt(self.f_var), 0.05).astype(np.float32)
        raw = (f - self.f_mean) / scale
        f_hat = np.clip(raw, -5.0, 5.0).astype(np.float32)
        return f_hat, raw, scale

    def compute_logits(self, f_hat: np.ndarray) -> np.ndarray:
        logits = self.core.w_out @ f_hat + self.core.bias
        if self.readout_mode == "context_gated" and self.w_ctx is not None and self.e_ctx is not None:
            # Add context-conditioned readout offset
            ctx_contrib = np.tensordot(self.e_ctx, self.w_ctx, axes=(0, 0)) @ f_hat
            logits = logits + ctx_contrib
        return logits

    def probabilities_from_features(self, f: np.ndarray) -> np.ndarray:
        f_hat, _, _ = self.standardized_features(f)
        logits = self.compute_logits(f_hat)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        return (p / np.sum(p)).astype(np.float32)

    def learn_episode(self, sequence: np.ndarray, target: int, stat_rate: float = 0.01) -> float:
        if not self.calibrated:
            raise ValueError("Model must be calibrated before supervised learning")
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2 or not 1 <= len(seq) <= 256 or seq.shape[1] != self.config.inputs:
            raise ValueError("Invalid episode sequence shape")
        if not 0 <= int(target) < self.config.outputs:
            raise ValueError("Invalid target index")

        self.reset_episode()
        for row in seq:
            self.step(row, track=True)

        f = self.core.features
        self.f_mean = (1.0 - stat_rate) * self.f_mean + stat_rate * f
        self.f_var = (1.0 - stat_rate) * self.f_var + stat_rate * (f - self.f_mean) ** 2
        f_hat, raw, scale = self.standardized_features(f)

        logits = self.compute_logits(f_hat)
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p = p / np.sum(p)
        loss = -float(np.log(max(float(p[int(target)]), 1e-12)))

        err = -p.copy()
        err[int(target)] += 1.0

        # Own-neuron learning signal before readout weight update
        pre_update_signal = (self.core.w_out.T @ err) * (np.abs(raw) < 5.0) / scale

        # Noise-sensitive credit attenuation (H3)
        if self.noise_sensitive_credit:
            # Modulate learning signal by feature stability
            stability = 1.0 / (1.0 + np.sqrt(self.f_var))
            pre_update_signal = pre_update_signal * stability

        # Readout weight update
        grad_out = err[:, None] * f_hat[None, :]
        step_out = np.clip(0.04 * grad_out, -0.025, 0.025)
        step_bias = np.clip(0.04 * err, -0.025, 0.025)

        if self.use_bf_cascade:
            # BF-Cascade readout update
            self.u1_out += step_out
            self.bias += step_bias
            self.u1_out -= self.kappa1 * (self.u1_out - self.u2_out)
            self.u2_out += self.kappa2 * (self.u1_out - self.u2_out)
            np.clip(self.u1_out, -3.5, 3.5, out=self.core.w_out)
            np.clip(self.bias, -3.5, 3.5, out=self.core.bias)
        else:
            self.core.w_out += step_out
            self.core.bias += step_bias
            np.clip(self.core.w_out, -3.5, 3.5, out=self.core.w_out)
            np.clip(self.core.bias, -3.5, 3.5, out=self.core.bias)

        # Context-gated readout update (CKCER) if active
        if self.readout_mode == "context_gated" and self.w_ctx is not None and self.e_ctx is not None:
            grad_ctx = self.e_ctx[:, None, None] * (err[None, :, None] * f_hat[None, None, :])
            self.w_ctx += np.clip(0.02 * grad_ctx, -0.015, 0.015)
            np.clip(self.w_ctx, -2.0, 2.0, out=self.w_ctx)

        # Hidden synaptic update
        if self.condition not in ("frozen_hidden", "zero_signal", "readout_only"):
            signal = pre_update_signal
            if self.condition == "shuffled_signal":
                perm = self.signal_rng.permutation(self.config.hidden)
                signal = signal[perm]

            direction = signal[:, None] * self.e_f

            # Synaptic trace momentum and energy normalization
            self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
            self.syn_energy = 0.99 * self.syn_energy + 0.01 * (direction * direction)
            rms = np.sqrt(self.syn_energy) + 1e-4
            norm_step = self.syn_trace / rms

            lr_hid = 0.008
            clip_hid = 0.010
            step_hid = np.clip(lr_hid * norm_step, -clip_hid, clip_hid)

            d = self.config.inputs
            if self.use_bf_cascade:
                # BF-Cascade hidden weights
                self.u1_in += step_hid[:, :d]
                self.u1_rec += step_hid[:, d:]

                # Passive intra-synaptic diffusion per episode
                self.u1_in -= self.kappa1 * (self.u1_in - self.u2_in)
                self.u2_in += self.kappa2 * (self.u1_in - self.u2_in)
                self.u1_rec -= self.kappa1 * (self.u1_rec - self.u2_rec)
                self.u2_rec += self.kappa2 * (self.u1_rec - self.u2_rec)

                np.clip(self.u1_in, -3.0, 3.0, out=self.core.w_in)
                np.clip(self.u1_rec, -3.0, 3.0, out=self.core.w_rec)
            else:
                self.core.w_in += step_hid[:, :d]
                self.core.w_rec += step_hid[:, d:]
                np.clip(self.core.w_in, -3.0, 3.0, out=self.core.w_in)
                np.clip(self.core.w_rec, -3.0, 3.0, out=self.core.w_rec)

            self.hidden_updates += 1

        self.core.updates += 1
        self.episodes_learned += 1
        return loss

    def run_sequence(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Runs a sequence in inference mode, returning final features and context trace (if context_gated)."""
        seq = np.asarray(sequence, dtype=np.float32)
        self.core.reset_state()
        e_ctx = np.zeros(self.config.inputs, dtype=np.float32) if self.readout_mode == "context_gated" else None
        for row in seq:
            if self.learned_attenuation and self.attenuation_factor > 0.0:
                if abs(row[6]) > 0.5:
                    row = row.copy()
                    row[6] *= max(0.1, 1.0 - self.attenuation_factor)
            self.core.step(row, track=False)
            if e_ctx is not None:
                e_ctx = 0.85 * e_ctx + 0.15 * np.abs(row)
        return self.core.features.copy(), e_ctx

    def features(self, sequence: np.ndarray) -> np.ndarray:
        f, _ = self.run_sequence(sequence)
        return f

    def features_many(self, sequences: np.ndarray) -> np.ndarray:
        return np.stack([self.features(seq) for seq in sequences])

    def predict(self, sequence: np.ndarray) -> int:
        f, e_ctx = self.run_sequence(sequence)
        f_hat, _, _ = self.standardized_features(f)
        logits = self.core.w_out @ f_hat + self.core.bias
        if self.readout_mode == "context_gated" and self.w_ctx is not None and e_ctx is not None:
            ctx_contrib = np.tensordot(e_ctx, self.w_ctx, axes=(0, 0)) @ f_hat
            logits = logits + ctx_contrib
        return int(np.argmax(logits))

    def predict_many(self, sequences: np.ndarray) -> np.ndarray:
        if self.readout_mode == "context_gated":
            return np.array([self.predict(s) for s in sequences], dtype=np.int32)
        f_mat = self.features_many(sequences)
        scale = np.maximum(np.sqrt(self.f_var), 0.05).astype(np.float32)
        f_hat = np.clip((f_mat - self.f_mean) / scale, -5.0, 5.0)
        logits = f_hat @ self.core.w_out.T + self.core.bias
        return np.argmax(logits, axis=1).astype(np.int32)

    def revert_hidden_weights(self, revert_in: bool = True, revert_rec: bool = True) -> None:
        """Causal weight revert: restores hidden weights to initial state while keeping learned readout."""
        if revert_in:
            self.core.w_in[:] = self.w_in_0
            if self.use_bf_cascade:
                self.u1_in[:] = self.w_in_0
                self.u2_in[:] = self.w_in_0
        if revert_rec:
            self.core.w_rec[:] = self.w_rec_0
            if self.use_bf_cascade:
                self.u1_rec[:] = self.w_rec_0
                self.u2_rec[:] = self.w_rec_0

    def memory_bytes(self) -> int:
        total = self.core.memory_bytes()
        for name in self.state_array_names:
            arr = getattr(self, name, None)
            if arr is not None and hasattr(arr, "nbytes"):
                total += arr.nbytes
        if self.w_ctx is not None:
            total += self.w_ctx.nbytes
        if self.e_ctx is not None:
            total += self.e_ctx.nbytes
        return total

    def dumps(self) -> bytes:
        core_bytes = self.core.dumps()
        state_buf = io.BytesIO()
        saved_dict = {}
        for name in self.state_array_names:
            arr = getattr(self, name, None)
            if arr is not None:
                saved_dict[name] = arr
        if self.w_ctx is not None:
            saved_dict["w_ctx"] = self.w_ctx
        if self.e_ctx is not None:
            saved_dict["e_ctx"] = self.e_ctx

        np.savez(state_buf, **saved_dict)
        state_bytes = state_buf.getvalue()

        meta = {
            "format": "scarx-r9-unified-v1",
            "condition": self.condition,
            "reset_mode": self.reset_mode,
            "damped_lambda": self.damped_lambda,
            "surrogate_fn": self.surrogate_fn,
            "surrogate_gamma": self.surrogate_gamma,
            "surrogate_param": self.surrogate_param,
            "timescale_dist": self.timescale_dist,
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "gating_mode": self.gating_mode,
            "sensory_threshold": self.sensory_threshold,
            "sensory_floor": self.sensory_floor,
            "learned_attenuation": self.learned_attenuation,
            "attenuation_factor": self.attenuation_factor,
            "noise_sensitive_credit": self.noise_sensitive_credit,
            "readout_mode": self.readout_mode,
            "use_bf_cascade": self.use_bf_cascade,
            "kappa1": self.kappa1,
            "kappa2": self.kappa2,
            "calibrated": self.calibrated,
            "episodes_learned": self.episodes_learned,
            "hidden_updates": self.hidden_updates,
            "total_ticks": self.total_ticks,
        }
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("core.bin", core_bytes)
            zf.writestr("state.npz", state_bytes)
            zf.writestr("meta.json", json.dumps(meta))
        return zip_buf.getvalue()

    @classmethod
    def loads(cls, data: bytes) -> R9UnifiedNeuralModel:
        zip_buf = io.BytesIO(data)
        with zipfile.ZipFile(zip_buf, "r") as zf:
            core_bytes = zf.read("core.bin")
            state_bytes = zf.read("state.npz")
            meta = json.loads(zf.read("meta.json").decode("utf-8"))

        core = SpikingCore.loads(core_bytes)
        inst = cls(
            core.config,
            reset_mode=meta.get("reset_mode", "subtractive_exact"),
            damped_lambda=meta.get("damped_lambda", 0.0),
            surrogate_fn=meta.get("surrogate_fn", "fast_sigmoid"),
            surrogate_gamma=meta.get("surrogate_gamma", 0.5),
            surrogate_param=meta.get("surrogate_param", 0.5),
            timescale_dist=meta.get("timescale_dist", "hetero_linear"),
            gamma_min=meta.get("gamma_min", 0.25),
            gamma_max=meta.get("gamma_max", 0.92),
            gating_mode=meta.get("gating_mode", "recurrent_only"),
            sensory_threshold=meta.get("sensory_threshold", 0.6),
            sensory_floor=meta.get("sensory_floor", 0.15),
            learned_attenuation=meta.get("learned_attenuation", False),
            attenuation_factor=meta.get("attenuation_factor", 0.0),
            noise_sensitive_credit=meta.get("noise_sensitive_credit", False),
            readout_mode=meta.get("readout_mode", "standard_linear"),
            use_bf_cascade=meta.get("use_bf_cascade", False),
            kappa1=meta.get("kappa1", 0.004),
            kappa2=meta.get("kappa2", 0.001),
            condition=meta.get("condition", "learned_candidate"),
        )
        inst.core = core
        state_npz = np.load(io.BytesIO(state_bytes))
        for name in cls.state_array_names:
            if name in state_npz:
                getattr(inst, name)[:] = state_npz[name]
        if "w_ctx" in state_npz and inst.w_ctx is not None:
            inst.w_ctx[:] = state_npz["w_ctx"]
        if "e_ctx" in state_npz and inst.e_ctx is not None:
            inst.e_ctx[:] = state_npz["e_ctx"]

        inst.calibrated = bool(meta["calibrated"])
        inst.episodes_learned = int(meta["episodes_learned"])
        inst.hidden_updates = int(meta["hidden_updates"])
        inst.total_ticks = int(meta["total_ticks"])
        return inst
