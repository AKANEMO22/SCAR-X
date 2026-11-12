"""SCAR-X Round 9b Factorial MTSG e-prop with Strict Matched Controls.

Analytical 4x3x2 = 24 factorial design:
- Factor A (Eligibility decay): uniform_fast (0.25), uniform_mid (0.60), uniform_slow (0.92), heterogeneous (linspace 0.25..0.92)
- Factor B (Sensory gating): none, recurrent_only, soft_sensory
- Factor C (Eligibility reset): own_reset (subtractive reset Jacobian), stopped_reset (stopped reset derivative)

Strict matched controls:
- learned: genuine credit assignment
- shuffled_signal: identical traces, surrogate, gating, reset, optimizer; ONLY credit signal permuted across neurons
- zero_signal: identical traces; zero credit signal to hidden layer
- frozen_hidden: identical forward dynamics; hidden weights frozen
- readout_only: only readout weights updated; hidden frozen
"""
from __future__ import annotations

import copy
import io
import json
import numpy as np

from .core import Config, SpikingCore
from ..evaluation.readout_eligibility import spike_slope


DECAY_MODES = ("uniform_fast", "uniform_mid", "uniform_slow", "heterogeneous")
GATING_MODES = ("none", "recurrent_only", "soft_sensory")
RESET_MODES = ("own_reset", "stopped_reset")
CONTROL_MODES = ("learned", "shuffled_signal", "zero_signal", "frozen_hidden", "readout_only")


class FactorialMTSGEProp:
    """Unified factorial learner implementing all 24 configurations and strict matched controls."""

    def __init__(
        self,
        config: Config,
        decay_mode: str = "heterogeneous",
        gating_mode: str = "recurrent_only",
        reset_mode: str = "own_reset",
        control_mode: str = "learned",
        sensory_threshold: float = 0.6,
        lr_hid: float = 0.008,
        clip_hid: float = 0.010,
        lr_out: float = 0.04,
        clip_out: float = 0.025,
    ) -> None:
        if decay_mode not in DECAY_MODES:
            raise ValueError(f"Unknown decay_mode: {decay_mode}. Must be one of {DECAY_MODES}")
        if gating_mode not in GATING_MODES:
            raise ValueError(f"Unknown gating_mode: {gating_mode}. Must be one of {GATING_MODES}")
        if reset_mode not in RESET_MODES:
            raise ValueError(f"Unknown reset_mode: {reset_mode}. Must be one of {RESET_MODES}")
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"Unknown control_mode: {control_mode}. Must be one of {CONTROL_MODES}")

        if config.neuron != "alif" or config.hidden != 48 or config.fan_in != 8:
            raise ValueError("SCAR-X R9b requires ALIF SpikingCore with hidden=48, fan_in=8")
        config.validate()

        self.config = config
        self.decay_mode = decay_mode
        self.gating_mode = gating_mode
        self.reset_mode = reset_mode
        self.control_mode = control_mode
        self.sensory_threshold = float(sensory_threshold)

        self.lr_hid = float(lr_hid)
        self.clip_hid = float(clip_hid)
        self.lr_out = float(lr_out)
        self.clip_out = float(clip_out)

        self.core = SpikingCore(config)
        h = config.hidden
        d_tot = config.inputs + config.fan_in

        # Eligibility tensors
        self.e_v = np.zeros((h, d_tot), dtype=np.float32)
        self.e_a = np.zeros((h, d_tot), dtype=np.float32)
        self.e_f = np.zeros((h, d_tot), dtype=np.float32)

        # Synaptic momentum and energy normalizers
        self.syn_trace = np.zeros((h, d_tot), dtype=np.float32)
        self.syn_energy = np.full((h, d_tot), 1e-4, dtype=np.float32)

        # Running feature statistics for standardization
        self.f_mean = np.zeros(h, dtype=np.float32)
        self.f_var = np.full(h, 0.01, dtype=np.float32)

        # Initial hidden weights for causal revert testing
        self.w_in_0 = self.core.w_in.copy()
        self.w_rec_0 = self.core.w_rec.copy()

        # Decay vector gamma_e_vec
        self.gamma_e_vec = np.zeros((h, 1), dtype=np.float32)
        self._init_decay_vector()

        # Deterministic RNG for matched signal shuffling
        self.signal_rng = np.random.default_rng(config.seed + 920261112)

        self.calibrated = False
        self.episodes_learned = 0
        self.hidden_updates = 0
        self.total_ticks = 0

        # Per-step audit metrics
        self.last_step_eligibility_norm = 0.0
        self.last_step_signal_norm = 0.0
        self.last_step_update_norm = 0.0

    def _init_decay_vector(self) -> None:
        h = self.config.hidden
        if self.decay_mode == "uniform_fast":
            self.gamma_e_vec.fill(0.25)
        elif self.decay_mode == "uniform_mid":
            self.gamma_e_vec.fill(0.60)
        elif self.decay_mode == "uniform_slow":
            self.gamma_e_vec.fill(0.92)
        elif self.decay_mode == "heterogeneous":
            self.gamma_e_vec[:] = np.linspace(0.25, 0.92, h, dtype=np.float32)[:, None]

    def calibrate(self, sequences: np.ndarray) -> None:
        """Standardizes feature statistics using baseline sequences."""
        seqs = np.asarray(sequences, dtype=np.float32)
        if seqs.ndim != 3 or seqs.shape[2] != self.config.inputs or len(seqs) < 1:
            raise ValueError("Invalid calibration sequences")
        if not np.all(np.isfinite(seqs)):
            raise ValueError("Non-finite values in calibration sequences")
        for seq in seqs:
            self.core.reset_state()
            for row in seq:
                self.core.step(row, track=False)
            f = self.core.features
            self.f_mean = 0.95 * self.f_mean + 0.05 * f
            self.f_var = 0.95 * self.f_var + 0.05 * (f - self.f_mean) ** 2
        self.f_var = np.maximum(1e-6, self.f_var)
        self.calibrated = True

    def reset_episode(self) -> None:
        self.core.reset_state()
        self.e_v.fill(0.0)
        self.e_a.fill(0.0)
        self.e_f.fill(0.0)

    def compute_sensory_gate(self, row: np.ndarray) -> float:
        s_act = float(np.max(np.abs(row)))
        if self.gating_mode == "none":
            return 1.0
        elif self.gating_mode == "recurrent_only":
            return 1.0 if s_act > self.sensory_threshold else 0.15
        elif self.gating_mode == "soft_sensory":
            # Smooth sigmoidal gate between 0.15 and 1.0
            diff = (s_act - self.sensory_threshold) / 0.10
            diff = np.clip(diff, -10.0, 10.0)
            sig = 1.0 / (1.0 + np.exp(-diff))
            return float(0.15 + 0.85 * sig)
        return 1.0

    def step(self, x: np.ndarray, track: bool = True) -> np.ndarray:
        c = self.config
        row = np.asarray(x, dtype=np.float32)
        if row.shape != (c.inputs,) or not np.isfinite(row).all():
            raise ValueError("Invalid input vector")

        # All conditions, including controls, compute identical eligibility dynamics
        if track:
            old_z = self.core.spikes
            theta = c.threshold + c.adaptation_strength * self.core.adaptation
            i_in = self.core.w_in @ row
            i_rec = np.sum(self.core.w_rec * old_z[self.core.indices], axis=1)
            u_carry = c.leak * self.core.voltage + i_rec
            u = u_carry + i_in
            psi = spike_slope(u - theta)
            pre = np.concatenate(
                (np.broadcast_to(row, (c.hidden, c.inputs)), old_z[self.core.indices]),
                axis=1,
            )
            u_e = c.leak * self.e_v + pre
            a_old = self.e_a.copy()
            z_e = psi[:, None] * (u_e - c.adaptation_strength * a_old)

        self.core.step(row, track=False)

        if track:
            # Factor C: Reset Jacobian formulation
            if self.reset_mode == "own_reset":
                # Baseline subtractive reset
                self.e_v[:] = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
            elif self.reset_mode == "stopped_reset":
                # Reset-aware derivative (stops voltage reset jump from zeroing trace)
                self.e_v[:] = u_e

            self.e_a[:] = c.adaptation_decay * a_old + z_e

            # Factor B: Sensory gating
            z_e_gated = z_e.copy()
            g_rec = self.compute_sensory_gate(row)
            if self.gating_mode in ("recurrent_only", "soft_sensory"):
                z_e_gated[:, c.inputs:] *= g_rec

            # Factor A: Timescale decay filtering
            self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e_gated

        self.total_ticks += 1
        return self.core.features

    def standardized_features(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        safe_var = np.maximum(self.f_var, 0.0)
        scale = np.maximum(np.sqrt(safe_var), 0.05).astype(np.float32)
        raw = (f - self.f_mean) / scale
        f_hat = np.clip(raw, -5.0, 5.0).astype(np.float32)
        return f_hat, raw, scale

    def probabilities_from_features(self, f: np.ndarray) -> np.ndarray:
        f_hat, _, _ = self.standardized_features(f)
        logits = self.core.w_out @ f_hat + self.core.bias
        logits = logits - np.max(logits)
        p = np.exp(logits)
        return (p / np.sum(p)).astype(np.float32)

    def learn_episode(self, sequence: np.ndarray, target: int, stat_rate: float = 0.01) -> float:
        if not self.calibrated:
            raise ValueError("Model must be calibrated before supervised learning")
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2 or not 1 <= len(seq) <= 256 or seq.shape[1] != self.config.inputs:
            raise ValueError("Invalid episode sequence shape")
        if not np.all(np.isfinite(seq)):
            raise ValueError("Non-finite values in episode sequence")
        if not 0 <= int(target) < self.config.outputs:
            raise ValueError("Invalid target index")

        self.reset_episode()
        for row in seq:
            self.step(row, track=True)

        f = self.core.features
        self.f_mean = (1.0 - stat_rate) * self.f_mean + stat_rate * f
        self.f_var = np.maximum(1e-6, (1.0 - stat_rate) * self.f_var + stat_rate * (f - self.f_mean) ** 2)
        f_hat, raw, scale = self.standardized_features(f)

        logits = self.core.w_out @ f_hat + self.core.bias
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p = p / np.sum(p)
        loss = -float(np.log(max(float(p[int(target)]), 1e-12)))

        err = -p.copy()
        err[int(target)] += 1.0

        # Pre-update credit signal before modifying readout weights
        pre_update_signal = (self.core.w_out.T @ err) * (np.abs(raw) < 5.0) / scale

        # Readout update (only when not frozen_hidden)
        if self.control_mode != "frozen_hidden":
            grad_out = err[:, None] * f_hat[None, :]
            self.core.w_out += np.clip(self.lr_out * grad_out, -self.clip_out, self.clip_out)
            self.core.bias += np.clip(self.lr_out * err, -self.clip_out, self.clip_out)
            np.clip(self.core.w_out, -3.5, 3.5, out=self.core.w_out)
            np.clip(self.core.bias, -3.5, 3.5, out=self.core.bias)

        # Log norms
        self.last_step_eligibility_norm = float(np.linalg.norm(self.e_f))
        self.last_step_signal_norm = float(np.linalg.norm(pre_update_signal))

        # Hidden update according to control_mode
        if self.control_mode not in ("frozen_hidden", "readout_only"):
            if self.control_mode == "learned":
                signal = pre_update_signal
            elif self.control_mode == "shuffled_signal":
                perm = self.signal_rng.permutation(self.config.hidden)
                signal = pre_update_signal[perm]
            elif self.control_mode == "zero_signal":
                signal = np.zeros_like(pre_update_signal)
            else:
                signal = pre_update_signal

            direction = signal[:, None] * self.e_f

            self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
            self.syn_energy = 0.99 * self.syn_energy + 0.01 * (direction * direction)
            rms = np.sqrt(self.syn_energy) + 1e-4
            norm_step = self.syn_trace / rms

            step = np.clip(self.lr_hid * norm_step, -self.clip_hid, self.clip_hid)
            self.last_step_update_norm = float(np.linalg.norm(step))

            d = self.config.inputs
            self.core.w_in += step[:, :d]
            self.core.w_rec += step[:, d:]
            np.clip(self.core.w_in, -3.0, 3.0, out=self.core.w_in)
            np.clip(self.core.w_rec, -3.0, 3.0, out=self.core.w_rec)
            self.hidden_updates += 1
        else:
            self.last_step_update_norm = 0.0

        self.episodes_learned += 1
        return loss

    def revert_hidden(self) -> None:
        """Causal intervention: strictly reverts hidden synaptic weights to initialization."""
        self.core.w_in[:] = self.w_in_0
        self.core.w_rec[:] = self.w_rec_0

    def features(self, sequence: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2 or seq.shape[1] != self.config.inputs or not np.all(np.isfinite(seq)):
            raise ValueError("Invalid sequence for features extraction")
        self.core.reset_state()
        for row in seq:
            self.core.step(row, track=False)
        return self.core.features.copy()

    def features_many(self, sequences: np.ndarray) -> np.ndarray:
        return np.stack([self.features(seq) for seq in sequences])

    def predict(self, sequence: np.ndarray) -> int:
        f = self.features(sequence)
        p = self.probabilities_from_features(f)
        return int(np.argmax(p))

    def predict_many(self, sequences: np.ndarray) -> np.ndarray:
        return np.array([self.predict(seq) for seq in sequences], dtype=np.int32)

    def clone(self) -> FactorialMTSGEProp:
        return copy.deepcopy(self)
