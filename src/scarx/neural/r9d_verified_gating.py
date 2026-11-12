"""SCAR-X Round 9d Verified Neural Gating & Dynamic Conductance Architectures.

Corrects all Phase 0 architectural bugs:
- Bug A: Corrected e-prop gradient for channel gating:
  e_gate[j, k] tracks neuron j sensitivity to channel k.
  Delta s_k = lr_gate * sum_j (delta_j * e_gate[j, k]).
  Non-zero, mathematically derived from Bellec et al. (Nature Communications 2020).
- Bug B: Only 2 principled candidates:
  Candidate A: learned_channel_eprop (E-prop learned channel selectivity)
  Candidate B: dgn_adaptive_conductance (Dynamic Gated Neuron, Bai et al., arXiv:2509.03281)
- Bug C: Unified engine for learned and shuffled_signal controls.
  All forward steps, traces, surrogates, and stats are 100% bit-identical before permutation.
- Bug D: Strict separation of direct causal drop, probe revert drop, frozen probe gain, and shuffled contrast.
- Bug E: Real process RSS measurement via psutil; real per-seed Student's t CI95.
- Bug F: Exact separation between R9b recurrent-trace gating and forward gating.
"""
from __future__ import annotations

import copy
import math
import os
from typing import Any
import numpy as np
import psutil

from scarx.neural.core import Config, SpikingCore
from scarx.evaluation.readout_eligibility import spike_slope


R9D_ARCHITECTURES = (
    "no_gating",
    "r9b_recurrent_gating",
    "learned_channel_eprop",
    "dgn_adaptive_conductance",
    "capacity_matched_control",
)

CONTROL_MODES = (
    "learned",
    "shuffled_signal",
    "zero_signal",
    "frozen_hidden",
    "readout_only",
)


class VerifiedNeuralGatingALIF:
    """Rigorous ALIF H48 implementation with verified e-prop gating and DGN conductance."""

    def __init__(
        self,
        config: Config,
        gating_arch: str = "learned_channel_eprop",
        control_mode: str = "learned",
        decay_mode: str = "uniform_fast",
        reset_mode: str = "stopped_reset",
        lr_hid: float = 0.008,
        lr_out: float = 0.04,
        lr_gate: float = 0.015,
        clip_hid: float = 0.010,
        clip_out: float = 0.025,
        clip_gate: float = 0.040,
        dgn_coupling: float = 0.15,
        dgn_decay: float = 0.80,
    ) -> None:
        if gating_arch not in R9D_ARCHITECTURES:
            raise ValueError(f"Unknown gating_arch: {gating_arch}. Must be one of {R9D_ARCHITECTURES}")
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"Unknown control_mode: {control_mode}. Must be one of {CONTROL_MODES}")
        if config.neuron != "alif" or config.hidden != 48 or config.fan_in != 8:
            raise ValueError("SCAR-X requires ALIF with hidden=48, fan_in=8")

        config.validate()
        self.config = config
        self.gating_arch = gating_arch
        self.control_mode = control_mode
        self.decay_mode = decay_mode
        self.reset_mode = reset_mode

        self.lr_hid = float(lr_hid)
        self.lr_out = float(lr_out)
        self.lr_gate = float(lr_gate)
        self.clip_hid = float(clip_hid)
        self.clip_out = float(clip_out)
        self.clip_gate = float(clip_gate)

        # DGN dynamic conductance parameters (Bai et al., arXiv:2509.03281)
        self.dgn_coupling = float(dgn_coupling)
        self.dgn_decay = float(dgn_decay)

        self.core = SpikingCore(config)
        h = config.hidden
        d = config.inputs
        k = config.fan_in
        d_tot = d + k

        # Eligibility tensors
        self.e_v = np.zeros((h, d_tot), dtype=np.float32)
        self.e_a = np.zeros((h, d_tot), dtype=np.float32)
        self.e_f = np.zeros((h, d_tot), dtype=np.float32)

        # Bug A fix: Dedicated 2D gate eligibility tensor e_gate in R^{H x K}
        self.e_gate = np.zeros((h, d), dtype=np.float32)

        # Candidate A: Learned channel gate logit s_k in R^d -> g_k = sigmoid(s_k)
        # Initialize at 1.5 -> sigmoid(1.5) approx 0.82 (open by default)
        self.channel_gate_logits = np.full(d, 1.5, dtype=np.float32)

        # Candidate B: Dynamic conductance state per neuron
        self.g_dyn = np.zeros(h, dtype=np.float32)

        # Capacity matched extra weights (for capacity_matched_control baseline)
        rng_cap = np.random.default_rng(config.seed + 777)
        self.cap_weights = rng_cap.normal(0, 0.05, (h, 8)).astype(np.float32)
        self.cap_state = np.zeros(8, dtype=np.float32)

        # Decay vector
        self.gamma_e_vec = np.zeros((h, 1), dtype=np.float32)
        if decay_mode == "uniform_fast":
            self.gamma_e_vec.fill(0.25)
        elif decay_mode == "uniform_slow":
            self.gamma_e_vec.fill(0.92)
        elif decay_mode == "heterogeneous":
            self.gamma_e_vec[:] = np.linspace(0.25, 0.92, h, dtype=np.float32)[:, None]
        else:
            self.gamma_e_vec.fill(0.25)

        # Synaptic momentum and energy normalizers
        self.syn_trace = np.zeros((h, d_tot), dtype=np.float32)
        self.syn_energy = np.full((h, d_tot), 1e-4, dtype=np.float32)

        # Gate momentum and energy normalizers
        self.gate_trace = np.zeros(d, dtype=np.float32)
        self.gate_energy = np.full(d, 1e-4, dtype=np.float32)

        # Running standardization statistics
        self.f_mean = np.zeros(h, dtype=np.float32)
        self.f_var = np.full(h, 0.01, dtype=np.float32)
        self.calibrated = False

        # Initial weights for causal revert testing
        self.w_in_0 = self.core.w_in.copy()
        self.w_rec_0 = self.core.w_rec.copy()

        # Deterministic RNG for matched signal shuffling
        self.signal_rng = np.random.default_rng(config.seed + 920261112)

        self.episodes_learned = 0
        self.hidden_updates = 0
        self.last_gate_vector = np.ones(d, dtype=np.float32)
        self.last_gate_step = np.zeros(d, dtype=np.float32)

    def calibrate(self, sequences: np.ndarray) -> None:
        seqs = np.asarray(sequences, dtype=np.float32)
        for seq in seqs:
            self.reset_episode()
            for row in seq:
                self.step(row, track=False)
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
        self.e_gate.fill(0.0)
        self.g_dyn.fill(0.0)
        self.cap_state.fill(0.0)

    def compute_input_gate(self, row: np.ndarray) -> np.ndarray:
        """Computes effective sensory input vector x_eff."""
        c = self.config
        d = c.inputs
        arch = self.gating_arch

        if arch == "learned_channel_eprop":
            # Candidate A: g_k = sigmoid(s_k)
            clipped_s = np.clip(self.channel_gate_logits, -4.0, 4.0)
            g_vec = 1.0 / (1.0 + np.exp(-clipped_s))
            self.last_gate_vector = g_vec
            return row * g_vec
        else:
            self.last_gate_vector = np.ones(d, dtype=np.float32)
            return row.copy()

    def step(self, x: np.ndarray, track: bool = True) -> np.ndarray:
        c = self.config
        row = np.asarray(x, dtype=np.float32)
        if row.shape != (c.inputs,) or not np.all(np.isfinite(row)):
            raise ValueError("Invalid input vector")

        # 1. Compute effective input
        x_eff = self.compute_input_gate(row)

        old_z = self.core.spikes
        theta = c.threshold + c.adaptation_strength * self.core.adaptation

        i_in = self.core.w_in @ x_eff
        i_rec = np.sum(self.core.w_rec * old_z[self.core.indices], axis=1)

        # Candidate B: DGN dynamic conductance shunting (Bai et al., 2025)
        if self.gating_arch == "dgn_adaptive_conductance":
            # Conductance evolution driven by input energy
            input_norm = float(np.sum(np.abs(row)))
            self.g_dyn = self.dgn_decay * self.g_dyn + (1.0 - self.dgn_decay) * (self.dgn_coupling * input_norm)
            # Effective leak conductance: higher input energy -> stronger leak shunt
            eff_leak = np.clip(c.leak - self.g_dyn, 0.10, 0.95).astype(np.float32)
            u_carry = eff_leak * self.core.voltage + i_rec
        else:
            u_carry = c.leak * self.core.voltage + i_rec

        # Capacity matched baseline auxiliary current
        if self.gating_arch == "capacity_matched_control":
            self.cap_state = 0.85 * self.cap_state + 0.15 * row
            i_in += 0.02 * (self.cap_weights @ self.cap_state)

        u = u_carry + i_in
        psi = spike_slope(u - theta)

        if track:
            pre = np.concatenate(
                (np.broadcast_to(x_eff, (c.hidden, c.inputs)), old_z[self.core.indices]),
                axis=1,
            )
            u_e = c.leak * self.e_v + pre
            a_old = self.e_a.copy()
            z_e = psi[:, None] * (u_e - c.adaptation_strength * a_old)

            # Bug A fix: Update gate eligibility tensor e_gate in R^{H x K}
            # Sensitivity: d(u_h)/d(s_k) = w_in[h, k] * x_k * g_k * (1 - g_k)
            if self.gating_arch == "learned_channel_eprop":
                g_k = self.last_gate_vector
                gate_pre = psi[:, None] * self.core.w_in * (row * g_k * (1.0 - g_k))[None, :]
                self.e_gate[:] = 0.85 * self.e_gate + 0.15 * gate_pre

        # Core state updates
        self.core.voltage = u
        self.core.spikes = (self.core.voltage >= theta).astype(np.float32)
        self.core.voltage -= self.core.spikes * theta
        self.core.adaptation = c.adaptation_decay * self.core.adaptation + self.core.spikes
        self.core.features = c.feature_decay * self.core.features + (1.0 - c.feature_decay) * self.core.spikes

        if track:
            # Stopped-reset vs own-reset
            if self.reset_mode == "stopped_reset":
                self.e_v[:] = u_e
            else:
                self.e_v[:] = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old

            self.e_a[:] = c.adaptation_decay * a_old + z_e

            # Bug F fix: R9b exact recurrent gating baseline
            z_e_filtered = z_e.copy()
            if self.gating_arch == "r9b_recurrent_gating":
                s_act = float(np.max(np.abs(row)))
                g_rec = 1.0 if s_act > 0.60 else 0.15
                z_e_filtered[:, c.inputs:] *= g_rec

            self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e_filtered

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
            raise ValueError("Model must be calibrated before learning")
        seq = np.asarray(sequence, dtype=np.float32)

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

        # Pre-update credit signal delta in R^H before modifying readout weights
        pre_update_signal = (self.core.w_out.T @ err) * (np.abs(raw) < 5.0) / scale

        # Readout gradient (updated only if not frozen_hidden)
        if self.control_mode != "frozen_hidden":
            grad_out = err[:, None] * f_hat[None, :]
            self.core.w_out += np.clip(self.lr_out * grad_out, -self.clip_out, self.clip_out)
            self.core.bias += np.clip(self.lr_out * err, -self.clip_out, self.clip_out)
            np.clip(self.core.w_out, -3.5, 3.5, out=self.core.w_out)
            np.clip(self.core.bias, -3.5, 3.5, out=self.core.bias)

        # Bug C fix: Unified control mode handling
        if self.control_mode not in ("frozen_hidden", "readout_only"):
            if self.control_mode == "learned":
                credit_signal = pre_update_signal
            elif self.control_mode == "shuffled_signal":
                perm = self.signal_rng.permutation(self.config.hidden)
                credit_signal = pre_update_signal[perm]
            elif self.control_mode == "zero_signal":
                credit_signal = np.zeros_like(pre_update_signal)
            else:
                credit_signal = pre_update_signal

            # Hidden synaptic weights update
            direction = credit_signal[:, None] * self.e_f
            self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
            self.syn_energy = 0.99 * self.syn_energy + 0.01 * (direction * direction)
            rms = np.sqrt(self.syn_energy) + 1e-4
            step = np.clip(self.lr_hid * (self.syn_trace / rms), -self.clip_hid, self.clip_hid)

            d = self.config.inputs
            self.core.w_in += step[:, :d]
            self.core.w_rec += step[:, d:]
            np.clip(self.core.w_in, -3.0, 3.0, out=self.core.w_in)
            np.clip(self.core.w_rec, -3.0, 3.0, out=self.core.w_rec)
            self.hidden_updates += 1

            # Bug A fix: Corrected e-prop gradient for channel gating logit s_k
            # grad_k = sum_j (credit_signal_j * e_gate[j, k])
            if self.gating_arch == "learned_channel_eprop":
                gate_grad = np.sum(credit_signal[:, None] * self.e_gate, axis=0)  # (inputs,)
                self.gate_trace = 0.85 * self.gate_trace + 0.15 * gate_grad
                self.gate_energy = 0.95 * self.gate_energy + 0.05 * (gate_grad * gate_grad)
                gate_rms = np.sqrt(self.gate_energy) + 1e-4
                gate_step = np.clip(self.lr_gate * (self.gate_trace / gate_rms), -self.clip_gate, self.clip_gate)
                self.last_gate_step = gate_step.copy()
                self.channel_gate_logits += gate_step
                np.clip(self.channel_gate_logits, -3.5, 3.5, out=self.channel_gate_logits)

        self.episodes_learned += 1
        return loss

    def revert_hidden(self) -> None:
        """Causal intervention: strictly reverts hidden synaptic weights to initialization."""
        self.core.w_in[:] = self.w_in_0
        self.core.w_rec[:] = self.w_rec_0

    def features(self, sequence: np.ndarray) -> np.ndarray:
        self.reset_episode()
        for row in sequence:
            self.step(row, track=False)
        return self.core.features.copy()

    def predict(self, sequence: np.ndarray) -> int:
        f = self.features(sequence)
        p = self.probabilities_from_features(f)
        return int(np.argmax(p))

    def predict_many(self, sequences: np.ndarray) -> np.ndarray:
        return np.array([self.predict(seq) for seq in sequences], dtype=np.int32)

    def clone(self) -> VerifiedNeuralGatingALIF:
        return copy.deepcopy(self)


def get_current_process_rss() -> int:
    """Bug E fix: returns real live RSS bytes of current process."""
    try:
        proc = psutil.Process(os.getpid())
        return int(proc.memory_info().rss)
    except Exception:
        return -1
