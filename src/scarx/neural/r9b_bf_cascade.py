"""SCAR-X Round 9b Benna-Fusi Metaplastic Cascade with Targeted Consolidation Modes.

Consolidation modes evaluated in R9b:
1. 'none': Unconsolidated baseline (kappa1 = kappa2 = 0)
2. 'full': Standard BF-Cascade across input, recurrent, and readout synapses
3. 'readout_only': Cascade applied exclusively to readout synapses
4. 'hidden_only': Cascade applied exclusively to hidden synapses (input & recurrent)
5. 'selective': Activity-dependent cascade diffusion (inactive synapses consolidate faster)
"""
from __future__ import annotations

import io
import json
import numpy as np

from .core import Config, SpikingCore
from ..evaluation.readout_eligibility import spike_slope

CONSOLIDATION_MODES = ("none", "full", "readout_only", "hidden_only", "selective")


class R9bBFCascade:
    """Metaplastic continual learning cascade with targeted consolidation ablation."""

    def __init__(
        self,
        config: Config,
        consolidation_mode: str = "full",
        kappa1: float = 0.004,
        kappa2: float = 0.001,
        gamma_min: float = 0.25,
        gamma_max: float = 0.92,
        sensory_threshold: float = 0.6,
    ) -> None:
        if consolidation_mode not in CONSOLIDATION_MODES:
            raise ValueError(f"Unknown consolidation_mode: {consolidation_mode}. Must be one of {CONSOLIDATION_MODES}")
        if config.neuron != "alif" or config.hidden != 48 or config.fan_in != 8:
            raise ValueError("SCAR-X R9b requires ALIF SpikingCore with hidden=48, fan_in=8")
        config.validate()

        self.config = config
        self.consolidation_mode = consolidation_mode
        self.kappa1 = float(kappa1)
        self.kappa2 = float(kappa2)
        self.sensory_threshold = float(sensory_threshold)

        self.core = SpikingCore(config)
        h = config.hidden
        d = config.inputs
        k = config.fan_in
        o = config.outputs

        # Two-compartment weights: u1 (active fast transmission) and u2 (slow reservoir)
        self.u1_in = self.core.w_in.copy()
        self.u2_in = self.core.w_in.copy()

        self.u1_rec = self.core.w_rec.copy()
        self.u2_rec = self.core.w_rec.copy()

        self.u1_out = self.core.w_out.copy()
        self.u2_out = self.core.w_out.copy()
        self.bias = self.core.bias.copy()

        # Traces
        d_tot = d + k
        self.e_v = np.zeros((h, d_tot), dtype=np.float32)
        self.e_a = np.zeros((h, d_tot), dtype=np.float32)
        self.e_f = np.zeros((h, d_tot), dtype=np.float32)

        self.syn_trace = np.zeros((h, d_tot), dtype=np.float32)
        self.syn_energy = np.full((h, d_tot), 1e-4, dtype=np.float32)

        self.f_mean = np.zeros(h, dtype=np.float32)
        self.f_var = np.full(h, 0.01, dtype=np.float32)

        self.gamma_e_vec = np.linspace(gamma_min, gamma_max, h, dtype=np.float32)[:, None]
        self.calibrated = False
        self.episodes_learned = 0
        self.hidden_updates = 0

    def sync_active_weights_to_core(self) -> None:
        self.core.w_in[:] = self.u1_in
        self.core.w_rec[:] = self.u1_rec

    def calibrate(self, sequences: np.ndarray) -> None:
        seqs = np.asarray(sequences, dtype=np.float32)
        self.sync_active_weights_to_core()
        for seq in seqs:
            self.core.reset_state()
            for row in seq:
                self.core.step(row, track=False)
            f = self.core.features
            self.f_mean = 0.95 * self.f_mean + 0.05 * f
            self.f_var = 0.95 * self.f_var + 0.05 * (f - self.f_mean) ** 2
        self.calibrated = True

    def reset_episode(self) -> None:
        self.sync_active_weights_to_core()
        self.core.reset_state()
        self.e_v.fill(0.0)
        self.e_a.fill(0.0)
        self.e_f.fill(0.0)

    def step(self, x: np.ndarray, track: bool = True) -> np.ndarray:
        c = self.config
        row = np.asarray(x, dtype=np.float32)

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
            # Baseline subtractive reset
            self.e_v[:] = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
            self.e_a[:] = c.adaptation_decay * a_old + z_e

            s_activity = float(np.max(np.abs(row)))
            g_rec = 1.0 if s_activity > self.sensory_threshold else 0.15
            z_e_gated = z_e.copy()
            z_e_gated[:, c.inputs:] *= g_rec
            self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e_gated

        return self.core.features

    def standardized_features(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = np.maximum(np.sqrt(self.f_var), 0.05).astype(np.float32)
        raw = (f - self.f_mean) / scale
        f_hat = np.clip(raw, -5.0, 5.0).astype(np.float32)
        return f_hat, raw, scale

    def learn_episode(self, sequence: np.ndarray, target: int, stat_rate: float = 0.01) -> float:
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

        w_eff = self.u1_out
        logits = w_eff @ f_hat + self.bias
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p = p / np.sum(p)
        loss = -float(np.log(max(float(p[int(target)]), 1e-12)))

        err = -p.copy()
        err[int(target)] += 1.0

        # Pre-update credit signal
        pre_update_signal = (w_eff.T @ err) * (np.abs(raw) < 5.0) / scale

        # Readout updates
        grad_out = err[:, None] * f_hat[None, :]
        self.u1_out += np.clip(0.04 * grad_out, -0.025, 0.025)
        self.bias += np.clip(0.04 * err, -0.025, 0.025)
        np.clip(self.u1_out, -3.5, 3.5, out=self.u1_out)
        np.clip(self.bias, -3.5, 3.5, out=self.bias)

        # Hidden updates
        direction = pre_update_signal[:, None] * self.e_f
        self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
        self.syn_energy = 0.99 * self.syn_energy + 0.01 * (direction * direction)
        rms = np.sqrt(self.syn_energy) + 1e-4
        norm_step = self.syn_trace / rms

        step = np.clip(0.008 * norm_step, -0.010, 0.010)
        d = self.config.inputs
        self.u1_in += step[:, :d]
        self.u1_rec += step[:, d:]
        np.clip(self.u1_in, -3.0, 3.0, out=self.u1_in)
        np.clip(self.u1_rec, -3.0, 3.0, out=self.u1_rec)

        # Passive intra-synaptic diffusion per episode
        mode = self.consolidation_mode
        if mode != "none":
            if mode == "selective":
                # Synapses with lower plasticity energy consolidate faster
                mod_hid = 1.0 / (1.0 + 20.0 * np.sqrt(self.syn_energy))
                diff_in = self.u1_in - self.u2_in
                self.u1_in -= (self.kappa1 * mod_hid[:, :d]) * diff_in
                self.u2_in += (self.kappa2 * mod_hid[:, :d]) * diff_in

                diff_rec = self.u1_rec - self.u2_rec
                self.u1_rec -= (self.kappa1 * mod_hid[:, d:]) * diff_rec
                self.u2_rec += (self.kappa2 * mod_hid[:, d:]) * diff_rec

                diff_out = self.u1_out - self.u2_out
                self.u1_out -= self.kappa1 * diff_out
                self.u2_out += self.kappa2 * diff_out

            else:
                if mode in ("full", "hidden_only"):
                    diff_in = self.u1_in - self.u2_in
                    self.u1_in -= self.kappa1 * diff_in
                    self.u2_in += self.kappa2 * diff_in

                    diff_rec = self.u1_rec - self.u2_rec
                    self.u1_rec -= self.kappa1 * diff_rec
                    self.u2_rec += self.kappa2 * diff_rec

                if mode in ("full", "readout_only"):
                    diff_out = self.u1_out - self.u2_out
                    self.u1_out -= self.kappa1 * diff_out
                    self.u2_out += self.kappa2 * diff_out

        self.sync_active_weights_to_core()
        self.hidden_updates += 1
        self.episodes_learned += 1
        return loss

    def features(self, sequence: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence, dtype=np.float32)
        self.sync_active_weights_to_core()
        self.core.reset_state()
        for row in seq:
            self.core.step(row, track=False)
        return self.core.features.copy()

    def features_many(self, sequences: np.ndarray) -> np.ndarray:
        return np.stack([self.features(seq) for seq in sequences])

    def predict(self, sequence: np.ndarray) -> int:
        seq = np.asarray(sequence, dtype=np.float32)
        self.sync_active_weights_to_core()
        self.core.reset_state()
        for row in seq:
            self.core.step(row, track=False)
        f_hat, _, _ = self.standardized_features(self.core.features)
        logits = self.u1_out @ f_hat + self.bias
        return int(np.argmax(logits))

    def predict_many(self, sequences: np.ndarray) -> np.ndarray:
        return np.array([self.predict(seq) for seq in sequences], dtype=np.int32)
