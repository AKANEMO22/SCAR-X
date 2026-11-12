"""SCAR-X Round 8 Research: Continuous Metaplastic Synaptic Cascade (BF-Cascade)
and Multi-Key Associative Context-Gated Readout.

Source: Benna & Fusi (2016), Nature Neuroscience 19, 1697-1708.

Key mechanisms:
1. Two-compartment intra-synaptic diffusive cascade (u1, u2) per synapse.
   u1: fast transmission efficacy that receives online learning updates.
   u2: slow metaplastic reservoir that passively consolidates stable weight components.
2. Passive intra-synaptic diffusion per episode:
   u1 <- u1 - kappa1 * (u1 - u2)
   u2 <- u2 + kappa2 * (u1 - u2)
   No task boundary signals, no oracle snapshots, no replay buffers!
3. Context-Gated Readout for multi-key associative retrieval (2 -> 4 -> 8 keys).
"""
from __future__ import annotations

import io
import json
import numpy as np

from .core import Config, SpikingCore
from ..evaluation.readout_eligibility import spike_slope


class BennaFusiMetaplasticCascade:
    """Task-agnostic continual learning & multi-key associative memory wrapper."""

    state_array_names = (
        "u1_in",
        "u2_in",
        "u1_rec",
        "u2_rec",
        "u1_out",
        "u2_out",
        "bias",
        "e_v",
        "e_a",
        "e_f",
        "syn_trace",
        "syn_energy",
        "f_mean",
        "f_var",
        "gamma_e_vec",
    )

    def __init__(
        self,
        config: Config,
        use_cascade: bool = True,
        use_context_gating: bool = False,
        kappa1: float = 0.004,
        kappa2: float = 0.001,
        gamma_min: float = 0.25,
        gamma_max: float = 0.92,
        sensory_threshold: float = 0.8,
    ) -> None:
        if config.neuron != "alif" or config.hidden != 48 or config.fan_in != 8:
            raise ValueError("Round 8 requires ALIF SpikingCore with hidden=48, fan_in=8")
        config.validate()

        self.config = config
        self.use_cascade = bool(use_cascade)
        self.use_context_gating = bool(use_context_gating)
        self.kappa1 = float(kappa1)
        self.kappa2 = float(kappa2)
        self.sensory_threshold = float(sensory_threshold)

        self.core = SpikingCore(config)
        h = config.hidden
        d = config.inputs
        k = config.fan_in
        o = config.outputs

        # Two-compartment weights: u1 (active transmission) and u2 (slow reservoir)
        self.u1_in = self.core.w_in.copy()
        self.u2_in = self.core.w_in.copy()

        self.u1_rec = self.core.w_rec.copy()
        self.u2_rec = self.core.w_rec.copy()

        self.u1_out = self.core.w_out.copy()
        self.u2_out = self.core.w_out.copy()
        self.bias = self.core.bias.copy()

        # Context gating tensors if enabled (channels 4..5 query cues)
        self.n_context = 2
        self.w_ctx = np.zeros((self.n_context, o, h), dtype=np.float32)

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
            self.e_v[:] = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
            self.e_a[:] = c.adaptation_decay * a_old + z_e

            s_task = float(np.sum(np.abs(row[:6])))
            g_rec = 1.0 if s_task > self.sensory_threshold else 0.15
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
        # Track context cue from query tick (tick len-2 or query channels 4..5)
        context_cue = np.zeros(self.n_context, dtype=np.float32)
        for row in seq:
            self.step(row, track=True)
            if np.sum(np.abs(row[4:6])) > 0.5:
                context_cue = row[4:6].copy()

        f = self.core.features
        self.f_mean = (1.0 - stat_rate) * self.f_mean + stat_rate * f
        self.f_var = (1.0 - stat_rate) * self.f_var + stat_rate * (f - self.f_mean) ** 2
        f_hat, raw, scale = self.standardized_features(f)

        # Compute effective readout weights
        w_eff = self.u1_out.copy()
        if self.use_context_gating:
            w_eff += context_cue[0] * self.w_ctx[0] + context_cue[1] * self.w_ctx[1]

        logits = w_eff @ f_hat + self.bias
        logits = logits - np.max(logits)
        p = np.exp(logits)
        p = p / np.sum(p)
        loss = -float(np.log(max(float(p[int(target)]), 1e-12)))

        err = -p.copy()
        err[int(target)] += 1.0

        # Pre-update own-neuron credit signal
        pre_update_signal = (w_eff.T @ err) * (np.abs(raw) < 5.0) / scale

        # Readout updates
        grad_out = err[:, None] * f_hat[None, :]
        self.u1_out += np.clip(0.04 * grad_out, -0.025, 0.025)
        self.bias += np.clip(0.04 * err, -0.025, 0.025)
        np.clip(self.u1_out, -3.5, 3.5, out=self.u1_out)
        np.clip(self.bias, -3.5, 3.5, out=self.bias)

        if self.use_context_gating:
            for m in range(self.n_context):
                self.w_ctx[m] += np.clip(0.02 * context_cue[m] * grad_out, -0.015, 0.015)
                np.clip(self.w_ctx[m], -2.0, 2.0, out=self.w_ctx[m])

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

        # Passive intra-synaptic diffusion per episode (BF-Cascade)
        if self.use_cascade:
            diff_in = self.u1_in - self.u2_in
            self.u1_in -= self.kappa1 * diff_in
            self.u2_in += self.kappa2 * diff_in

            diff_rec = self.u1_rec - self.u2_rec
            self.u1_rec -= self.kappa1 * diff_rec
            self.u2_rec += self.kappa2 * diff_rec

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
        context_cue = np.zeros(self.n_context, dtype=np.float32)
        for row in seq:
            self.core.step(row, track=False)
            if np.sum(np.abs(row[4:6])) > 0.5:
                context_cue = row[4:6].copy()

        f = self.core.features
        scale = np.maximum(np.sqrt(self.f_var), 0.05).astype(np.float32)
        f_hat = np.clip((f - self.f_mean) / scale, -5.0, 5.0)

        w_eff = self.u1_out.copy()
        if self.use_context_gating:
            w_eff += context_cue[0] * self.w_ctx[0] + context_cue[1] * self.w_ctx[1]

        logits = w_eff @ f_hat + self.bias
        return int(np.argmax(logits))

    def predict_many(self, sequences: np.ndarray) -> np.ndarray:
        return np.array([self.predict(s) for s in sequences], dtype=np.int32)

    def memory_bytes(self) -> int:
        core_mem = self.core.memory_bytes()
        aux_mem = sum(getattr(self, name).nbytes for name in self.state_array_names)
        if self.use_context_gating:
            aux_mem += self.w_ctx.nbytes
        return core_mem + aux_mem
