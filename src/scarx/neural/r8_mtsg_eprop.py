"""SCAR-X Round 8 Research: Multi-Timescale Sensory-Gated e-prop (MTSG e-prop).

Purely local hidden + readout synaptic plasticity operating on the unchanged
1-compartment recurrent ALIF SpikingCore (H=48, fan_in=8, hard binary spikes).

Key mechanisms:
1. Heterogeneous neuronal eligibility decays gamma_{e,j} in [0.25, 0.92].
2. Sensory event gating G_{sensory}(t) = II(||x_{0:5}(t)||_1 > 0.4) freezing noise in delay ticks.
3. Pre-update own-neuron hidden credit signal with homeostatic feature standardization.
4. Synaptic trace momentum and energy normalization.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np

from .core import Config, SpikingCore
from ..evaluation.readout_eligibility import spike_slope


class MultiTimescaleSensoryGatedEProp:
    """Purely local MTSG e-prop hidden + readout plasticity wrapper around SpikingCore(H=48)."""

    conditions = (
        "frozen_hidden",
        "zero_signal",
        "shuffled_signal",
        "readout_only",
        "r35_baseline",
        "ew_he_legacy",
        "learned_mtsg",
    )

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
    )

    def __init__(
        self,
        config: Config,
        condition: str = "learned_mtsg",
        gamma_min: float = 0.25,
        gamma_max: float = 0.92,
        sensory_threshold: float = 0.8,
    ) -> None:
        if condition not in self.conditions:
            raise ValueError(f"Unsupported condition: {condition}")
        if config.neuron != "alif" or config.hidden != 48 or config.fan_in != 8:
            raise ValueError("Round 8 requires ALIF SpikingCore with hidden=48, fan_in=8")
        config.validate()

        self.config = config
        self.condition = condition
        self.sensory_threshold = float(sensory_threshold)
        self.core = SpikingCore(config)

        h = config.hidden
        d_tot = config.inputs + config.fan_in
        self.e_v = np.zeros((h, d_tot), dtype=np.float32)
        self.e_a = np.zeros((h, d_tot), dtype=np.float32)
        self.e_f = np.zeros((h, d_tot), dtype=np.float32)

        self.syn_trace = np.zeros((h, d_tot), dtype=np.float32)
        self.syn_energy = np.full((h, d_tot), 1e-4, dtype=np.float32)

        self.f_mean = np.zeros(h, dtype=np.float32)
        self.f_var = np.full(h, 0.01, dtype=np.float32)

        self.w_in_0 = self.core.w_in.copy()
        self.w_rec_0 = self.core.w_rec.copy()

        # Multi-timescale eligibility decays across neurons
        if condition in ("r35_baseline",):
            self.gamma_e_vec = np.full((h, 1), 0.92, dtype=np.float32)
        elif condition in ("ew_he_legacy",):
            self.gamma_e_vec = np.full((h, 1), 0.25, dtype=np.float32)
        else:
            self.gamma_e_vec = np.linspace(gamma_min, gamma_max, h, dtype=np.float32)[:, None]

        self.signal_rng = np.random.default_rng(config.seed + 35100000)
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

    def step(self, x: np.ndarray, track: bool = True) -> np.ndarray:
        c = self.config
        row = np.asarray(x, dtype=np.float32)
        if row.shape != (c.inputs,) or not np.isfinite(row).all():
            raise ValueError("Invalid input vector")

        is_hidden_active = self.condition not in ("frozen_hidden", "readout_only")

        if track and is_hidden_active:
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

        if track and is_hidden_active:
            self.e_v[:] = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
            self.e_a[:] = c.adaptation_decay * a_old + z_e

            # Sensory event gating: dampen recurrent noise while allowing input synapses to learn distractor suppression
            if self.condition == "learned_mtsg":
                s_task = float(np.sum(np.abs(row[:6])))
                g_rec = 1.0 if s_task > self.sensory_threshold else 0.15
                z_e_gated = z_e.copy()
                z_e_gated[:, c.inputs:] *= g_rec
                self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e_gated
            elif self.condition == "r35_baseline":
                self.e_f[:] = 0.92 * self.e_f + 0.08 * z_e
            else:
                self.e_f[:] = 0.25 * self.e_f + 0.75 * z_e

        self.total_ticks += 1
        return self.core.features

    def standardized_features(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = np.maximum(np.sqrt(self.f_var), 0.05).astype(np.float32)
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
        if not 0 <= int(target) < self.config.outputs:
            raise ValueError("Invalid target index")

        self.reset_episode()
        for row in seq:
            self.step(row, track=True)

        f = self.core.features
        self.f_mean = (1.0 - stat_rate) * self.f_mean + stat_rate * f
        self.f_var = (1.0 - stat_rate) * self.f_var + stat_rate * (f - self.f_mean) ** 2
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

        # Readout update
        grad_out = err[:, None] * f_hat[None, :]
        self.core.w_out += np.clip(0.04 * grad_out, -0.025, 0.025)
        self.core.bias += np.clip(0.04 * err, -0.025, 0.025)
        np.clip(self.core.w_out, -3.5, 3.5, out=self.core.w_out)
        np.clip(self.core.bias, -3.5, 3.5, out=self.core.bias)

        perm = self.signal_rng.permutation(self.config.hidden)

        # Hidden update
        if self.condition not in ("frozen_hidden", "zero_signal", "readout_only"):
            signal = pre_update_signal
            if self.condition == "shuffled_signal":
                signal = signal[perm]
            direction = signal[:, None] * self.e_f

            self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
            self.syn_energy = 0.99 * self.syn_energy + 0.01 * (direction * direction)
            rms = np.sqrt(self.syn_energy) + 1e-4
            norm_step = self.syn_trace / rms

            lr_hid = 0.002 if self.condition == "r35_baseline" else 0.008
            clip_hid = 0.002 if self.condition == "r35_baseline" else 0.010
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

    def features(self, sequence: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence, dtype=np.float32)
        self.core.reset_state()
        for row in seq:
            self.core.step(row, track=False)
        return self.core.features.copy()

    def features_many(self, sequences: np.ndarray) -> np.ndarray:
        return np.stack([self.features(seq) for seq in sequences])

    def predict(self, sequence: np.ndarray) -> int:
        f = self.features(sequence)
        return int(np.argmax(self.probabilities_from_features(f)))

    def predict_many(self, sequences: np.ndarray) -> np.ndarray:
        f_mat = self.features_many(sequences)
        scale = np.maximum(np.sqrt(self.f_var), 0.05).astype(np.float32)
        f_hat = np.clip((f_mat - self.f_mean) / scale, -5.0, 5.0)
        return np.argmax(f_hat @ self.core.w_out.T + self.core.bias, axis=1).astype(np.int32)

    def revert_hidden_weights(self, revert_in: bool = True, revert_rec: bool = True) -> None:
        """Causal intervention: resets hidden weights to initial state while keeping learned readout."""
        if revert_in:
            self.core.w_in[:] = self.w_in_0
        if revert_rec:
            self.core.w_rec[:] = self.w_rec_0

    def memory_bytes(self) -> int:
        return self.core.memory_bytes() + sum(getattr(self, name).nbytes for name in self.state_array_names)

    def dumps(self) -> bytes:
        core_bytes = self.core.dumps()
        state_buf = io.BytesIO()
        np.savez(state_buf, **{name: getattr(self, name) for name in self.state_array_names})
        state_bytes = state_buf.getvalue()
        meta = {
            "format": "scarx-r8-mtsg-v1",
            "condition": self.condition,
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
    def loads(cls, data: bytes) -> MultiTimescaleSensoryGatedEProp:
        zip_buf = io.BytesIO(data)
        with zipfile.ZipFile(zip_buf, "r") as zf:
            core_bytes = zf.read("core.bin")
            state_bytes = zf.read("state.npz")
            meta = json.loads(zf.read("meta.json").decode("utf-8"))
        core = SpikingCore.loads(core_bytes)
        inst = cls(core.config, condition=meta["condition"])
        inst.core = core
        state_npz = np.load(io.BytesIO(state_bytes))
        for name in cls.state_array_names:
            getattr(inst, name)[:] = state_npz[name]
        inst.calibrated = bool(meta["calibrated"])
        inst.episodes_learned = int(meta["episodes_learned"])
        inst.hidden_updates = int(meta["hidden_updates"])
        inst.total_ticks = int(meta["total_ticks"])
        return inst
