"""SCAR-X Round 13: Modular ALIF Engram Architecture.

Implements:
- Modular ALIF SNN with H=48, D=8, K=8, Fan-in=8.
- Stable global neuron IDs.
- Exact SCAR-X own-reset online learning algorithm.
- Full engram serialization (weights, topology, normalization, traces, metadata).
- Memory deallocation via unload() for SSD paging.
- Neural spike interfaces for modular composition.
"""
from typing import Optional, Dict, Any, Tuple
import gc
import numpy as np
from scarx.neural.core import Config, SpikingCore
from scarx.evaluation.readout_eligibility import spike_slope


class ModularALIFModule:
    """Independent ALIF SNN module forming part of a modular cognitive architecture."""

    def __init__(
        self,
        module_name: str,
        start_neuron_id: int = 0,
        inputs: int = 8,
        hidden: int = 48,
        outputs: int = 8,
        fan_in: int = 8,
        seed: int = 140011,
    ):
        self.module_name = module_name
        self.start_neuron_id = start_neuron_id
        self.neuron_ids = np.arange(start_neuron_id, start_neuron_id + hidden, dtype=np.int32)
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.fan_in = fan_in
        self.seed = seed

        self.config = Config(inputs=inputs, outputs=outputs, hidden=hidden, fan_in=fan_in, seed=seed)
        self.core = SpikingCore(self.config)

        # Eligibility traces & local learning state
        h = hidden
        d_tot = inputs + fan_in
        self.e_v = np.zeros((h, d_tot), dtype=np.float32)
        self.e_a = np.zeros((h, d_tot), dtype=np.float32)
        self.e_f = np.zeros((h, d_tot), dtype=np.float32)

        self.gamma_e_vec = np.full((h, 1), self.config.feature_decay, dtype=np.float32)

        # Second-order normalized momentum (Adam-like online updates)
        self.syn_trace = np.zeros((h, d_tot), dtype=np.float32)
        self.syn_energy = np.zeros((h, d_tot), dtype=np.float32)

        # Feature normalization
        self.f_mean = np.zeros(h, dtype=np.float32)
        self.f_var = np.ones(h, dtype=np.float32)
        self.calibrated = False

        self.total_episodes = 0
        self.total_ticks = 0
        self.is_unloaded = False

    def calibrate(self, sequences: np.ndarray) -> None:
        """Calibrate running mean and variance of hidden features on initial episodes."""
        if self.is_unloaded:
            raise RuntimeError(f"Cannot calibrate unloaded module {self.module_name}")
        seqs = np.asarray(sequences, dtype=np.float32)
        for seq in seqs:
            self.core.reset_state()
            for row in seq:
                self.core.step(row, track=False)
            f = self.core.features
            self.f_mean = 0.95 * self.f_mean + 0.05 * f
            self.f_var = 0.95 * self.f_var + 0.05 * (f - self.f_mean) ** 2
        self.calibrated = True

    def standardized_features(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = np.maximum(np.sqrt(self.f_var), 0.05).astype(np.float32)
        raw = (f - self.f_mean) / scale
        f_hat = np.clip(raw, -5.0, 5.0).astype(np.float32)
        return f_hat, raw, scale

    def reset_episode(self) -> None:
        if self.is_unloaded:
            raise RuntimeError(f"Cannot reset unloaded module {self.module_name}")
        self.core.reset_state()
        self.e_v.fill(0.0)
        self.e_a.fill(0.0)
        self.e_f.fill(0.0)

    def step(self, x: np.ndarray, track: bool = True) -> np.ndarray:
        """Single time-step execution with ALIF own-reset trace tracking."""
        if self.is_unloaded:
            raise RuntimeError(f"Cannot step unloaded module {self.module_name}")
        c = self.config
        row = np.asarray(x, dtype=np.float32)

        if track:
            old_z = self.core.spikes
            theta = c.threshold + c.adaptation_strength * self.core.adaptation
            i_in = self.core.w_in @ row
            i_rec = np.sum(self.core.w_rec * old_z[self.core.indices], axis=1)
            u = (c.leak * self.core.voltage + i_rec) + i_in
            psi = spike_slope(u - theta)
            pre = np.concatenate(
                (np.broadcast_to(row, (c.hidden, c.inputs)), old_z[self.core.indices]),
                axis=1,
            )

            u_e = c.leak * self.e_v + pre
            a_old = self.e_a.copy()
            z_e = psi[:, None] * (u_e - c.adaptation_strength * a_old)

        # Single unchanged neural core forward
        self.core.step(row, track=False)

        if track:
            self.e_v[:] = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
            self.e_a[:] = c.adaptation_decay * a_old + z_e
            self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e

        self.total_ticks += 1
        return self.core.spikes

    def forward_sequence(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process sequence through module and return (probabilities, hidden_spikes_trajectory)."""
        if self.is_unloaded:
            raise RuntimeError(f"Cannot run forward on unloaded module {self.module_name}")
        self.reset_episode()
        spike_record = []
        for row in sequence:
            spk = self.step(row, track=False)
            spike_record.append(spk.copy())

        f = self.core.features
        f_hat, _, _ = self.standardized_features(f)
        logits = self.core.w_out @ f_hat + self.core.bias
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        return probs, np.array(spike_record, dtype=np.float32)

    def learn_episode(self, sequence: np.ndarray, target: int, stat_rate: float = 0.01) -> float:
        """Online local supervised update on a single episode."""
        if self.is_unloaded:
            raise RuntimeError(f"Cannot learn on unloaded module {self.module_name}")
        if not self.calibrated:
            raise RuntimeError("Module must be calibrated before supervised learning")

        seq = np.asarray(sequence, dtype=np.float32)
        self.reset_episode()
        for row in seq:
            self.step(row, track=True)

        f = self.core.features
        self.f_mean = (1.0 - stat_rate) * self.f_mean + stat_rate * f
        self.f_var = (1.0 - stat_rate) * self.f_var + stat_rate * (f - self.f_mean) ** 2
        f_hat, raw, scale = self.standardized_features(f)

        logits = self.core.w_out @ f_hat + self.core.bias
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        loss = -float(np.log(max(float(probs[int(target)]), 1e-12)))

        err = -probs.copy()
        err[int(target)] += 1.0

        # Credit signal
        signal = (self.core.w_out.T @ err) * (np.abs(raw) < 5.0) / scale

        # Readout update
        grad_out = err[:, None] * f_hat[None, :]
        self.core.w_out += np.clip(0.04 * grad_out, -0.025, 0.025)
        self.core.bias += np.clip(0.04 * err, -0.025, 0.025)
        np.clip(self.core.w_out, -3.5, 3.5, out=self.core.w_out)
        np.clip(self.core.bias, -3.5, 3.5, out=self.core.bias)

        # Hidden update
        direction = signal[:, None] * self.e_f
        self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
        self.syn_energy = 0.99 * self.syn_energy + 0.01 * (direction * direction)
        rms = np.sqrt(self.syn_energy) + 1e-4
        norm_step = self.syn_trace / rms

        lr_hid = 0.008
        clip_hid = 0.010
        step = np.clip(lr_hid * norm_step, -clip_hid, clip_hid)

        d = self.inputs
        self.core.w_in += step[:, :d]
        self.core.w_rec += step[:, d:]
        np.clip(self.core.w_in, -3.0, 3.0, out=self.core.w_in)
        np.clip(self.core.w_rec, -3.0, 3.0, out=self.core.w_rec)

        self.total_episodes += 1
        return loss

    def get_engram(self) -> Dict[str, Any]:
        """Extract full serializable neural engram representation."""
        if self.is_unloaded:
            raise RuntimeError(f"Cannot get engram from unloaded module {self.module_name}")
        return {
            "module_name": self.module_name,
            "neuron_ids": self.neuron_ids.tolist(),
            "inputs": self.inputs,
            "hidden": self.hidden,
            "outputs": self.outputs,
            "fan_in": self.fan_in,
            "seed": self.seed,
            "indices": self.core.indices.tolist(),
            "w_in": self.core.w_in.copy().tolist(),
            "w_rec": self.core.w_rec.copy().tolist(),
            "w_out": self.core.w_out.copy().tolist(),
            "bias": self.core.bias.copy().tolist(),
            "f_mean": self.f_mean.copy().tolist(),
            "f_var": self.f_var.copy().tolist(),
            "syn_trace": self.syn_trace.copy().tolist(),
            "syn_energy": self.syn_energy.copy().tolist(),
            "total_episodes": self.total_episodes,
            "calibrated": self.calibrated,
        }

    def load_engram(self, engram: Dict[str, Any]) -> None:
        """Restore neural engram with bit-level exactness."""
        if self.is_unloaded:
            self.reload_fresh()
        if engram["hidden"] != self.hidden or engram["inputs"] != self.inputs:
            raise ValueError("Incompatible engram topology")

        self.module_name = engram["module_name"]
        self.neuron_ids = np.array(engram["neuron_ids"], dtype=np.int32)
        self.core.indices = np.array(engram["indices"], dtype=np.int64)
        self.core.w_in[:] = np.array(engram["w_in"], dtype=np.float32)
        self.core.w_rec[:] = np.array(engram["w_rec"], dtype=np.float32)
        self.core.w_out[:] = np.array(engram["w_out"], dtype=np.float32)
        self.core.bias[:] = np.array(engram["bias"], dtype=np.float32)

        self.f_mean[:] = np.array(engram["f_mean"], dtype=np.float32)
        self.f_var[:] = np.array(engram["f_var"], dtype=np.float32)
        self.syn_trace[:] = np.array(engram["syn_trace"], dtype=np.float32)
        self.syn_energy[:] = np.array(engram["syn_energy"], dtype=np.float32)

        self.total_episodes = engram.get("total_episodes", 0)
        self.calibrated = engram.get("calibrated", True)
        self.is_unloaded = False

    def unload(self) -> None:
        """Free weight arrays from active RAM."""
        self.is_unloaded = True
        del self.core
        self.core = None
        self.syn_trace = None
        self.syn_energy = None
        self.f_mean = None
        self.f_var = None
        self.e_a = None
        self.e_v = None
        self.e_f = None
        gc.collect()

    def reload_fresh(self) -> None:
        """Re-allocate empty uninitialized structures ready to load engram."""
        self.config = Config(inputs=self.inputs, outputs=self.outputs, hidden=self.hidden, fan_in=self.fan_in, seed=self.seed)
        self.core = SpikingCore(self.config)
        h = self.hidden
        d_tot = self.inputs + self.fan_in
        self.e_v = np.zeros((h, d_tot), dtype=np.float32)
        self.e_a = np.zeros((h, d_tot), dtype=np.float32)
        self.e_f = np.zeros((h, d_tot), dtype=np.float32)
        self.gamma_e_vec = np.full((h, 1), self.config.feature_decay, dtype=np.float32)
        self.syn_trace = np.zeros((h, d_tot), dtype=np.float32)
        self.syn_energy = np.zeros((h, d_tot), dtype=np.float32)
        self.f_mean = np.zeros(h, dtype=np.float32)
        self.f_var = np.ones(h, dtype=np.float32)
        self.is_unloaded = False
