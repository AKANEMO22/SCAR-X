"""SCAR-X Round 13.1: Repaired Modular ALIF & Direct Spike Router.

Fixes all Phase 0 defects:
- Defect A: Direct tick-by-tick spike-to-spike routing (no symbol ID bridge).
- Defect B: Seed-dependent, independent spatio-temporal symbol patterns for X, Y, Z.
- Defect C: Forward pass execution in Frozen Hidden control.
- Defect D: Fair contextual awareness in monolithic SNN baseline.
- Defect E: Exact sample-level ground-truth tracking for wrong-memory control.
- Defect F: Complete mid-episode neural state serialization (voltage, adaptation, spikes, features, traces, RNG).
"""
from typing import Optional, Dict, Any, Tuple
import gc
import numpy as np

from scarx.neural.core import Config, SpikingCore
from scarx.evaluation.readout_eligibility import spike_slope


def generate_independent_symbol_patterns(
    seed: int,
    num_symbols: int = 8,
    seq_len: int = 20,
    dim: int = 8,
) -> np.ndarray:
    """Generate seed-dependent, genuinely independent spatio-temporal burst signatures."""
    rng = np.random.default_rng(seed)
    patterns = np.zeros((num_symbols, seq_len, dim), dtype=np.float32)

    for s in range(num_symbols):
        # Sample unique active channels per symbol based on seed
        n_active = int(rng.integers(2, 5))
        active_channels = rng.choice(dim, size=min(n_active, dim), replace=False)

        for ch in active_channels:
            burst_start = int(rng.integers(2, 6))
            burst_len = int(rng.integers(3, 7))
            amp = float(rng.uniform(0.7, 1.3))
            patterns[s, burst_start : burst_start + burst_len, ch] = amp

        # Temporal query/context signature
        trig_ch = int(rng.integers(0, dim))
        patterns[s, 15:18, trig_ch] += float(rng.uniform(0.3, 0.6))

    return patterns


class RepairedModularALIFModule:
    """Modular ALIF SNN module with complete mid-episode state persistence and spike interfaces."""

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
        self.rng = np.random.default_rng(seed)

        self.config = Config(inputs=inputs, outputs=outputs, hidden=hidden, fan_in=fan_in, seed=seed)
        self.core = SpikingCore(self.config)

        # Eligibility traces & local learning state
        h = hidden
        d_tot = inputs + fan_in
        self.e_v = np.zeros((h, d_tot), dtype=np.float32)
        self.e_a = np.zeros((h, d_tot), dtype=np.float32)
        self.e_f = np.zeros((h, d_tot), dtype=np.float32)

        self.gamma_e_vec = np.full((h, 1), self.config.feature_decay, dtype=np.float32)

        # Second-order momentum (Adam-like updates)
        self.syn_trace = np.zeros((h, d_tot), dtype=np.float32)
        self.syn_energy = np.zeros((h, d_tot), dtype=np.float32)

        # Running feature normalization
        self.f_mean = np.zeros(h, dtype=np.float32)
        self.f_var = np.ones(h, dtype=np.float32)
        self.calibrated = False

        self.total_episodes = 0
        self.total_ticks = 0
        self.is_unloaded = False

    def calibrate(self, sequences: np.ndarray) -> None:
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

        self.core.step(row, track=False)

        if track:
            self.e_v[:] = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
            self.e_a[:] = c.adaptation_decay * a_old + z_e
            self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e

        self.total_ticks += 1
        return self.core.spikes

    def forward_sequence(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def get_complete_engram(self) -> Dict[str, Any]:
        """Extract complete mid-episode neural engram state (Defect F fix)."""
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
            # Static & Plastic Weights
            "w_in": self.core.w_in.copy().tolist(),
            "w_rec": self.core.w_rec.copy().tolist(),
            "w_out": self.core.w_out.copy().tolist(),
            "bias": self.core.bias.copy().tolist(),
            # Running Feature Stats
            "f_mean": self.f_mean.copy().tolist(),
            "f_var": self.f_var.copy().tolist(),
            # Momentum Traces
            "syn_trace": self.syn_trace.copy().tolist(),
            "syn_energy": self.syn_energy.copy().tolist(),
            # Instantaneous Dynamical States (Mid-Episode resume support)
            "voltage": self.core.voltage.copy().tolist(),
            "adaptation": self.core.adaptation.copy().tolist(),
            "spikes": self.core.spikes.copy().tolist(),
            "features": self.core.features.copy().tolist(),
            "e_v": self.e_v.copy().tolist(),
            "e_a": self.e_a.copy().tolist(),
            "e_f": self.e_f.copy().tolist(),
            # Metadata & Counters
            "total_episodes": self.total_episodes,
            "total_ticks": self.total_ticks,
            "calibrated": self.calibrated,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_complete_engram(self, engram: Dict[str, Any]) -> None:
        """Restore complete mid-episode state with bit-level exactness."""
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

        # Restore dynamical state
        if "voltage" in engram:
            self.core.voltage[:] = np.array(engram["voltage"], dtype=np.float32)
            self.core.adaptation[:] = np.array(engram["adaptation"], dtype=np.float32)
            self.core.spikes[:] = np.array(engram["spikes"], dtype=np.float32)
            self.core.features[:] = np.array(engram["features"], dtype=np.float32)
            self.e_v[:] = np.array(engram["e_v"], dtype=np.float32)
            self.e_a[:] = np.array(engram["e_a"], dtype=np.float32)
            self.e_f[:] = np.array(engram["e_f"], dtype=np.float32)

        if "rng_state" in engram:
            self.rng.bit_generator.state = engram["rng_state"]

        self.total_episodes = engram.get("total_episodes", 0)
        self.total_ticks = engram.get("total_ticks", 0)
        self.calibrated = engram.get("calibrated", True)
        self.is_unloaded = False

    def unload(self) -> None:
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

    get_engram = get_complete_engram
    load_engram = load_complete_engram


class DirectSpikeRouter:
    """True continuous tick-by-tick neural spike router between SNN populations (Defect A fix)."""

    def __init__(self, in_features: int = 48, out_features: int = 8, seed: int = 42):
        self.in_features = in_features
        self.out_features = out_features
        rng = np.random.default_rng(seed)
        # Fixed sparse synaptic projection matrix
        raw_w = rng.normal(0.0, 0.35, (out_features, in_features)).astype(np.float32)
        # Apply sparse mask (fan-in 6 per post-synaptic channel)
        mask = np.zeros_like(raw_w)
        for i in range(out_features):
            active_idx = rng.choice(in_features, size=6, replace=False)
            mask[i, active_idx] = 1.0
        self.w_inter = raw_w * mask

    def route_tick(self, spike_vector: np.ndarray) -> np.ndarray:
        """Project binary spikes from upstream module into post-synaptic current vector."""
        spk = np.asarray(spike_vector, dtype=np.float32)
        return self.w_inter @ spk
