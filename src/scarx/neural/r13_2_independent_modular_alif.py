"""SCAR-X Round 13.2: Truly Independent Modular ALIF & Fair Neural Interfaces.

Core Principles:
1. Module A is trained ONLY on Task A (X -> Y).
2. Module B is trained ONLY on Task B (Y -> Z).
3. Module B NEVER receives X, Mod_A(X), or Task C labels during training.
4. Routing Options for Task C (X -> Z):
   - FixedSpikeRouter: Fixed random sparse projection (unsupervised, no C training).
   - LearnedRepresentationAligner: Synaptic alignment trained ONLY on mapping
     Module A's internal activity to sensory pattern of Y. NO Z labels, NO C loss!
   - SymbolicRouter: Oracle ceiling control (argmax(A) -> pattern(Y) -> B).
5. Dynamic parameter counting directly from array shapes.
6. Bit-level state persistence to SSD.
"""
from typing import Dict, Any, Tuple, Optional, List
import gc
import json
import time
from pathlib import Path
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
        n_active = int(rng.integers(2, 5))
        active_channels = rng.choice(dim, size=min(n_active, dim), replace=False)

        for ch in active_channels:
            burst_start = int(rng.integers(2, 6))
            burst_len = int(rng.integers(3, 7))
            amp = float(rng.uniform(0.7, 1.3))
            patterns[s, burst_start : burst_start + burst_len, ch] = amp

        # Temporal context signature
        trig_ch = int(rng.integers(0, dim))
        patterns[s, 15:18, trig_ch] += float(rng.uniform(0.3, 0.6))

    return patterns


class IndependentALIFModule:
    """ALIF SNN Module with rigorous parameter introspection and complete state persistence."""

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

        # Traces & local plasticity state
        h = hidden
        d_tot = inputs + fan_in
        self.e_v = np.zeros((h, d_tot), dtype=np.float32)
        self.e_a = np.zeros((h, d_tot), dtype=np.float32)
        self.e_f = np.zeros((h, d_tot), dtype=np.float32)

        self.gamma_e_vec = np.full((h, 1), self.config.feature_decay, dtype=np.float32)
        self.syn_trace = np.zeros((h, d_tot), dtype=np.float32)
        self.syn_energy = np.zeros((h, d_tot), dtype=np.float32)

        self.f_mean = np.zeros(h, dtype=np.float32)
        self.f_var = np.ones(h, dtype=np.float32)
        self.calibrated = False

        self.total_episodes = 0
        self.total_ticks = 0
        self.is_unloaded = False

    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Compute exact parameter counts directly from weight matrices."""
        if self.is_unloaded:
            return {"w_in": 0, "w_rec": 0, "w_out": 0, "bias": 0, "trainable_total": 0, "nonzero_synapses": 0}
        w_in_count = self.core.w_in.size
        w_rec_count = self.core.w_rec.size
        w_out_count = self.core.w_out.size
        bias_count = self.core.bias.size
        trainable = w_in_count + w_rec_count + w_out_count + bias_count

        nonzero = int(
            np.count_nonzero(self.core.w_in)
            + np.count_nonzero(self.core.w_rec)
            + np.count_nonzero(self.core.w_out)
            + np.count_nonzero(self.core.bias)
        )
        return {
            "w_in": w_in_count,
            "w_rec": w_rec_count,
            "w_out": w_out_count,
            "bias": bias_count,
            "trainable_total": trainable,
            "nonzero_synapses": nonzero,
        }

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
            self.f_var = 0.95 * self.f_var + 0.05 * ((f - self.f_mean) ** 2)
        self.f_var = np.maximum(self.f_var, 1e-4)
        self.calibrated = True

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
            pre = np.concatenate((np.broadcast_to(row, (c.hidden, c.inputs)), old_z[self.core.indices]), axis=1)

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

    def standardized_features(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        f_centered = f - self.f_mean
        scale = np.sqrt(self.f_var) + 1e-4
        f_hat = f_centered / scale
        return f_hat, f_centered, float(np.mean(scale))

    def forward_sequence(self, sequence: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        if self.is_unloaded:
            raise RuntimeError(f"Cannot run forward on unloaded module {self.module_name}")
        self.reset_episode()
        spike_record = []
        for row in sequence:
            spk = self.step(row, track=False)
            spike_record.append(spk)

        f = self.core.features
        f_hat, _, _ = self.standardized_features(f)
        logits = self.core.w_out @ f_hat + self.core.bias
        shifted = logits - np.max(logits)
        exp_logits = np.exp(np.clip(shifted, -20.0, 20.0))
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)
        return probs, spike_record

    def learn_episode(self, sequence: np.ndarray, target: int) -> float:
        if self.is_unloaded:
            raise RuntimeError(f"Cannot learn on unloaded module {self.module_name}")
        self.reset_episode()
        for row in sequence:
            self.step(row, track=True)

        f = self.core.features
        f_hat, raw, scale = self.standardized_features(f)
        logits = self.core.w_out @ f_hat + self.core.bias
        shifted = logits - np.max(logits)
        exp_logits = np.exp(np.clip(shifted, -20.0, 20.0))
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)
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

        # Hidden synaptic update
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
        """Extract complete neural engram state for SSD persistence."""
        if self.is_unloaded:
            raise RuntimeError(f"Cannot extract engram from unloaded module {self.module_name}")
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
            "voltage": self.core.voltage.copy().tolist(),
            "adaptation": self.core.adaptation.copy().tolist(),
            "spikes": self.core.spikes.copy().tolist(),
            "features": self.core.features.copy().tolist(),
            "e_v": self.e_v.copy().tolist(),
            "e_a": self.e_a.copy().tolist(),
            "e_f": self.e_f.copy().tolist(),
            "total_episodes": self.total_episodes,
            "total_ticks": self.total_ticks,
            "calibrated": self.calibrated,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_complete_engram(self, engram: Dict[str, Any]) -> None:
        """Restore complete state with exact bit-level reproducibility."""
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


# =============================================================================
# ROUTING & INTERFACE MECHANISMS (COMPLETELY FREE OF TASK C OR Z-LABELS)
# =============================================================================

class FixedSpikeRouter:
    """Fixed sparse random projection router (No training, zero-shot signal forwarding)."""

    def __init__(self, in_features: int = 48, out_features: int = 8, seed: int = 42):
        self.in_features = in_features
        self.out_features = out_features
        rng = np.random.default_rng(seed)
        raw_w = rng.normal(0.0, 0.35, (out_features, in_features)).astype(np.float32)
        mask = np.zeros_like(raw_w)
        for i in range(out_features):
            active_idx = rng.choice(in_features, size=min(6, in_features), replace=False)
            mask[i, active_idx] = 1.0
        self.w_inter = raw_w * mask

    @property
    def total_parameters(self) -> int:
        return self.w_inter.size

    @property
    def nonzero_parameters(self) -> int:
        return int(np.count_nonzero(self.w_inter))

    def route_tick(self, spike_vector: np.ndarray) -> np.ndarray:
        return self.w_inter @ np.asarray(spike_vector, dtype=np.float32)


class LearnedRepresentationAligner:
    """Learned neural interface trained ONLY on aligning Module A's spike output to Y sensory patterns.
    
    STRICT SCIENTIFIC CONSTRAINT:
    - This interface NEVER sees Module B.
    - This interface NEVER sees target Z.
    - This interface NEVER receives Task C loss.
    - Loss = MSE(W_align * s_A(t) + bias, sensory_pattern_Y(t)).
    """

    def __init__(self, in_features: int = 48, out_features: int = 8, seed: int = 42, lr: float = 0.01):
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.1, (out_features, in_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

    @property
    def total_parameters(self) -> int:
        return self.w.size + self.bias.size

    @property
    def nonzero_parameters(self) -> int:
        return int(np.count_nonzero(self.w) + np.count_nonzero(self.bias))

    def forward_tick(self, spike_vector: np.ndarray) -> np.ndarray:
        return self.w @ np.asarray(spike_vector, dtype=np.float32) + self.bias

    def train_alignment_pair(self, spike_sequence: List[np.ndarray], target_y_pattern: np.ndarray) -> float:
        """Update alignment weights using ONLY sensory pattern of Y as target."""
        total_loss = 0.0
        t_steps = min(len(spike_sequence), len(target_y_pattern))
        for t in range(t_steps):
            spk = np.asarray(spike_sequence[t], dtype=np.float32)
            target = np.asarray(target_y_pattern[t], dtype=np.float32)

            pred = self.w @ spk + self.bias
            err = target - pred
            loss = float(np.mean(err ** 2))
            total_loss += loss

            # Gradient update on linear projection
            grad_w = err[:, None] * spk[None, :]
            self.w += self.lr * grad_w
            self.bias += self.lr * err

            np.clip(self.w, -2.0, 2.0, out=self.w)
            np.clip(self.bias, -2.0, 2.0, out=self.bias)

        return total_loss / max(t_steps, 1)


class SymbolicRouter:
    """Oracle symbolic ceiling control: argmax(Mod_A) -> lookup canon pattern(Y) -> Mod_B."""

    def __init__(self, patterns_y: np.ndarray):
        self.patterns_y = patterns_y

    @property
    def total_parameters(self) -> int:
        return 0  # Symbolic zero-parameter lookup

    def route_symbolic(self, probs_a: np.ndarray) -> np.ndarray:
        y_hat = int(np.argmax(probs_a))
        return self.patterns_y[y_hat]
