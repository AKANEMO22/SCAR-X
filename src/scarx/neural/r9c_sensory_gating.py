"""SCAR-X Round 9c Learned Selective Sensory Gating (LLSS) Architectures.

Implements and ablates input selectivity mechanisms:
Candidates:
- Candidate A: Channel-Adaptive Gating (CAG) / Learned Channel Gating (LCG)
  Per-channel adaptive gate g_k in [0, 1] learned via local reward/credit assignment.
- Candidate B: Predictive Input Novelty (PIN)
  Estimates input prediction error from recurrent state to modulate transmission.
- Candidate C: Learned Inhibitory Gate (LIG)
  4 capacity-matched inhibitory interneurons providing feedforward inhibition to soma.
- Candidate D: State-Conditioned Sensory Selection (SCSS)
  Gating conditioned on neuron somatic state (membrane potential, adaptation, eligibility trace).

Ablations:
- no_gating: g = 1.0 everywhere
- scalar_max_mag: G(t) = max(|x(t)|) (Round 8/9b baseline)
- fixed_channel: static uniform channel scaling
- random_channel: random static channel weights
- learned_channel: Candidate A
- predictive_novelty: Candidate B
- learned_inhibitory: Candidate C
- state_conditioned: Candidate D
- capacity_matched: extra parameters matched without selective gating
"""
from __future__ import annotations

import copy
import numpy as np

from scarx.neural.core import Config, SpikingCore
from scarx.evaluation.readout_eligibility import spike_slope


GATING_ARCHITECTURES = (
    "no_gating",
    "scalar_max_mag",
    "fixed_channel",
    "random_channel",
    "learned_channel",
    "predictive_novelty",
    "learned_inhibitory",
    "state_conditioned",
    "capacity_matched",
)


class SelectiveSensoryALIF:
    """Unified ALIF H48 network with configurable learned sensory gating architectures."""

    def __init__(
        self,
        config: Config,
        gating_arch: str = "learned_channel",
        decay_mode: str = "uniform_fast",
        reset_mode: str = "stopped_reset",
        lr_hid: float = 0.008,
        lr_out: float = 0.04,
        lr_gate: float = 0.02,
        clip_hid: float = 0.010,
        clip_out: float = 0.025,
    ) -> None:
        if gating_arch not in GATING_ARCHITECTURES:
            raise ValueError(f"Unknown gating_arch: {gating_arch}. Must be one of {GATING_ARCHITECTURES}")
        if config.neuron != "alif" or config.hidden != 48 or config.fan_in != 8:
            raise ValueError("SCAR-X requires ALIF with hidden=48, fan_in=8")

        config.validate()
        self.config = config
        self.gating_arch = gating_arch
        self.decay_mode = decay_mode
        self.reset_mode = reset_mode

        self.lr_hid = float(lr_hid)
        self.lr_out = float(lr_out)
        self.lr_gate = float(lr_gate)
        self.clip_hid = float(clip_hid)
        self.clip_out = float(clip_out)

        self.core = SpikingCore(config)
        h = config.hidden
        d = config.inputs
        k = config.fan_in
        d_tot = d + k

        # Eligibility tensors
        self.e_v = np.zeros((h, d_tot), dtype=np.float32)
        self.e_a = np.zeros((h, d_tot), dtype=np.float32)
        self.e_f = np.zeros((h, d_tot), dtype=np.float32)

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

        # Standardization statistics
        self.f_mean = np.zeros(h, dtype=np.float32)
        self.f_var = np.full(h, 0.01, dtype=np.float32)
        self.calibrated = False

        # Initial weights for causal revert testing
        self.w_in_0 = self.core.w_in.copy()
        self.w_rec_0 = self.core.w_rec.copy()

        # Gate parameters per architecture
        rng_gate = np.random.default_rng(config.seed + 888)

        # Candidate A: Learned channel gating logit s_k in R^d -> g_k = sigmoid(s_k)
        # Initialize at 1.5 -> sigmoid(1.5) approx 0.82 (open by default)
        self.channel_gate_logits = np.full(d, 1.5, dtype=np.float32)
        self.channel_gate_elig = np.zeros(d, dtype=np.float32)

        # Random channel weights for ablation
        self.random_channel_weights = rng_gate.uniform(0.3, 1.0, d).astype(np.float32)

        # Candidate B: Predictive novelty predictor (W_pred: d x h, b_pred: d)
        self.w_pred = rng_gate.normal(0, 0.05, (d, h)).astype(np.float32)
        self.b_pred = np.zeros(d, dtype=np.float32)

        # Candidate C: Inhibitory interneurons (4 neurons)
        self.n_inh = 4
        self.w_inh_in = rng_gate.normal(0.2, 0.1, (self.n_inh, d)).astype(np.float32)
        self.w_inh_out = rng_gate.uniform(0.1, 0.3, (h, self.n_inh)).astype(np.float32)
        self.v_inh = np.zeros(self.n_inh, dtype=np.float32)

        # Candidate D: State-conditioned coefficients
        # g_h = sigmoid(beta_0 + beta_v * v - beta_a * a - beta_e * ||e_f||)
        self.scss_beta = np.array([1.5, -0.5, -1.0, -1.5], dtype=np.float32)  # [intercept, v_coeff, a_coeff, e_coeff]

        # Capacity matched extra weights (for capacity_matched baseline)
        self.cap_matched_weights = rng_gate.normal(0, 0.05, (h, 4)).astype(np.float32)
        self.cap_matched_state = np.zeros(4, dtype=np.float32)

        self.episodes_learned = 0
        self.hidden_updates = 0
        self.last_gate_vector = np.ones(d, dtype=np.float32)

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
        self.channel_gate_elig.fill(0.0)
        self.v_inh.fill(0.0)
        self.cap_matched_state.fill(0.0)

    def compute_input_gate(self, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes effective sensory input and neuron-specific modulation vector.

        Returns:
            x_eff: shape (inputs,) effective input arriving at soma
            h_mod: shape (hidden,) per-neuron modulation multiplier
        """
        c = self.config
        d = c.inputs
        h = c.hidden
        arch = self.gating_arch

        x_eff = row.copy()
        h_mod = np.ones(h, dtype=np.float32)

        if arch == "no_gating":
            self.last_gate_vector = np.ones(d, dtype=np.float32)
            return x_eff, h_mod

        elif arch == "scalar_max_mag":
            # Round 8/9b baseline: scalar gate based on max(|x|)
            s_act = float(np.max(np.abs(row)))
            g_scalar = 1.0 if s_act > 0.6 else 0.15
            # In R8/R9b it was applied only to recurrent trace, but let's test input modulation:
            x_eff = row * g_scalar
            self.last_gate_vector = np.full(d, g_scalar, dtype=np.float32)
            return x_eff, h_mod

        elif arch == "fixed_channel":
            g_vec = np.full(d, 0.6, dtype=np.float32)
            x_eff = row * g_vec
            self.last_gate_vector = g_vec
            return x_eff, h_mod

        elif arch == "random_channel":
            x_eff = row * self.random_channel_weights
            self.last_gate_vector = self.random_channel_weights
            return x_eff, h_mod

        elif arch == "learned_channel":
            # Candidate A: g_k = sigmoid(s_k)
            # Bound logit to prevent overflow
            clipped_logits = np.clip(self.channel_gate_logits, -6.0, 6.0)
            g_vec = 1.0 / (1.0 + np.exp(-clipped_logits))
            x_eff = row * g_vec
            self.last_gate_vector = g_vec
            return x_eff, h_mod

        elif arch == "predictive_novelty":
            # Candidate B: predict x from previous spikes
            x_pred = self.w_pred @ self.core.spikes + self.b_pred
            residual = np.abs(row - x_pred)
            # Novelty attenuation: if residual is unexpectedly large, attenuate
            # But smooth sigmoid so signals aren't zeroed
            attenuation = 1.0 / (1.0 + np.maximum(0.0, residual - 0.5))
            x_eff = row * attenuation
            self.last_gate_vector = attenuation
            return x_eff, h_mod

        elif arch == "learned_inhibitory":
            # Candidate C: inhibitory interneurons
            # Feedforward excitation of inhibitory neurons
            i_inh = self.w_inh_in @ np.abs(row)
            self.v_inh = 0.8 * self.v_inh + i_inh
            z_inh = np.maximum(0.0, self.v_inh - 0.4)
            # Inhibitory feedback subtracts from somatic current inside step()
            # Here we reflect it in h_mod or return
            self.last_gate_vector = np.ones(d, dtype=np.float32)
            return x_eff, h_mod

        elif arch == "state_conditioned":
            # Candidate D: gate conditioned on soma memory state
            # If neuron already has high stored eligibility trace ||e_f||, protect it from input
            e_norm = np.linalg.norm(self.e_f, axis=1)  # (hidden,)
            z_mod = (
                self.scss_beta[0]
                + self.scss_beta[1] * self.core.voltage
                - self.scss_beta[2] * self.core.adaptation
                - self.scss_beta[3] * e_norm
            )
            h_mod = 1.0 / (1.0 + np.exp(-np.clip(z_mod, -6.0, 6.0)))
            self.last_gate_vector = np.ones(d, dtype=np.float32)
            return x_eff, h_mod

        elif arch == "capacity_matched":
            # Extra baseline parameters updating auxiliary recurrent state without selective gating
            self.cap_matched_state = 0.85 * self.cap_matched_state + 0.15 * (row[:4])
            aux_current = self.cap_matched_weights @ self.cap_matched_state
            # Added as small non-selective background to soma
            self.last_gate_vector = np.ones(d, dtype=np.float32)
            return x_eff, h_mod

        return x_eff, h_mod

    def step(self, x: np.ndarray, track: bool = True) -> np.ndarray:
        c = self.config
        row = np.asarray(x, dtype=np.float32)
        if row.shape != (c.inputs,) or not np.isfinite(row).all():
            raise ValueError("Invalid input vector")

        # Compute selective gating
        x_eff, h_mod = self.compute_input_gate(row)

        old_z = self.core.spikes
        theta = c.threshold + c.adaptation_strength * self.core.adaptation

        # Somatic input current
        if self.gating_arch == "state_conditioned":
            i_in = (self.core.w_in @ x_eff) * h_mod
        else:
            i_in = self.core.w_in @ x_eff

        i_rec = np.sum(self.core.w_rec * old_z[self.core.indices], axis=1)

        # Inhibitory subtraction for Candidate C
        if self.gating_arch == "learned_inhibitory":
            z_inh = np.maximum(0.0, self.v_inh - 0.4)
            i_in -= self.w_inh_out @ z_inh

        # Capacity matched auxiliary current
        if self.gating_arch == "capacity_matched":
            aux_cur = self.cap_matched_weights @ self.cap_matched_state
            i_in += 0.05 * aux_cur

        u_carry = c.leak * self.core.voltage + i_rec
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

            # Track channel eligibility for Candidate A (learned channel gating)
            # Gradient wrt channel logit s_k: x_k * (1 - g_k) * sum_h (w_in[h, k] * psi_h)
            if self.gating_arch == "learned_channel":
                g_vec = self.last_gate_vector
                # Sensitivity of soma activation to channel k
                ch_sens = np.sum(self.core.w_in * psi[:, None], axis=0)  # (inputs,)
                self.channel_gate_elig += row * g_vec * (1.0 - g_vec) * ch_sens

        # Step core voltage, spikes, and adaptation
        self.core.voltage = u
        self.core.spikes = (self.core.voltage >= theta).astype(np.float32)
        self.core.voltage -= self.core.spikes * theta
        self.core.adaptation = c.adaptation_decay * self.core.adaptation + self.core.spikes
        self.core.features = c.feature_decay * self.core.features + (1.0 - c.feature_decay) * self.core.spikes

        if track:
            # Stopped-reset Jacobian
            if self.reset_mode == "stopped_reset":
                self.e_v[:] = u_e
            else:
                self.e_v[:] = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old

            self.e_a[:] = c.adaptation_decay * a_old + z_e
            self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e

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

        # Readout gradient
        grad_out = err[:, None] * f_hat[None, :]
        self.core.w_out += np.clip(self.lr_out * grad_out, -self.clip_out, self.clip_out)
        self.core.bias += np.clip(self.lr_out * err, -self.clip_out, self.clip_out)
        np.clip(self.core.w_out, -3.5, 3.5, out=self.core.w_out)
        np.clip(self.core.bias, -3.5, 3.5, out=self.core.bias)

        # Pre-update credit signal for hidden layers
        pre_update_signal = (self.core.w_out.T @ err) * (np.abs(raw) < 5.0) / scale

        # Hidden synaptic update
        direction = pre_update_signal[:, None] * self.e_f
        self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
        self.syn_energy = 0.99 * self.syn_energy + 0.01 * (direction * direction)
        rms = np.sqrt(self.syn_energy) + 1e-4
        norm_step = self.syn_trace / rms
        step = np.clip(self.lr_hid * norm_step, -self.clip_hid, self.clip_hid)

        d = self.config.inputs
        self.core.w_in += step[:, :d]
        self.core.w_rec += step[:, d:]
        np.clip(self.core.w_in, -3.0, 3.0, out=self.core.w_in)
        np.clip(self.core.w_rec, -3.0, 3.0, out=self.core.w_rec)
        self.hidden_updates += 1

        # Learned channel gate update (Candidate A)
        if self.gating_arch == "learned_channel":
            # Scalar credit alignment: dot product between error projection and channel eligibility
            # Gate update: delta s_k = lr_gate * sum_h (pre_update_signal_h * channel_elig_k)
            # Normalized by norm of signal
            sig_scale = float(np.linalg.norm(pre_update_signal)) + 1e-4
            gate_step = self.lr_gate * (self.channel_gate_elig / sig_scale) * np.sum(err)
            self.channel_gate_logits += np.clip(gate_step, -0.05, 0.05)
            # Keep logits within reasonable operational bounds [-3.0, 3.0]
            np.clip(self.channel_gate_logits, -3.0, 3.0, out=self.channel_gate_logits)

        self.episodes_learned += 1
        return loss

    def revert_hidden(self) -> None:
        """Causal intervention: reverts hidden synaptic weights to initialization."""
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

    def clone(self) -> SelectiveSensoryALIF:
        return copy.deepcopy(self)
