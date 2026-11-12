"""Independent, self-contained replication engine for SCAR-X Round 10.

Implements independent eligibility estimators:
- Own-reset (Standard local E-prop with own reset spike terms)
- Stopped-reset (Local E-prop with spike reset terms dropped)
- E-prop matched (Bellec et al. 2020: detached reset spike, adaptive threshold retained)
- E-prop aligned (Bellec et al. 2020: single timescale, no sensory gating)
- BrainTrace D-RTRL (Chao et al. 2024: explicit 3x3 block-diagonal parameter-dimensional recurrence)
- Full Surrogate Oracle (recurrent BPTT/RTRL control for gradient validation)

All algorithms run on exactly ONE unchanged ALIF SpikingCore (H48, D8, Fan-in 8).
"""
from __future__ import annotations

import copy
import numpy as np
from ..neural.core import Config, SpikingCore
from ..evaluation.readout_eligibility import spike_slope


def local_blocks_d_rtrl(
    config: Config, theta: np.ndarray, spike: np.ndarray, psi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3x3 local block Jacobian D_j and input coupling B_j for D-RTRL.

    States: s_j = [v_j, a_j, f_j]^T
    D_j = d(s_j[t]) / d(s_j[t-1])
    B_j = d(s_j[t]) / d(I_j[t])
    """
    c = config
    h = len(theta)
    d = np.zeros((h, 3, 3), dtype=np.float32)
    b = np.column_stack((1.0 - theta * psi, psi, (1.0 - c.feature_decay) * psi)).astype(np.float32)

    # d(v')/d(v) = leak * (1 - theta * psi)
    d[:, 0, 0] = c.leak * (1.0 - theta * psi)
    # d(v')/d(a) = beta * (theta * psi - z)
    d[:, 0, 1] = c.adaptation_strength * (theta * psi - spike)
    # d(a')/d(v) = leak * psi
    d[:, 1, 0] = c.leak * psi
    # d(a')/d(a) = rho - beta * psi
    d[:, 1, 1] = c.adaptation_decay - c.adaptation_strength * psi
    # d(f')/d(v) = (1 - gamma_f) * leak * psi
    d[:, 2, 0] = (1.0 - c.feature_decay) * c.leak * psi
    # d(f')/d(a) = -(1 - gamma_f) * beta * psi
    d[:, 2, 1] = -(1.0 - c.feature_decay) * c.adaptation_strength * psi
    # d(f')/d(f) = gamma_f
    d[:, 2, 2] = c.feature_decay
    return d, b


class IndependentLearningALIF:
    """Unified, independent engine for ALIF credit assignment evaluation."""

    def __init__(
        self,
        config: Config,
        method: str = "own",
        condition: str = "learned_mtsg",
        sensory_threshold: float = 0.6,
    ):
        config.validate()
        if (config.hidden, config.inputs, config.outputs, config.fan_in) != (48, 8, 2, 8):
            raise ValueError("SCAR-X scope strictly requires H48 D8 O2 K8")

        self.config = config
        self.method = method
        self.condition = condition
        self.sensory_threshold = sensory_threshold
        self.core = SpikingCore(config)

        h = config.hidden
        d_tot = config.inputs + config.fan_in

        # Eligibility traces
        self.e_v = np.zeros((h, d_tot), dtype=np.float32)
        self.e_a = np.zeros((h, d_tot), dtype=np.float32)
        self.e_f = np.zeros((h, d_tot), dtype=np.float32)

        if method.startswith("acr_static_"):
            self.e_v_stop = np.zeros((h, d_tot), dtype=np.float32)
            self.e_a_stop = np.zeros((h, d_tot), dtype=np.float32)
            self.e_f_own = np.zeros((h, d_tot), dtype=np.float32)
            self.e_f_stop = np.zeros((h, d_tot), dtype=np.float32)
            self.acr_alpha = float(int(method.split("_")[-1])) / 100.0

        # Multi-timescale eligibility decay vector
        if method == "eprop_aligned":
            self.trace_kind = "slow"
            self.gate_mode = "none"
            self.gamma_e_vec = np.full((h, 1), config.feature_decay, dtype=np.float32)
        elif method == "braintrace":
            self.trace_kind = "slow"
            self.gate_mode = "none"
            self.gamma_e_vec = np.full((h, 1), config.feature_decay, dtype=np.float32)
        else:
            self.trace_kind = "multi"
            self.gate_mode = "activity"
            self.gamma_e_vec = np.linspace(0.25, 0.92, h, dtype=np.float32)[:, None]

        # Optimizer state
        self.syn_trace = np.zeros((h, d_tot), dtype=np.float32)
        self.syn_energy = np.full((h, d_tot), 1e-4, dtype=np.float32)

        # Feature normalization statistics
        self.f_mean = np.zeros(h, dtype=np.float32)
        self.f_var = np.full(h, 0.01, dtype=np.float32)

        # Tracking and anchors
        self.w_in_0 = self.core.w_in.copy()
        self.w_rec_0 = self.core.w_rec.copy()
        self.signal_rng = np.random.default_rng(config.seed + 35100000)
        self.calibrated = False
        self.episodes_learned = 0
        self.hidden_updates = 0
        self.total_ticks = 0

    def calibrate(self, sequences: np.ndarray) -> None:
        seqs = np.asarray(sequences, dtype=np.float32)
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
        if hasattr(self, "e_v_stop"):
            self.e_v_stop.fill(0.0)
            self.e_a_stop.fill(0.0)
            self.e_f_own.fill(0.0)
            self.e_f_stop.fill(0.0)

    def step(self, x: np.ndarray, track: bool = True) -> np.ndarray:
        c = self.config
        row = np.asarray(x, dtype=np.float32)
        if row.shape != (c.inputs,) or not np.isfinite(row).all():
            raise ValueError("Invalid input vector")

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

            if self.method == "braintrace":
                previous = np.stack((self.e_v, self.e_a, self.e_f), axis=1)
            elif self.method.startswith("acr_static_"):
                own_u = c.leak * self.e_v + pre
                own_a = self.e_a.copy()
                stop_u = c.leak * self.e_v_stop + pre
                stop_a = self.e_a_stop.copy()
                own_z = psi[:, None] * (own_u - c.adaptation_strength * own_a)
                stop_z = psi[:, None] * (stop_u - c.adaptation_strength * stop_a)
            else:
                u_e = c.leak * self.e_v + pre
                a_old = self.e_a.copy()
                z_e = psi[:, None] * (u_e - c.adaptation_strength * a_old)

        # Single unchanged neural core forward
        self.core.step(row, track=False)

        if track:
            if self.method == "braintrace":
                d, b = local_blocks_d_rtrl(c, theta, self.core.spikes, psi)
                current = np.einsum("hab,hbk->hak", d, previous) + b[:, :, None] * pre[:, None, :]
                self.e_v[:] = current[:, 0]
                self.e_a[:] = current[:, 1]
                self.e_f[:] = current[:, 2]
            elif self.method.startswith("acr_static_"):
                self.e_v[:] = own_u - theta[:, None] * own_z - self.core.spikes[:, None] * c.adaptation_strength * own_a
                self.e_v_stop[:] = stop_u
                self.e_a[:] = c.adaptation_decay * own_a + own_z
                self.e_a_stop[:] = c.adaptation_decay * stop_a + stop_z
                gate = 1.0 if float(np.max(np.abs(row))) > self.sensory_threshold else 0.15
                own_z[:, c.inputs:] *= gate
                stop_z[:, c.inputs:] *= gate
                self.e_f_own[:] = self.gamma_e_vec * self.e_f_own + (1.0 - self.gamma_e_vec) * own_z
                self.e_f_stop[:] = self.gamma_e_vec * self.e_f_stop + (1.0 - self.gamma_e_vec) * stop_z
                a = self.acr_alpha
                self.e_f[:] = (1.0 - a) * self.e_f_own + a * self.e_f_stop
                if a == 0.0:
                    self.e_f[:] = self.e_f_own
                elif a == 1.0:
                    self.e_f[:] = self.e_f_stop
            elif self.method == "stopped":
                # Stopped-reset: drop spike reset terms from voltage derivative
                self.e_v[:] = u_e
                self.e_a[:] = c.adaptation_decay * a_old + z_e
                if self.gate_mode == "activity":
                    gate = 1.0 if float(np.max(np.abs(row))) > self.sensory_threshold else 0.15
                    z_e[:, c.inputs:] *= gate
                self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e
            elif self.method == "eprop_matched":
                # E-prop matched: detach reset spike, retain adaptive theta
                self.e_v[:] = u_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
                self.e_a[:] = c.adaptation_decay * a_old + z_e
                if self.gate_mode == "activity":
                    gate = 1.0 if float(np.max(np.abs(row))) > self.sensory_threshold else 0.15
                    z_e[:, c.inputs:] *= gate
                self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e
            elif self.method == "eprop_aligned":
                # E-prop aligned: detach reset spike, single slow timescale, no gate
                self.e_v[:] = u_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
                self.e_a[:] = c.adaptation_decay * a_old + z_e
                self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e
            else:
                # Default "own": full own-neuron reset derivatives
                self.e_v[:] = (
                    u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
                )
                self.e_a[:] = c.adaptation_decay * a_old + z_e
                if self.gate_mode == "activity":
                    gate = 1.0 if float(np.max(np.abs(row))) > self.sensory_threshold else 0.15
                    z_e[:, c.inputs:] *= gate
                self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e

        self.total_ticks += 1
        return self.core.features

    def standardized_features(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = np.maximum(np.sqrt(self.f_var), 0.05).astype(np.float32)
        raw = (f - self.f_mean) / scale
        f_hat = np.clip(raw, -5.0, 5.0).astype(np.float32)
        return f_hat, raw, scale

    def learn_episode(
        self,
        sequence: np.ndarray,
        target: int,
        stat_rate: float = 0.01,
        record_diagnostics: bool = False,
    ) -> tuple[float, dict | None]:
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

        logits = self.core.w_out @ f_hat + self.core.bias
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= np.sum(p)
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
        diag = {} if record_diagnostics else None

        # Hidden update
        if self.condition not in ("frozen_hidden", "zero_signal", "readout_only"):
            signal = pre_update_signal
            if self.condition == "shuffled_signal":
                signal = signal[perm]
            direction = signal[:, None] * self.e_f

            # Track pre-optimizer direction
            if record_diagnostics:
                diag["raw_direction"] = direction.copy()
                diag["signal"] = signal.copy()
                diag["e_f"] = self.e_f.copy()

            self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
            self.syn_energy = 0.99 * self.syn_energy + 0.01 * (direction * direction)
            rms = np.sqrt(self.syn_energy) + 1e-4
            norm_step = self.syn_trace / rms

            lr_hid = 0.008
            clip_hid = 0.010
            step = np.clip(lr_hid * norm_step, -clip_hid, clip_hid)

            if record_diagnostics:
                diag["norm_step"] = norm_step.copy()
                diag["step"] = step.copy()
                diag["rms"] = rms.copy()

            d = self.config.inputs
            self.core.w_in += step[:, :d]
            self.core.w_rec += step[:, d:]
            np.clip(self.core.w_in, -3.0, 3.0, out=self.core.w_in)
            np.clip(self.core.w_rec, -3.0, 3.0, out=self.core.w_rec)
            self.hidden_updates += 1

        self.core.updates += 1
        self.episodes_learned += 1
        return loss, diag

    def features_many(self, sequences: np.ndarray) -> np.ndarray:
        out = []
        for seq in sequences:
            self.core.reset_state()
            for row in seq:
                self.core.step(row, track=False)
            out.append(self.core.features.copy())
        return np.asarray(out, dtype=np.float32)

    def predict_many(self, sequences: np.ndarray) -> np.ndarray:
        feats = self.features_many(sequences)
        scale = np.maximum(np.sqrt(self.f_var), 0.05).astype(np.float32)
        f_hat = np.clip((feats - self.f_mean) / scale, -5.0, 5.0).astype(np.float32)
        logits = f_hat @ self.core.w_out.T + self.core.bias
        return np.argmax(logits, axis=1).astype(np.int32)

    def revert_hidden_weights(self, revert_in: bool = True, revert_rec: bool = True) -> None:
        if revert_in:
            self.core.w_in[:] = self.w_in_0
        if revert_rec:
            self.core.w_rec[:] = self.w_rec_0
