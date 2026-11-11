"""Dendritic Coincidence-Gated ALIF (DC-ALIF) with Own-Neuron Forward Eligibility
and Counterfactual Readout Usefulness Consolidation.

Isolated evaluation prototype for campaign `scarx-neural-round1-20261111` (Agent: `antigravity`).
Strictly obeys all SCAR-X non-negotiable constraints:
- Single 48-neuron ALIF spiking recurrent core (`fan_in=8`, `plastic_rows=12`).
- Uses only the Proposal 30 approved one-step readout-error broadcast `W_out.T @ error`
  combined with own-neuron forward eligibility traces (`e_d`, `e_v`, `e_a`, `e_f`).
  Never transports error hidden->hidden (`W_rec.T`) or reverse-in-time in the candidate.
- Includes an isolated `DCALIFBPTTControl` (`control_only=True`) strictly as an upper-bound
  learnability control, rejected by all candidate and serving loaders.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from typing import Any

import numpy as np

from ..neural.core import Config, SpikingCore
from .readout_conditioning import ConditionedReadout


def spike_slope(margin: np.ndarray | float, width: float = 0.4) -> np.ndarray:
    raw = np.asarray(margin, dtype=np.float64) / width
    p = 1.0 / (1.0 + np.exp(-np.clip(raw, -20.0, 20.0)))
    return (p * (1.0 - p) / width * (np.abs(raw) < 20.0)).astype(np.float32)


def contrast_matrix(inputs: int, gamma_inh: float = 0.85) -> np.ndarray:
    """Symmetric full-rank disynaptic lateral contrast operator P_gamma on adjacent afferent pairs."""
    p = np.eye(inputs, dtype=np.float32)
    for m in range(inputs // 2):
        i0, i1 = 2 * m, 2 * m + 1
        p[i0, i1] = -float(gamma_inh)
        p[i1, i0] = -float(gamma_inh)
    return p


def compute_coact(x: np.ndarray, pulse_thresh: float = 0.35) -> float:
    """Physical instantaneous multi-channel co-activation index in [0, 1] from raw input x(t)."""
    xp = np.maximum(0.0, np.abs(np.asarray(x, dtype=np.float64)) - pulse_thresh)
    e1 = float(np.sum(xp))
    e2 = float(np.sum(xp * xp))
    if e1 <= 1e-6:
        return 0.0
    return float(min(1.0, max(0.0, 2.0 * (e1 * e1 - e2) / (e1 * e1 + 1e-6))))


class DCALIFEligibility:
    """Proposal 30 own-neuron forward eligibility learner for Dendritic Coincidence-Gated ALIF (DC-ALIF)."""

    conditions = ("correct", "no_gate", "shuffled", "zero", "frozen")
    array_names = ("d", "e_d", "e_v", "e_a", "e_f", "bank", "moment", "variance", "output_mask", "omega", "anchor_out")

    # Dendritic compartment hyperparameters (locked)
    d_decay = 0.96
    gamma_inh = 0.85
    alpha_s = 0.32
    alpha_d = 0.48
    gamma_q = 0.95
    beta_g = 1.35
    beta_d = 1.65
    adam_lr = 0.0035
    step_clip = 0.008

    def __init__(self, config: Config, condition: str = "correct", usefulness_beta: float = 0.0) -> None:
        if (
            condition not in self.conditions
            or config.neuron != "alif"
            or config.hidden != 48
            or config.fan_in != 8
            or not 1 <= config.inputs <= 454
            or not 2 <= config.outputs <= 266
        ):
            raise ValueError("Bounded 48-ALIF DC-ALIF research configuration required")
        config.validate()
        self.core = SpikingCore(config)
        self.head = ConditionedReadout(self.core.w_out, self.core.bias, "normalized")
        self.core.w_out = self.head.weights
        self.core.bias = self.head.bias
        self.condition = condition
        self.usefulness_beta = float(usefulness_beta)

        h = config.hidden
        d_in = config.inputs
        k = config.fan_in
        self.P = contrast_matrix(d_in, self.gamma_inh)

        # Dynamic dendritic plateau trace (48 floats = 192 bytes)
        self.d = np.zeros(h, dtype=np.float32)

        # Own-neuron forward eligibility arrays (48 x (D + K))
        for name in ("e_d", "e_v", "e_a", "e_f", "bank", "moment", "variance"):
            setattr(self, name, np.zeros((h, d_in + k), dtype=np.float32))
        self.output_mask = np.ones(config.outputs, dtype=bool)
        self.omega = np.zeros_like(self.core.w_out)
        self.anchor_out = self.core.w_out.copy()

        self.signal_rng = np.random.default_rng(config.seed + 35100000)
        self.total_ticks = 0
        self.steps_in_segment = 0
        self.endpoints = 0
        self.hidden_updates = 0
        self.segments_learned = 0
        self.segment_active = False
        self.track = False
        self.last_direction_norm: float | None = None

    def begin(self, track: bool = True) -> None:
        if type(track) is not bool or (self.segment_active and self.endpoints):
            raise ValueError("Finish/rollback pending supervised segment first")
        self.core.reset_state()
        self.d.fill(0.0)
        for n in ("e_d", "e_v", "e_a", "e_f", "bank"):
            getattr(self, n).fill(0.0)
        self.steps_in_segment = 0
        self.endpoints = 0
        self.segment_active = True
        self.track = track and self.condition != "frozen"

    def step(self, x: np.ndarray) -> np.ndarray:
        c = self.core.config
        x = np.asarray(x, dtype=np.float32)
        if not self.segment_active or self.steps_in_segment >= 256 or x.shape != (c.inputs,) or not np.isfinite(x).all():
            raise ValueError("Active bounded finite input segment required")

        old_z = self.core.spikes.copy()
        old_d = self.d.copy()
        old_v = self.core.voltage.copy()
        old_a = self.core.adaptation.copy()
        old_f = self.core.features.copy()

        no_gate = self.condition == "no_gate"
        w_in_hat = self.core.w_in @ self.P if not no_gate else self.core.w_in
        u_in = w_in_hat * x[None, :]  # (H, D)
        I_in = np.sum(u_in, axis=1)  # (H,)
        rec_z = old_z[self.core.indices]  # (H, K)
        I_rec = np.sum(self.core.w_rec * rec_z, axis=1)  # (H,)
        I_total = (I_in + I_rec).astype(np.float32)

        # Pairwise group sums s_{j,m}(t) for zero-diagonal cross-pair coincidence q_j(t)
        n_pairs = c.inputs // 2
        if n_pairs > 0 and not no_gate:
            u_pairs = u_in[:, : 2 * n_pairs].reshape(c.hidden, n_pairs, 2)
            s_pairs = np.sum(u_pairs, axis=2)  # (H, n_pairs)
            sq_pairs = np.sum(s_pairs * s_pairs, axis=1)
            if 2 * n_pairs < c.inputs:
                sq_pairs = sq_pairs + u_in[:, -1] ** 2
            q_t = (0.5 * (I_in * I_in - sq_pairs)).astype(np.float32)
            coact = compute_coact(x)
            alpha_d_eff = float(self.alpha_d * ((1.0 - coact) ** 2))
            alpha_s_eff = float(self.alpha_s * (1.0 - coact))
            beta_g_eff = float(self.beta_g * (1.0 - coact))
            gamma_q_eff = float(self.gamma_q)
        else:
            s_pairs = np.zeros((c.hidden, 0), dtype=np.float32)
            q_t = np.zeros(c.hidden, dtype=np.float32)
            alpha_d_eff = float(self.alpha_d)
            alpha_s_eff = float(self.alpha_s)
            beta_g_eff = 0.0
            gamma_q_eff = 0.0

        # Dendritic plateau update
        s_d = np.tanh(alpha_d_eff * I_total + gamma_q_eff * q_t).astype(np.float32)
        d_new = (self.d_decay * old_d + (1.0 - self.d_decay) * s_d).astype(np.float32)

        # Bilinear dendritic-somatic coincidence gate C_j(t) = g_gate_j(t) * d_j(t-1)
        if no_gate:
            g_gate = np.ones(c.hidden, dtype=np.float32)
            dg_dI = np.zeros(c.hidden, dtype=np.float32)
        else:
            g_gate = np.tanh(beta_g_eff * I_total).astype(np.float32)
            dg_dI = (beta_g_eff * (1.0 - g_gate * g_gate)).astype(np.float32)

        C_t = (g_gate * old_d).astype(np.float32)

        # Somatic ALIF update
        theta = (c.threshold + c.adaptation_strength * old_a).astype(np.float32)
        u = (c.leak * old_v + alpha_s_eff * I_total + self.beta_d * C_t).astype(np.float32)
        z_new = (u >= theta).astype(np.float32)
        v_new = (u - z_new * theta).astype(np.float32)
        a_new = (c.adaptation_decay * old_a + z_new).astype(np.float32)
        f_new = (c.feature_decay * old_f + (1.0 - c.feature_decay) * z_new).astype(np.float32)

        self.d[:] = d_new
        self.core.voltage[:] = v_new
        self.core.adaptation[:] = a_new
        self.core.spikes[:] = z_new
        self.core.features[:] = f_new

        if self.track:
            psi = spike_slope(u - theta)
            x_b = np.broadcast_to(x, (c.hidden, c.inputs))
            dI_dw_in = x_b @ self.P if not no_gate else x_b  # (H, D)
            dI_dw_rec = rec_z  # (H, K)
            dI_dw = np.concatenate((dI_dw_in, dI_dw_rec), axis=1).astype(np.float32)

            if n_pairs > 0 and not no_gate:
                s_expanded = np.repeat(s_pairs, 2, axis=1)
                if 2 * n_pairs < c.inputs:
                    s_expanded = np.concatenate((s_expanded, u_in[:, -1:]), axis=1)
                dq_dwhat = x_b * (I_in[:, None] - s_expanded)
                dq_dw_in = dq_dwhat @ self.P
            else:
                dq_dw_in = np.zeros((c.hidden, c.inputs), dtype=np.float32)
            dq_dw = np.concatenate((dq_dw_in, np.zeros((c.hidden, c.fan_in), dtype=np.float32)), axis=1).astype(
                np.float32
            )

            ed_old = self.e_d.copy()
            ev_old = self.e_v.copy()
            ea_old = self.e_a.copy()
            ef_old = self.e_f.copy()

            ds_d = (1.0 - s_d[:, None] ** 2) * (alpha_d_eff * dI_dw + gamma_q_eff * dq_dw)
            dC = (dg_dI[:, None] * old_d[:, None]) * dI_dw + g_gate[:, None] * ed_old

            self.e_d[:] = self.d_decay * ed_old + (1.0 - self.d_decay) * ds_d
            u_e = c.leak * ev_old + alpha_s_eff * dI_dw + self.beta_d * dC
            z_e = psi[:, None] * (u_e - c.adaptation_strength * ea_old)
            self.e_v[:] = u_e - theta[:, None] * z_e - z_new[:, None] * c.adaptation_strength * ea_old
            self.e_a[:] = c.adaptation_decay * ea_old + z_e
            self.e_f[:] = c.feature_decay * ef_old + (1.0 - c.feature_decay) * z_e
            if not all(np.isfinite(getattr(self, n)).all() for n in ("e_d", "e_v", "e_a", "e_f")):
                raise ValueError("Nonfinite local DC-ALIF sensitivity")

        self.total_ticks += 1
        self.steps_in_segment += 1
        return self.core.features

    def accumulate(self, target: int) -> float:
        if (
            not self.segment_active
            or not self.steps_in_segment
            or not self.head.frozen
            or type(target) is not int
            or not 0 <= target < len(self.output_mask)
            or not self.output_mask[target]
        ):
            raise ValueError("Calibrated supervised endpoint required")
        feature = self.core.features
        p = self.head.probabilities(feature, self.output_mask)
        error = -p.copy()
        error[target] += 1.0
        scale = np.maximum(np.sqrt(self.head.m2 / (self.head.count - 1)), 0.05).astype(np.float32)
        raw = (feature - self.head.mean.astype(np.float32)) / scale
        signal = (self.head.weights.T @ error) * (np.abs(raw) < 5.0) / scale
        permutation = self.signal_rng.permutation(self.core.config.hidden)
        if self.condition == "shuffled":
            signal = signal[permutation]
        if self.condition in ("correct", "no_gate", "shuffled"):
            self.bank += signal[:, None] * self.e_f
            if not np.isfinite(self.bank).all():
                raise ValueError("Nonfinite local direction")

        if self.usefulness_beta > 0.0:
            from .readout_usefulness import counterfactual_benefit

            old_w = self.head.weights.copy()
            f_trans = self.head.transform(feature)
            # Counterfactual benefit operates on scaled features in [-1, 1]
            f_unit = np.clip(f_trans / 5.0, -1.0, 1.0)
            w_scaled = np.clip(old_w * 5.0, -20.0, 20.0)
            g = 1.0 / (1.0 + self.usefulness_beta * self.omega)
            delta_w = 0.08 * error[:, None] * f_trans[None, :]
            delta_b = 0.08 * error
            self.head.weights[:] = np.clip(old_w + np.clip(delta_w * g, -0.025, 0.025), -3.0, 3.0)
            self.head.bias[:] = np.clip(self.head.bias + np.clip(delta_b, -0.025, 0.025), -3.0, 3.0)
            self.head.updates += 1
            new_omega = 0.995 * self.omega
            if int(np.argmax(p)) == target:
                benefit = counterfactual_benefit(w_scaled, f_unit, self.head.bias, self.output_mask, int(target))
                new_omega = np.maximum(new_omega, np.clip(benefit, 0.0, 1.0).astype(np.float32))
            self.omega[:] = new_omega
            loss = -float(np.log(max(float(p[target]), 1e-12)))
        else:
            loss = self.head.learn(feature, target, self.output_mask)

        self.core.updates += 1
        self.endpoints += 1
        return loss

    def _apply_direction(self, direction: np.ndarray) -> None:
        if direction.shape != self.bank.shape or not np.isfinite(direction).all():
            raise ValueError("Bounded finite synapse-local direction")
        self.last_direction_norm = float(np.linalg.norm(direction.astype(np.float64)))
        self.hidden_updates += 1
        self.moment *= 0.9
        self.moment += 0.1 * direction
        self.variance *= 0.999
        self.variance += 0.001 * direction * direction
        update = (
            self.adam_lr
            * (self.moment / (1.0 - 0.9**self.hidden_updates))
            / (np.sqrt(self.variance / (1.0 - 0.999**self.hidden_updates)) + 1e-8)
        )
        if self.usefulness_beta > 0.0:
            neuron_omega = np.max(self.omega, axis=0)[:, None]
            g_hid = 1.0 / (1.0 + 0.5 * self.usefulness_beta * neuron_omega)
            update = update * g_hid
        if not np.isfinite(update).all():
            raise ValueError("Nonfinite local Adam update")
        d = self.core.config.inputs
        for name, delta in (("w_in", update[:, :d]), ("w_rec", update[:, d:])):
            w = getattr(self.core, name)
            w += np.clip(delta, -self.step_clip, self.step_clip)
            np.clip(w, -3.0, 3.0, out=w)

    def finish(self) -> None:
        if not self.segment_active or not self.endpoints:
            raise ValueError("No pending endpoint to finish")
        if self.condition in ("correct", "no_gate", "shuffled"):
            self._apply_direction(self.bank / self.endpoints)
        self.segments_learned += 1
        self.segment_active = False
        self.track = False

    @staticmethod
    def _sequence(sequence: np.ndarray, dimensions: int) -> np.ndarray:
        x = np.asarray(sequence, dtype=np.float32)
        if x.ndim != 2 or not 1 <= len(x) <= 256 or x.shape[1] != dimensions or not np.isfinite(x).all():
            raise ValueError("Finite 1..256 tick sequence required")
        return x

    def features(self, sequence: np.ndarray, reset_each_tick: bool = False) -> np.ndarray:
        x = self._sequence(sequence, self.core.config.inputs)
        self.begin(False)
        for row in x:
            if reset_each_tick:
                self.core.reset_state()
                self.d.fill(0.0)
            self.step(row)
        self.segment_active = False
        return self.core.features.copy()

    def features_many(self, sequences: np.ndarray, reset_each_tick: bool = False) -> np.ndarray:
        return np.stack([self.features(x, reset_each_tick) for x in sequences])

    def calibrate(self, sequences: np.ndarray) -> None:
        if len(sequences) != 128 or self.head.frozen:
            raise ValueError("Exactly 128 unlabeled calibration inputs required")
        while self.head.count < 128:
            self.head.observe(self.features(sequences[self.head.count]))
        self.head.freeze()

    def learn(self, sequence: np.ndarray, target: int) -> float:
        x = self._sequence(sequence, self.core.config.inputs)
        if not self.head.frozen or type(target) is not int or not 0 <= target < self.core.config.outputs:
            raise ValueError("Valid calibrated supervision required")
        snapshot = copy.deepcopy(self.__dict__)
        try:
            self.begin(True)
            for row in x:
                self.step(row)
            loss = self.accumulate(target)
            self.finish()
            return loss
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            raise

    def predict(self, sequence: np.ndarray) -> int:
        return int(self.head.probabilities(self.features(sequence), self.output_mask).argmax())

    def memory_bytes(self) -> int:
        return (
            self.core.memory_bytes()
            + self.head.mean.nbytes
            + self.head.m2.nbytes
            + self.P.nbytes
            + sum(getattr(self, n).nbytes for n in self.array_names)
        )

    def dumps(self) -> bytes:
        core = self.core.dumps()
        head = self.head.dumps()
        state_io = io.BytesIO()
        np.savez(state_io, **{n: getattr(self, n) for n in self.array_names})
        state = state_io.getvalue()
        meta = {
            "format": "scarx-dc-alif-eligibility-v1",
            "condition": self.condition,
            "usefulness_beta": self.usefulness_beta,
            "signal_rng": self.signal_rng.bit_generator.state,
            **{
                n: getattr(self, n)
                for n in (
                    "total_ticks",
                    "steps_in_segment",
                    "endpoints",
                    "hidden_updates",
                    "segments_learned",
                    "segment_active",
                    "track",
                    "last_direction_norm",
                )
            },
            **{n + "_sha256": hashlib.sha256(p).hexdigest() for n, p in (("core", core), ("head", head), ("state", state))},
        }
        meta["metadata_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            for n, p in [
                ("core.npz", core),
                ("head.npz", head),
                ("state.npz", state),
                ("metadata.json", json.dumps(meta)),
            ]:
                z.writestr(zipfile.ZipInfo(n), p)
        raw = out.getvalue()
        if len(raw) > 4_000_000:
            raise MemoryError("DC-ALIF checkpoint exceeds 4MB")
        return raw

    @classmethod
    def loads(cls, payload: bytes) -> "DCALIFEligibility":
        if len(payload) > 4_000_000:
            raise ValueError("DC-ALIF checkpoint bound")
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if (
                set(z.namelist()) != {"core.npz", "head.npz", "state.npz", "metadata.json"}
                or len(z.namelist()) != 4
                or sum(i.file_size for i in z.infolist()) > 4_000_000
                or z.getinfo("metadata.json").file_size > 65536
            ):
                raise ValueError("DC-ALIF checkpoint members")
            blobs = {n: z.read(n + ".npz") for n in ("core", "head", "state")}
            meta = json.loads(z.read("metadata.json"))
        checksum = meta.pop("metadata_sha256", None)
        if (
            meta.get("format") != "scarx-dc-alif-eligibility-v1"
            or meta.get("control_only") is True
            or hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest() != checksum
            or any(hashlib.sha256(p).hexdigest() != meta.get(n + "_sha256") for n, p in blobs.items())
        ):
            raise ValueError("DC-ALIF checkpoint checksum/format verification failed")
        with np.load(io.BytesIO(blobs["core"]), allow_pickle=False) as z:
            config = Config(**json.loads(str(z["metadata"]))["config"])
        m = cls(config, meta["condition"], usefulness_beta=float(meta.get("usefulness_beta", 0.0)))
        loaded = SpikingCore.loads(blobs["core"])
        head = ConditionedReadout.loads(blobs["head"])
        if not np.array_equal(loaded.w_out, head.weights) or not np.array_equal(loaded.bias, head.bias) or loaded.updates != head.updates:
            raise ValueError("DC-ALIF head/core mismatch")
        m.core = loaded
        m.head = head
        m.core.w_out = m.head.weights
        m.core.bias = m.head.bias
        with np.load(io.BytesIO(blobs["state"]), allow_pickle=False) as z:
            if set(z.files) != set(m.array_names):
                raise ValueError("DC-ALIF state fields mismatch")
            for name in m.array_names:
                a = z[name]
                template = getattr(m, name)
                if a.shape != template.shape or a.dtype != template.dtype or not np.isfinite(a).all() or (
                    name == "variance" and np.any(a < 0)
                ):
                    raise ValueError("DC-ALIF state array invalid")
                template[:] = a
        for name in ("total_ticks", "steps_in_segment", "endpoints", "hidden_updates", "segments_learned"):
            value = meta[name]
            if type(value) is not int or value < 0:
                raise ValueError("DC-ALIF counter invalid")
            setattr(m, name, value)
        for name in ("segment_active", "track"):
            if type(meta[name]) is not bool:
                raise ValueError("DC-ALIF flag invalid")
            setattr(m, name, meta[name])
        value = meta["last_direction_norm"]
        if value is not None and (type(value) not in (int, float) or not np.isfinite(value) or value < 0):
            raise ValueError("DC-ALIF direction norm invalid")
        m.last_direction_norm = value
        m.signal_rng.bit_generator.state = meta["signal_rng"]
        if m.condition in ("zero", "frozen") and (m.hidden_updates or np.any(m.moment) or np.any(m.variance)):
            raise ValueError("Inactive hidden optimizer")
        return m


class DCALIFBPTTControl:
    """Isolated reverse-time BPTT upper-bound control on DCALIFEligibility (`control_only=True`)."""

    control_only = True

    def __init__(self, reference: DCALIFEligibility) -> None:
        if not isinstance(reference, DCALIFEligibility) or reference.condition not in ("correct", "frozen"):
            raise ValueError("Matched DC-ALIF research reference required")
        self.reference = reference
        self.backward_passes = 0
        self.surrogate_forward_ticks = 0
        self.max_graph_state_error = 0.0
        self.max_graph_feature_error = 0.0

    def _direction(self, sequence: np.ndarray, recorded_states: list[tuple[np.ndarray, ...]], target: int) -> np.ndarray:
        import torch
        from .controls import WordTapeSurrogate

        m = self.reference
        c = m.core.config
        wi = torch.tensor(m.core.w_in, requires_grad=True)
        wr = torch.tensor(m.core.w_rec, requires_grad=True)
        P_t = torch.tensor(m.P)
        indices = torch.as_tensor(m.core.indices, dtype=torch.long)

        v, a, z, f, d = [torch.zeros(c.hidden) for _ in range(5)]
        n_pairs = c.inputs // 2

        for x_np, obs in zip(sequence, recorded_states):
            x_t = torch.as_tensor(x_np)
            wi_hat = wi @ P_t
            u_in = wi_hat * x_t[None, :]
            I_in = u_in.sum(dim=1)
            I_rec = (wr * z[indices]).sum(dim=1)
            I_total = I_in + I_rec

            if n_pairs > 0:
                u_pairs = u_in[:, : 2 * n_pairs].reshape(c.hidden, n_pairs, 2)
                s_pairs = u_pairs.sum(dim=2)
                sq_pairs = (s_pairs * s_pairs).sum(dim=1)
                if 2 * n_pairs < c.inputs:
                    sq_pairs = sq_pairs + u_in[:, -1] ** 2
                q_t = 0.5 * (I_in * I_in - sq_pairs)
                coact = compute_coact(x_np)
                alpha_d_eff = float(m.alpha_d * ((1.0 - coact) ** 2))
                alpha_s_eff = float(m.alpha_s * (1.0 - coact))
                beta_g_eff = float(m.beta_g * (1.0 - coact))
                gamma_q_eff = float(m.gamma_q)
            else:
                q_t = torch.zeros(c.hidden)
                alpha_d_eff = float(m.alpha_d)
                alpha_s_eff = float(m.alpha_s)
                beta_g_eff = 0.0
                gamma_q_eff = 0.0

            s_d = torch.tanh(alpha_d_eff * I_total + gamma_q_eff * q_t)
            d_new = m.d_decay * d + (1.0 - m.d_decay) * s_d
            g_gate = torch.tanh(beta_g_eff * I_total)
            C_t = g_gate * d

            theta = c.threshold + c.adaptation_strength * a
            u = c.leak * v + alpha_s_eff * I_total + m.beta_d * C_t
            z = WordTapeSurrogate.apply(u, theta, torch.as_tensor(obs[2]))
            v = u - z * theta
            a = c.adaptation_decay * a + z
            f = c.feature_decay * f + (1.0 - c.feature_decay) * z
            d = d_new

        errors = [
            float(np.max(np.abs(s.detach().numpy() - ref_arr)))
            for s, ref_arr in zip((v, a, z, f, d), (m.core.voltage, m.core.adaptation, m.core.spikes, m.core.features, m.d))
        ]
        if max(errors[0], errors[1], errors[2], errors[4]) > 1e-4 or errors[3] > 1e-6:
            raise ValueError(f"DC-ALIF BPTT forward parity error: {errors}")
        self.max_graph_state_error = max(self.max_graph_state_error, errors[0], errors[1], errors[2], errors[4])
        self.max_graph_feature_error = max(self.max_graph_feature_error, errors[3])

        scale = torch.as_tensor(np.maximum(np.sqrt(m.head.m2 / (m.head.count - 1)), 0.05).astype(np.float32))
        raw = (f - torch.as_tensor(m.head.mean.astype(np.float32))) / scale
        transformed = raw.detach().clamp(-5.0, 5.0) + (raw - raw.detach()) * (raw.detach().abs() < 5.0)
        logits = torch.as_tensor(m.head.weights) @ transformed + torch.as_tensor(m.head.bias)
        logits = torch.where(torch.as_tensor(m.output_mask), logits, torch.tensor(-float("inf")))
        loss = torch.logsumexp(logits, 0) - logits[target]
        loss.backward()
        direction = np.concatenate([-p.grad.detach().numpy().copy() for p in (wi, wr)], axis=1)
        if not np.isfinite(direction).all():
            raise ValueError("Nonfinite matched DC-ALIF BPTT direction")
        self.backward_passes += 1
        self.surrogate_forward_ticks += len(sequence)
        return direction

    def learn(self, sequence: np.ndarray, target: int) -> float:
        m = self.reference
        if m.condition == "frozen":
            return m.learn(sequence, target)
        x = m._sequence(sequence, m.core.config.inputs)
        if not m.head.frozen or type(target) is not int or not 0 <= target < m.core.config.outputs:
            raise ValueError("Control supervision")
        snapshot = copy.deepcopy(self.__dict__)
        try:
            m.begin(False)
            states = []
            for row in x:
                m.step(row)
                states.append(
                    (
                        m.core.voltage.copy(),
                        m.core.adaptation.copy(),
                        m.core.spikes.copy(),
                        m.core.features.copy(),
                        m.d.copy(),
                    )
                )
            direction = self._direction(x, states, target)
            loss = m.accumulate(target)
            m.bank[:] = direction
            m.finish()
            return loss
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            raise

    def features(self, sequence: np.ndarray, reset_each_tick: bool = False) -> np.ndarray:
        return self.reference.features(sequence, reset_each_tick)

    def features_many(self, sequences: np.ndarray, reset_each_tick: bool = False) -> np.ndarray:
        return self.reference.features_many(sequences, reset_each_tick)

    def predict(self, sequence: np.ndarray) -> int:
        return self.reference.predict(sequence)

    def dumps(self) -> bytes:
        raw = self.reference.dumps()
        meta = {
            "format": "scarx-dc-alif-bptt-control-v1",
            "control_only": True,
            "reference_sha256": hashlib.sha256(raw).hexdigest(),
            **{
                n: getattr(self, n)
                for n in ("backward_passes", "surrogate_forward_ticks", "max_graph_state_error", "max_graph_feature_error")
            },
        }
        meta["metadata_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            z.writestr(zipfile.ZipInfo("reference.dc_alif"), raw)
            z.writestr(zipfile.ZipInfo("control.json"), json.dumps(meta))
        return out.getvalue()

    @classmethod
    def loads(cls, payload: bytes) -> "DCALIFBPTTControl":
        if len(payload) > 5_000_000:
            raise ValueError("Matched DC-ALIF control bound")
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if (
                set(z.namelist()) != {"reference.dc_alif", "control.json"}
                or len(z.namelist()) != 2
                or sum(i.file_size for i in z.infolist()) > 5_000_000
                or z.getinfo("control.json").file_size > 65536
            ):
                raise ValueError("Matched DC-ALIF control members")
            raw = z.read("reference.dc_alif")
            meta = json.loads(z.read("control.json"))
        checksum = meta.pop("metadata_sha256", None)
        if (
            meta.get("format") != "scarx-dc-alif-bptt-control-v1"
            or meta.get("control_only") is not True
            or hashlib.sha256(raw).hexdigest() != meta.get("reference_sha256")
            or hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest() != checksum
        ):
            raise ValueError("Matched DC-ALIF control checksum")
        m = cls(DCALIFEligibility.loads(raw))
        for name in ("backward_passes", "surrogate_forward_ticks"):
            if type(meta[name]) is not int or meta[name] < 0:
                raise ValueError("Matched DC-ALIF control counter")
            setattr(m, name, meta[name])
        for name in ("max_graph_state_error", "max_graph_feature_error"):
            v = meta[name]
            if type(v) not in (int, float) or not np.isfinite(v) or v < 0:
                raise ValueError("Matched DC-ALIF parity statistic")
            setattr(m, name, v)
        if m.backward_passes != m.reference.hidden_updates:
            raise ValueError("Matched DC-ALIF control update consistency")
        return m


def dc_alif_finite_difference_check(
    model: DCALIFEligibility,
    sequence: np.ndarray,
    neurons: tuple[int, ...] = (0, 7, 19, 37),
    epsilon: float = 1e-5,
) -> dict[str, Any]:
    """Independent anchored-surrogate finite-difference check across 64 synapses (4 neurons x 16 synapses)."""
    c = model.core.config
    clone = DCALIFEligibility.loads(model.dumps())
    clone.begin(False)
    presyn_in = []
    presyn_rec = []
    spikes = []
    for x in sequence:
        presyn_in.append(np.asarray(x, dtype=np.float64).copy())
        presyn_rec.append(clone.core.spikes[clone.core.indices].astype(np.float64).copy())
        clone.step(x)
        spikes.append(clone.core.spikes.astype(np.float64).copy())

    model.begin(True)
    for x in sequence:
        model.step(x)

    for name in ("voltage", "adaptation", "spikes", "features"):
        np.testing.assert_array_equal(getattr(model.core, name), getattr(clone.core, name))
    np.testing.assert_array_equal(model.d, clone.d)

    P64 = model.P.astype(np.float64)
    n_pairs = c.inputs // 2

    def scalar_anchored_forward(
        weights: np.ndarray,
        j: int,
        baseline_weights: np.ndarray,
    ) -> tuple[float, float]:
        def run_pass(w_vec: np.ndarray, margins_anchor: list[float] | None) -> tuple[float, float, list[float]]:
            w_in_j = w_vec[: c.inputs]
            w_rec_j = w_vec[c.inputs :]
            w_hat_j = w_in_j @ P64 if model.condition != "no_gate" else w_in_j
            v = a = f = d = 0.0
            margins: list[float] = []
            for t_idx in range(len(sequence)):
                x_t = presyn_in[t_idx]
                r_t = presyn_rec[t_idx][j]
                z_obs = float(spikes[t_idx][j])
                u_in = w_hat_j * x_t
                I_in = float(np.sum(u_in))
                I_rec = float(np.dot(w_rec_j, r_t))
                I_total = I_in + I_rec

                if n_pairs > 0 and model.condition != "no_gate":
                    s_pairs = [u_in[2 * m] + u_in[2 * m + 1] for m in range(n_pairs)]
                    sq_pairs = sum(s * s for s in s_pairs)
                    if 2 * n_pairs < c.inputs:
                        sq_pairs += u_in[-1] ** 2
                    q_t = 0.5 * (I_in * I_in - sq_pairs)
                    coact = compute_coact(x_t)
                    alpha_d_eff = float(model.alpha_d * ((1.0 - coact) ** 2))
                    alpha_s_eff = float(model.alpha_s * (1.0 - coact))
                    beta_g_eff = float(model.beta_g * (1.0 - coact))
                    gamma_q_eff = float(model.gamma_q)
                else:
                    q_t = 0.0
                    alpha_d_eff = float(model.alpha_d)
                    alpha_s_eff = float(model.alpha_s)
                    beta_g_eff = 0.0
                    gamma_q_eff = 0.0

                s_d = float(np.tanh(alpha_d_eff * I_total + gamma_q_eff * q_t))
                d_new = model.d_decay * d + (1.0 - model.d_decay) * s_d
                g_gate = 1.0 if model.condition == "no_gate" else float(np.tanh(beta_g_eff * I_total))
                C_t = g_gate * d
                theta = c.threshold + c.adaptation_strength * a
                u = c.leak * v + alpha_s_eff * I_total + model.beta_d * C_t
                margin = (u - theta) / 0.4
                margins.append(margin)
                if margins_anchor is None:
                    local_z = z_obs
                else:
                    base_m = margins_anchor[t_idx]
                    act = 1.0 / (1.0 + np.exp(-np.clip(margin, -20.0, 20.0)))
                    anc = 1.0 / (1.0 + np.exp(-np.clip(base_m, -20.0, 20.0)))
                    local_z = z_obs + act - anc
                v = u - local_z * theta
                a = c.adaptation_decay * a + local_z
                f = c.feature_decay * f + (1.0 - c.feature_decay) * local_z
                d = d_new
            return f, d, margins

        _, _, base_margins = run_pass(baseline_weights, None)
        f_val, d_val, _ = run_pass(weights, base_margins)
        return f_val, d_val

    errors_f = []
    errors_d = []
    rows = []
    for j in neurons:
        weights = np.concatenate((model.core.w_in[j], model.core.w_rec[j])).astype(np.float64)
        num_f = []
        num_d = []
        for i in range(len(weights)):
            plus = weights.copy()
            minus = weights.copy()
            plus[i] += epsilon
            minus[i] -= epsilon
            fp, dp = scalar_anchored_forward(plus, j, weights)
            fm, dm = scalar_anchored_forward(minus, j, weights)
            num_f.append((fp - fm) / (2.0 * epsilon))
            num_d.append((dp - dm) / (2.0 * epsilon))
        num_f_arr = np.asarray(num_f)
        num_d_arr = np.asarray(num_d)
        trace_f = model.e_f[j].astype(np.float64)
        trace_d = model.e_d[j].astype(np.float64)
        np.testing.assert_allclose(trace_f, num_f_arr, atol=1e-5, rtol=1e-3)
        np.testing.assert_allclose(trace_d, num_d_arr, atol=1e-5, rtol=1e-3)
        errors_f.extend(np.abs(trace_f - num_f_arr).tolist())
        errors_d.extend(np.abs(trace_d - num_d_arr).tolist())
        rows.append(
            {
                "neuron": int(j),
                "synapses": len(weights),
                "max_abs_error_e_f": float(np.max(np.abs(trace_f - num_f_arr))),
                "max_abs_error_e_d": float(np.max(np.abs(trace_d - num_d_arr))),
            }
        )
    return {
        "finite_difference_synapses": len(errors_f),
        "max_absolute_error_e_f": float(max(errors_f)),
        "max_absolute_error_e_d": float(max(errors_d)),
        "neuron_rows": rows,
        "forward_state_exact": True,
        "oracle_scope": "Anchored smooth own-neuron surrogate (e_f and e_d), fixed presynaptic tape; NOT finite derivative of hard spikes.",
    }
