"""SCAR-X Round 7 RESEARCH ONLY: Event-Windowed Homeostatic Eligibility (EW-HE),
Coincidence-Gated Multiscale Homeostatic Eligibility (CG-MHE), and
Context-Anchored Metaplastic Synaptic Consolidation (CA-MSC).

Operates strictly on the unchanged 1-compartment ALIF SpikingCore (H=48, fan_in=8,
hard spikes) using own-neuron forward sensitivities, online homeostatic feature
standardization, and local homeostatic synaptic trace energy.
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

from ..neural.core import Config, SpikingCore
from ..learning.replay import ReplayBuffer
from .readout_eligibility import spike_slope
from .tasks21 import representation, ridge_probe
from .binding_audit21 import strict_binding


class EventWindowedHomeostaticEligibility:
    """Purely local hidden + readout synaptic plasticity wrapper around SpikingCore(H=48)."""

    conditions = (
        "frozen_hidden",
        "zero_signal",
        "shuffled_signal",
        "r35_baseline",
        "learned_ew_he",
        "learned_cg_mhe",
        "ew_he_ca_msc",
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
        "w_in_anchor",
        "w_rec_anchor",
        "w_out_anchor",
        "bias_anchor",
        "omega_hid",
        "omega_out",
        "gamma_e_vec",
    )

    def __init__(self, config: Config, condition: str = "learned_ew_he") -> None:
        if condition not in self.conditions:
            raise ValueError(f"Unsupported condition: {condition}")
        if config.neuron != "alif" or config.hidden != 48 or config.fan_in != 8:
            raise ValueError("Round 7 requires ALIF SpikingCore with hidden=48, fan_in=8")
        config.validate()
        self.config = config
        self.condition = condition
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

        self.w_in_anchor = self.core.w_in.copy()
        self.w_rec_anchor = self.core.w_rec.copy()
        self.w_out_anchor = self.core.w_out.copy()
        self.bias_anchor = self.core.bias.copy()

        self.omega_hid = np.zeros((h, d_tot), dtype=np.float32)
        self.omega_out = np.zeros_like(self.core.w_out)
        self.gamma_e_vec = np.linspace(0.15, 0.55, h, dtype=np.float32)[:, None]

        self.signal_rng = np.random.default_rng(config.seed + 35100000)
        self.calibrated = False
        self.consolidate_active = False
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

        if track and self.condition != "frozen_hidden":
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

            if self.condition == "learned_cg_mhe":
                m_carry = np.abs(u_carry) + c.threshold * c.adaptation_strength * a_old.mean(axis=1)
                m_inst = np.abs(i_in)
                chi = (2.0 * m_carry * m_inst) / (m_carry * m_carry + m_inst * m_inst + 0.05)
                z_e = z_e * (0.25 + 0.75 * chi[:, None])

        self.core.step(row, track=False)

        if track and self.condition != "frozen_hidden":
            self.e_v[:] = u_e - theta[:, None] * z_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
            self.e_a[:] = c.adaptation_decay * a_old + z_e
            if self.condition == "r35_baseline":
                self.e_f[:] = 0.92 * self.e_f + 0.08 * z_e
            elif self.condition == "learned_cg_mhe":
                self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e
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

        # Compute own-neuron hidden credit signal BEFORE updating readout weights w_out
        # to prevent post-update readout self-alignment bias (+eta_out * ||err||^2 * f_hat).
        pre_update_signal = (self.core.w_out.T @ err) * (np.abs(raw) < 5.0) / scale

        grad_out = err[:, None] * f_hat[None, :]
        pull_out = (
            0.001 * self.omega_out * (self.w_out_anchor - self.core.w_out)
            if (self.condition == "ew_he_ca_msc" and self.consolidate_active)
            else 0.0
        )
        self.core.w_out += np.clip(0.04 * grad_out + pull_out, -0.025, 0.025)
        self.core.bias += np.clip(0.04 * err, -0.025, 0.025)
        np.clip(self.core.w_out, -3.5, 3.5, out=self.core.w_out)
        np.clip(self.core.bias, -3.5, 3.5, out=self.core.bias)

        # Matched RNG consumption across all conditions
        perm = self.signal_rng.permutation(self.config.hidden)

        if self.condition not in ("frozen_hidden", "zero_signal"):
            signal = pre_update_signal
            if self.condition == "shuffled_signal":
                signal = signal[perm]
            direction = signal[:, None] * self.e_f

            if self.condition == "ew_he_ca_msc":
                active_syn = (np.abs(self.e_f) > 1e-5).astype(np.float32)
            else:
                active_syn = np.ones_like(self.e_f)

            self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
            self.syn_energy = 0.99 * self.syn_energy + 0.01 * (direction * direction)
            rms = np.sqrt(self.syn_energy) + 1e-4
            norm_step = (self.syn_trace * active_syn) / rms

            if self.condition == "ew_he_ca_msc" and self.consolidate_active:
                w_comb = np.concatenate((self.core.w_in, self.core.w_rec), axis=1)
                w_anc = np.concatenate((self.w_in_anchor, self.w_rec_anchor), axis=1)
                pull_hid = 0.0015 * self.omega_hid * (1.0 - active_syn) * (w_anc - w_comb)
            else:
                pull_hid = 0.0

            lr_hid = 0.002 if self.condition == "r35_baseline" else 0.008
            clip_hid = 0.002 if self.condition == "r35_baseline" else 0.01
            step = np.clip(lr_hid * norm_step + pull_hid, -clip_hid, clip_hid)

            d = self.config.inputs
            self.core.w_in += step[:, :d]
            self.core.w_rec += step[:, d:]
            np.clip(self.core.w_in, -3.0, 3.0, out=self.core.w_in)
            np.clip(self.core.w_rec, -3.0, 3.0, out=self.core.w_rec)

            if self.condition == "ew_he_ca_msc":
                self.omega_hid = np.minimum(5.0, 0.998 * self.omega_hid + 0.005 * np.abs(norm_step))
                self.omega_out = np.minimum(5.0, 0.998 * self.omega_out + 0.005 * np.abs(grad_out))

            self.hidden_updates += 1

        self.core.updates += 1
        self.episodes_learned += 1
        return loss

    def snapshot_consolidation_anchors(self) -> None:
        self.w_in_anchor[:] = self.core.w_in
        self.w_rec_anchor[:] = self.core.w_rec
        self.w_out_anchor[:] = self.core.w_out
        self.bias_anchor[:] = self.core.bias
        self.consolidate_active = True

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

    def memory_bytes(self) -> int:
        return self.core.memory_bytes() + sum(getattr(self, name).nbytes for name in self.state_array_names)

    def dumps(self) -> bytes:
        core_bytes = self.core.dumps()
        state_buf = io.BytesIO()
        np.savez(state_buf, **{name: getattr(self, name) for name in self.state_array_names})
        state_bytes = state_buf.getvalue()
        meta = {
            "format": "scarx-r7-ew-he-v1",
            "condition": self.condition,
            "calibrated": self.calibrated,
            "consolidate_active": self.consolidate_active,
            "episodes_learned": self.episodes_learned,
            "hidden_updates": self.hidden_updates,
            "total_ticks": self.total_ticks,
            "signal_rng": self.signal_rng.bit_generator.state,
            "core_sha256": hashlib.sha256(core_bytes).hexdigest(),
            "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        }
        meta["metadata_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("core.npz", core_bytes)
            zf.writestr("state.npz", state_bytes)
            zf.writestr("metadata.json", json.dumps(meta))
        raw = out.getvalue()
        if len(raw) > 4_000_000:
            raise MemoryError("Checkpoint exceeds 4MB bound")
        return raw

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.dumps()
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def loads(cls, payload: bytes) -> "EventWindowedHomeostaticEligibility":
        if len(payload) > 4_000_000:
            raise ValueError("Checkpoint exceeds 4MB bound")
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            if set(zf.namelist()) != {"core.npz", "state.npz", "metadata.json"}:
                raise ValueError("Invalid checkpoint archive members")
            core_bytes = zf.read("core.npz")
            state_bytes = zf.read("state.npz")
            meta = json.loads(zf.read("metadata.json"))
        expected_meta_hash = meta.pop("metadata_sha256", None)
        if (
            meta.get("format") != "scarx-r7-ew-he-v1"
            or hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest() != expected_meta_hash
            or hashlib.sha256(core_bytes).hexdigest() != meta.get("core_sha256")
            or hashlib.sha256(state_bytes).hexdigest() != meta.get("state_sha256")
        ):
            raise ValueError("Checkpoint checksum verification failed")
        loaded_core = SpikingCore.loads(core_bytes)
        obj = cls(loaded_core.config, meta["condition"])
        obj.core = loaded_core
        with np.load(io.BytesIO(state_bytes), allow_pickle=False) as st:
            if set(st.files) != set(cls.state_array_names):
                raise ValueError("State array mismatch")
            for name in cls.state_array_names:
                arr = st[name]
                tmpl = getattr(obj, name)
                if arr.shape != tmpl.shape or arr.dtype != tmpl.dtype or not np.isfinite(arr).all():
                    raise ValueError(f"Invalid state array {name}")
                tmpl[:] = arr
        obj.calibrated = bool(meta["calibrated"])
        obj.consolidate_active = bool(meta["consolidate_active"])
        obj.episodes_learned = int(meta["episodes_learned"])
        obj.hidden_updates = int(meta["hidden_updates"])
        obj.total_ticks = int(meta["total_ticks"])
        obj.signal_rng.bit_generator.state = meta["signal_rng"]
        return obj

    @classmethod
    def load(cls, path: str | Path) -> "EventWindowedHomeostaticEligibility":
        p = Path(path)
        if p.stat().st_size > 4_000_000:
            raise ValueError("Oversized checkpoint file")
        return cls.loads(p.read_bytes())


# ---------------------------------------------------------------------------
# P1 Task Generators (In-Distribution & OOD Timing/Distractor Shifts)
# ---------------------------------------------------------------------------

def strict_binding_task(seed: int, count: int, ood: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if not ood:
        return strict_binding(seed, count)
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 2
    x = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    for i in range(count):
        query = int(rng.integers(2))
        a = int(y[i]) ^ query
        values = (a, 1 - a)
        order = rng.permutation(2)
        positions = (0, 7)
        for position, key in zip(positions, order):
            x[i, position, key] += 1.5
            x[i, position, 2 + values[key]] += 1.5
        x[i, 10, 4 + query] += 1.5
        x[i, 11, 7] += 1.0
    order = rng.permutation(count)
    return x[order], y[order].astype(np.int32)


def context_selective_rule_task(seed: int, count: int, ood: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """3-tick 3-way parity rule switching (y = a ^ b ^ c):
    - Tick p1 in {1,2} (0 if ood): Feature a in {0,1} on ch 0..1
    - Tick p2 in {4,5,6} (7 if ood): Feature b in {0,1} on ch 2..3
    - Tick 10: Context rule cue c in {0,1} on ch 4..5 (c=0: XOR rule y=a^b; c=1: XNOR rule y=1-(a^b))
    - Tick 11: Readout marker on ch 7
    All 1st-order and 2nd-order marginals have zero correlation with y.
    """
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 2
    x = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    for i in range(count):
        a = int(rng.integers(2))
        b = int(rng.integers(2))
        c = a ^ b ^ int(y[i])
        p1 = 0 if ood else int(rng.integers(1, 3))
        p2 = 7 if ood else int(rng.integers(4, 7))
        x[i, p1, 0 + a] += 1.5
        x[i, p2, 2 + b] += 1.5
        x[i, 10, 4 + c] += 1.5
        x[i, 11, 7] += 1.0
    order = rng.permutation(count)
    return x[order], y[order].astype(np.int32)


def distractor_binding_task(seed: int, count: int, ood: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 2
    sigma = 0.14 if ood else 0.10
    x = rng.normal(0, sigma, (count, 12, 8)).astype(np.float32)
    for i in range(count):
        query = int(rng.integers(2))
        a = int(y[i]) ^ query
        values = (a, 1 - a)
        order = rng.permutation(2)
        positions = (0, 7) if ood else (int(rng.integers(1, 3)), int(rng.integers(4, 6)))
        for position, key in zip(positions, order):
            x[i, position, key] += 1.5
            x[i, position, 2 + values[key]] += 1.5
        for dt in (3, 7, 8):
            if dt not in positions:
                x[i, dt, 6] += float(rng.choice([-1.2, 1.2]))
        x[i, 10, 4 + query] += 1.5
        x[i, 11, 7] += 1.0
    order = rng.permutation(count)
    return x[order], y[order].astype(np.int32)


def generate_p1_task(task: str, seed: int, count: int, ood: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if task == "strict_binding":
        return strict_binding_task(seed, count, ood=ood)
    if task == "context_selective_rule":
        return context_selective_rule_task(seed, count, ood=ood)
    if task == "distractor_binding":
        return distractor_binding_task(seed, count, ood=ood)
    if task in ("xor", "order"):
        return representation(task, seed, count, ood=ood)
    raise ValueError(f"Unknown P1 task: {task}")


# ---------------------------------------------------------------------------
# Representation Geometry Metrics
# ---------------------------------------------------------------------------

def compute_geometry_metrics(f_mat: np.ndarray, y: np.ndarray) -> dict[str, float | list[float]]:
    centered = f_mat - f_mat.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(1, len(f_mat) - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals[::-1], 0.0)
    total = float(np.sum(eigvals))
    pr = (total * total) / float(np.sum(eigvals * eigvals) + 1e-12) if total > 0 else 0.0
    svals = [round(float(x), 4) for x in np.sqrt(eigvals[:5])]

    f0 = f_mat[y == 0]
    f1 = f_mat[y == 1]
    diff = f1.mean(axis=0) - f0.mean(axis=0)
    between = float(np.dot(diff, diff))
    within = float(np.mean(np.var(f0, axis=0) + np.var(f1, axis=0))) + 1e-8
    fdr = between / within
    return {
        "participation_ratio": float(pr),
        "fisher_discriminant_ratio": float(fdr),
        "top5_singular_values": svals,
    }


# ---------------------------------------------------------------------------
# P1 & P2 Evaluation Runners
# ---------------------------------------------------------------------------

P1_TASKS = (
    "strict_binding",
    "context_selective_rule",
    "distractor_binding",
    "xor",
    "order",
)

P1_CONDITIONS = (
    "initial_hidden",
    "frozen_hidden",
    "r35_baseline",
    "zero_signal",
    "shuffled_signal",
    "learned_ew_he",
    "learned_cg_mhe",
    "causal_revert",
)


def run_p1_task_seed(
    task: str,
    seed: int,
    n_cal: int = 128,
    n_train: int = 1024,
    n_fit: int = 256,
    n_test: int = 512,
) -> dict[str, dict[str, float | list[float]]]:
    cal_x, _ = generate_p1_task(task, seed + 100, n_cal, ood=False)
    tr_x, tr_y = generate_p1_task(task, seed + 200, n_train, ood=False)
    fit_x, fit_y = generate_p1_task(task, seed + 300, n_fit, ood=False)
    id_x, id_y = generate_p1_task(task, seed + 400, n_test, ood=False)
    ood_x, ood_y = generate_p1_task(task, seed + 500, n_test, ood=True)

    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0.0)
    results: dict[str, dict[str, float | list[float]]] = {}

    # 1. initial_hidden (before any training)
    init_model = EventWindowedHomeostaticEligibility(cfg, "frozen_hidden")
    init_model.calibrate(cal_x)
    ff_init = init_model.features_many(fit_x)
    tf_init = init_model.features_many(id_x)
    of_init = init_model.features_many(ood_x)
    geom_init = compute_geometry_metrics(tf_init, id_y)
    results["initial_hidden"] = {
        "id_direct_acc": float(np.mean(init_model.predict_many(id_x) == id_y)),
        "ood_direct_acc": float(np.mean(init_model.predict_many(ood_x) == ood_y)),
        "id_probe_acc": ridge_probe(ff_init, fit_y, tf_init, id_y),
        "ood_probe_acc": ridge_probe(ff_init, fit_y, of_init, ood_y),
        **geom_init,
    }

    trained_ew_he: EventWindowedHomeostaticEligibility | None = None

    for cond in ("frozen_hidden", "r35_baseline", "zero_signal", "shuffled_signal", "learned_ew_he", "learned_cg_mhe"):
        model = EventWindowedHomeostaticEligibility(cfg, cond)
        model.calibrate(cal_x)
        for seq, y in zip(tr_x, tr_y):
            model.learn_episode(seq, int(y), stat_rate=0.01)

        ff = model.features_many(fit_x)
        tf = model.features_many(id_x)
        of = model.features_many(ood_x)
        geom = compute_geometry_metrics(tf, id_y)
        results[cond] = {
            "id_direct_acc": float(np.mean(model.predict_many(id_x) == id_y)),
            "ood_direct_acc": float(np.mean(model.predict_many(ood_x) == ood_y)),
            "id_probe_acc": ridge_probe(ff, fit_y, tf, id_y),
            "ood_probe_acc": ridge_probe(ff, fit_y, of, ood_y),
            "array_bytes": int(model.memory_bytes()),
            **geom,
        }
        if cond == "learned_ew_he":
            trained_ew_he = model

    # 8. causal_revert: keep learned readout & homeostatic stats from learned_ew_he, revert hidden weights to initial
    assert trained_ew_he is not None
    reverted = copy.deepcopy(trained_ew_he)
    reverted.core.w_in[:] = reverted.w_in_0
    reverted.core.w_rec[:] = reverted.w_rec_0
    tf_rev = reverted.features_many(id_x)
    of_rev = reverted.features_many(ood_x)
    id_rev_acc = float(np.mean(reverted.predict_many(id_x) == id_y))
    ood_rev_acc = float(np.mean(reverted.predict_many(ood_x) == ood_y))
    results["causal_revert"] = {
        "id_direct_acc": id_rev_acc,
        "ood_direct_acc": ood_rev_acc,
        "id_probe_acc": ridge_probe(ff_init, fit_y, tf_rev, id_y),
        "ood_probe_acc": ridge_probe(ff_init, fit_y, of_rev, ood_y),
        "id_causal_drop": float(results["learned_ew_he"]["id_direct_acc"]) - id_rev_acc,
        "ood_causal_drop": float(results["learned_ew_he"]["ood_direct_acc"]) - ood_rev_acc,
    }
    return results


def wrap_skill_pulse(x_batch: np.ndarray, skill_id: int, n_skills: int = 3) -> np.ndarray:
    n, t, d = x_batch.shape
    out = np.zeros((n, t, d + n_skills), dtype=np.float32)
    out[:, :, :d] = x_batch
    out[:, 11, d + skill_id] = 0.8
    return out


def run_p2_continual_seed(seed: int, mode: str = "ew_he_ca_msc") -> dict[str, float | list[list[float]]]:
    """Sequential 3-skill continual learning (order -> xor -> strict_binding) in a single H=48 SpikingCore."""
    n_skills = 3
    d_in = 8 + n_skills
    cfg = Config(inputs=d_in, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0.0)
    cond = "ew_he_ca_msc" if mode == "ew_he_ca_msc" else "learned_ew_he"
    model = EventWindowedHomeostaticEligibility(cfg, cond)

    cal_batches = [
        wrap_skill_pulse(representation("order", seed + 10, 48)[0], 0, n_skills),
        wrap_skill_pulse(representation("xor", seed + 20, 48)[0], 1, n_skills),
        wrap_skill_pulse(strict_binding(seed + 30, 48)[0], 2, n_skills),
    ]
    model.calibrate(np.concatenate(cal_batches, axis=0))

    train_sets = [
        ("order", wrap_skill_pulse(representation("order", seed + 100, 768)[0], 0, n_skills), representation("order", seed + 100, 768)[1]),
        ("xor", wrap_skill_pulse(representation("xor", seed + 200, 1024)[0], 1, n_skills), representation("xor", seed + 200, 1024)[1]),
        ("strict_binding", wrap_skill_pulse(strict_binding(seed + 300, 1024)[0], 2, n_skills), strict_binding(seed + 300, 1024)[1]),
    ]
    test_sets = [
        ("order", wrap_skill_pulse(representation("order", seed + 1100, 256)[0], 0, n_skills), representation("order", seed + 1100, 256)[1]),
        ("xor", wrap_skill_pulse(representation("xor", seed + 1200, 256)[0], 1, n_skills), representation("xor", seed + 1200, 256)[1]),
        ("strict_binding", wrap_skill_pulse(strict_binding(seed + 1300, 256)[0], 2, n_skills), strict_binding(seed + 1300, 256)[1]),
    ]

    policy = "balanced" if mode == "ew_he_ca_msc" else "none"
    buffer = ReplayBuffer(256, 12, d_in, policy, seed + 7000)

    matrix = np.zeros((n_skills, n_skills), dtype=np.float64)
    for s_idx, (_, trx, try_) in enumerate(train_sets):
        for seq, y in zip(trx, try_):
            model.learn_episode(seq, int(y), stat_rate=0.005)
            if mode == "ew_he_ca_msc" and buffer.size > 0:
                for r_seq, r_y in buffer.sample(1):
                    model.learn_episode(r_seq, int(r_y), stat_rate=0.005)
        if mode == "ew_he_ca_msc":
            model.snapshot_consolidation_anchors()
            for seq, y in zip(trx, try_):
                buffer.add(seq, int(y), s_idx)

        for eval_idx in range(s_idx + 1):
            _, tx, ty = test_sets[eval_idx]
            matrix[s_idx, eval_idx] = float(np.mean(model.predict_many(tx) == ty))

    diag = np.diag(matrix)
    final = matrix[-1, :]
    forg = [max(matrix[t, j] for t in range(j, n_skills)) - final[j] for j in range(n_skills)]
    return {
        "acquisition_accuracy": float(diag.mean()),
        "final_accuracy": float(final.mean()),
        "mean_forgetting": float(np.mean(forg)),
        "worst_forgetting": float(np.max(forg)),
        "stage_matrix": [[round(float(x), 4) for x in row] for row in matrix],
        "array_bytes": int(model.memory_bytes() + buffer.x.nbytes + buffer.y.nbytes + buffer.context.nbytes),
    }
