"""SCAR-X Round 9c Causal Diagnostics & Multi-Layer Information Tracing.

Implements:
1. Multi-layer state tracing across 6 stages:
   S1: Input sequence (x)
   S2: Pre-reset membrane voltage (u)
   S3: Post-reset membrane voltage and spikes (v, z)
   S4: Adaptation state (a)
   S5: Filtered eligibility traces (e_f)
   S6: Readout features (f) and direct predictions
2. Non-linear decodability probes (Linear Ridge, Capacity-Controlled MLP, Oracle, Raw Input baseline)
   to classify:
   - INFORMATION_ABSENT
   - INFORMATION_PRESENT_BUT_NOT_LINEALLY_DECODABLE
   - READOUT_FAILURE
   - TRAINING_FAILURE
   - UNDETERMINED
3. Causal counterfactual state restorations:
   A: Voltage State Restoration (VSR)
   B: Spike State Restoration (SSR)
   C: Adaptation State Restoration (ASR)
   A+B+C: Full Soma State Restoration (FSR)
   E: Readout Counterfactual (RC)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from scarx.neural.core import Config
from scarx.neural.r9b_factorial_eprop import FactorialMTSGEProp
from scarx.evaluation.tasks21 import ridge_probe
from scarx.evaluation.r9c_paired_dataset import PairedBatch


@dataclass
class TimestepTrace:
    t: int
    x: np.ndarray        # (inputs,)
    u_pre: np.ndarray    # (hidden,)
    v_post: np.ndarray   # (hidden,)
    z: np.ndarray        # (hidden,)
    a: np.ndarray        # (hidden,)
    e_f_norm: np.ndarray # (hidden,)
    f: np.ndarray        # (hidden,)


@dataclass
class EpisodeTrace:
    sample_id: int
    label: int
    timesteps: list[TimestepTrace]
    final_features: np.ndarray
    prediction: int
    logits: np.ndarray


class ControlledNonLinearProbe:
    """Capacity-controlled small MLP probe (hidden=16, ReLU, L2 regularized)."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, lr: float = 0.05, l2: float = 0.01, epochs: int = 80):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n, d = x.shape
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0) + 1e-4
        x_norm = (x - self.mean) / self.std

        rng = np.random.default_rng(42)
        self.w1 = rng.normal(0, np.sqrt(2.0 / d), (d, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(0, np.sqrt(2.0 / self.hidden_dim), (self.hidden_dim, 2)).astype(np.float32)
        self.b2 = np.zeros(2, dtype=np.float32)

        y_onehot = np.eye(2)[y].astype(np.float32)

        for _ in range(self.epochs):
            # Forward
            h_pre = x_norm @ self.w1 + self.b1
            h = np.maximum(0.0, h_pre)
            logits = h @ self.w2 + self.b2
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Backward
            d_logits = (probs - y_onehot) / n
            dw2 = h.T @ d_logits + self.l2 * self.w2
            db2 = np.sum(d_logits, axis=0)
            dh = d_logits @ self.w2.T
            dh_pre = dh * (h_pre > 0)
            dw1 = x_norm.T @ dh_pre + self.l2 * self.w1
            db1 = np.sum(dh_pre, axis=0)

            # Gradient step with clipping
            self.w2 -= self.lr * np.clip(dw2, -1.0, 1.0)
            self.b2 -= self.lr * np.clip(db2, -1.0, 1.0)
            self.w1 -= self.lr * np.clip(dw1, -1.0, 1.0)
            self.b1 -= self.lr * np.clip(db1, -1.0, 1.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_norm = (x - self.mean) / self.std
        h = np.maximum(0.0, x_norm @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        return np.argmax(logits, axis=1).astype(np.int32)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(x)
        return float(np.mean(preds == y))


def trace_episode(model: FactorialMTSGEProp, sequence: np.ndarray, sample_id: int, label: int) -> EpisodeTrace:
    """Executes a sequence on a model, recording per-step state variables."""
    model.reset_episode()
    c = model.config
    timesteps: list[TimestepTrace] = []

    for t, row in enumerate(sequence):
        # 1. Input x
        x_row = np.asarray(row, dtype=np.float32)

        # 2. Compute pre-reset voltage u_pre
        old_z = model.core.spikes
        i_in = model.core.w_in @ x_row
        i_rec = np.sum(model.core.w_rec * old_z[model.core.indices], axis=1)
        u_pre = c.leak * model.core.voltage + i_rec + i_in

        # 3. Model forward step
        model.step(x_row, track=True)

        # 4. Post-reset voltage, spikes, adaptation, traces, features
        v_post = model.core.voltage.copy()
        z_post = model.core.spikes.copy()
        a_post = model.core.adaptation.copy()
        e_f_norm = np.linalg.norm(model.e_f, axis=1)
        f_post = model.core.features.copy()

        timesteps.append(
            TimestepTrace(
                t=t,
                x=x_row.copy(),
                u_pre=u_pre.copy(),
                v_post=v_post,
                z=z_post,
                a=a_post,
                e_f_norm=e_f_norm,
                f=f_post,
            )
        )

    final_f = model.core.features.copy()
    f_hat, _, _ = model.standardized_features(final_f)
    logits = model.core.w_out @ f_hat + model.core.bias
    prediction = int(np.argmax(logits))

    return EpisodeTrace(
        sample_id=sample_id,
        label=label,
        timesteps=timesteps,
        final_features=final_f,
        prediction=prediction,
        logits=logits,
    )


def run_causal_restoration_experiment(
    model: FactorialMTSGEProp,
    clean_seqs: np.ndarray,
    noisy_seqs: np.ndarray,
    labels: np.ndarray,
    distractor_ticks: tuple[int, ...] = (3, 7, 8),
) -> dict[str, Any]:
    """Runs counterfactual state restoration interventions on paired streams."""
    n = len(clean_seqs)
    clean_acc_count = 0
    noisy_acc_count = 0
    vsr_acc_count = 0   # Voltage State Restoration
    ssr_acc_count = 0   # Spike State Restoration
    asr_acc_count = 0   # Adaptation State Restoration
    fsr_acc_count = 0   # Full State Restoration (Voltage + Spikes + Adaptation)

    voltage_diffs_at_distractor = []
    extra_spikes_at_distractor = []
    adaptation_diffs_at_distractor = []

    for i in range(n):
        c_seq = clean_seqs[i]
        n_seq = noisy_seqs[i]
        y_true = int(labels[i])

        # 1. Clean run: collect clean state trajectory
        c_trace = trace_episode(model, c_seq, i, y_true)
        if c_trace.prediction == y_true:
            clean_acc_count += 1

        # 2. Noisy baseline run
        n_trace = trace_episode(model, n_seq, i, y_true)
        if n_trace.prediction == y_true:
            noisy_acc_count += 1

        # Record physical perturbations at distractor ticks
        for dt in distractor_ticks:
            if dt < len(c_trace.timesteps):
                v_diff = np.linalg.norm(n_trace.timesteps[dt].v_post - c_trace.timesteps[dt].v_post)
                voltage_diffs_at_distractor.append(v_diff)
                z_extra = np.sum(np.maximum(0.0, n_trace.timesteps[dt].z - c_trace.timesteps[dt].z))
                extra_spikes_at_distractor.append(z_extra)
                a_diff = np.linalg.norm(n_trace.timesteps[dt].a - c_trace.timesteps[dt].a)
                adaptation_diffs_at_distractor.append(a_diff)

        # 3. Intervention A: Voltage State Restoration (VSR)
        model.reset_episode()
        for t, row in enumerate(n_seq):
            model.step(row, track=True)
            if t in distractor_ticks:
                model.core.voltage[:] = c_trace.timesteps[t].v_post
        f_vsr = model.core.features
        p_vsr = model.probabilities_from_features(f_vsr)
        if int(np.argmax(p_vsr)) == y_true:
            vsr_acc_count += 1

        # 4. Intervention B: Spike State Restoration (SSR)
        model.reset_episode()
        for t, row in enumerate(n_seq):
            model.step(row, track=True)
            if t in distractor_ticks:
                model.core.spikes[:] = c_trace.timesteps[t].z
        f_ssr = model.core.features
        p_ssr = model.probabilities_from_features(f_ssr)
        if int(np.argmax(p_ssr)) == y_true:
            ssr_acc_count += 1

        # 5. Intervention C: Adaptation State Restoration (ASR)
        model.reset_episode()
        for t, row in enumerate(n_seq):
            model.step(row, track=True)
            if t in distractor_ticks:
                model.core.adaptation[:] = c_trace.timesteps[t].a
        f_asr = model.core.features
        p_asr = model.probabilities_from_features(f_asr)
        if int(np.argmax(p_asr)) == y_true:
            asr_acc_count += 1

        # 6. Intervention A+B+C: Full Soma State Restoration (FSR)
        model.reset_episode()
        for t, row in enumerate(n_seq):
            model.step(row, track=True)
            if t in distractor_ticks:
                model.core.voltage[:] = c_trace.timesteps[t].v_post
                model.core.spikes[:] = c_trace.timesteps[t].z
                model.core.adaptation[:] = c_trace.timesteps[t].a
        f_fsr = model.core.features
        p_fsr = model.probabilities_from_features(f_fsr)
        if int(np.argmax(p_fsr)) == y_true:
            fsr_acc_count += 1

    clean_acc = clean_acc_count / n
    noisy_acc = noisy_acc_count / n
    vsr_acc = vsr_acc_count / n
    ssr_acc = ssr_acc_count / n
    asr_acc = asr_acc_count / n
    fsr_acc = fsr_acc_count / n

    return {
        "n_samples": n,
        "clean_direct_acc": clean_acc,
        "noisy_direct_acc": noisy_acc,
        "vsr_direct_acc": vsr_acc,
        "ssr_direct_acc": ssr_acc,
        "asr_direct_acc": asr_acc,
        "fsr_direct_acc": fsr_acc,
        "recovery_vsr": (vsr_acc - noisy_acc) / max(1e-4, clean_acc - noisy_acc),
        "recovery_fsr": (fsr_acc - noisy_acc) / max(1e-4, clean_acc - noisy_acc),
        "mean_voltage_perturbation": float(np.mean(voltage_diffs_at_distractor)) if voltage_diffs_at_distractor else 0.0,
        "mean_extra_spikes": float(np.mean(extra_spikes_at_distractor)) if extra_spikes_at_distractor else 0.0,
        "mean_adaptation_perturbation": float(np.mean(adaptation_diffs_at_distractor)) if adaptation_diffs_at_distractor else 0.0,
    }


def evaluate_multi_layer_decodability(
    train_traces: list[EpisodeTrace],
    test_traces: list[EpisodeTrace],
    raw_train_x: np.ndarray,
    raw_test_x: np.ndarray,
) -> dict[str, Any]:
    """Evaluates linear vs non-linear decodability across computational layers."""
    train_y = np.array([t.label for t in train_traces], dtype=np.int32)
    test_y = np.array([t.label for t in test_traces], dtype=np.int32)

    # Stage 1: Raw inputs flattened
    s1_train = raw_train_x.reshape(len(raw_train_x), -1)
    s1_test = raw_test_x.reshape(len(raw_test_x), -1)

    # Stage 2: Membrane voltage trajectories across all timesteps
    s2_train = np.stack([np.concatenate([ts.u_pre for ts in tr.timesteps]) for tr in train_traces])
    s2_test = np.stack([np.concatenate([ts.u_pre for ts in tr.timesteps]) for tr in test_traces])

    # Stage 3: Spikes across all timesteps
    s3_train = np.stack([np.concatenate([ts.z for ts in tr.timesteps]) for tr in train_traces])
    s3_test = np.stack([np.concatenate([ts.z for ts in tr.timesteps]) for tr in test_traces])

    # Stage 4: Adaptation across all timesteps
    s4_train = np.stack([np.concatenate([ts.a for ts in tr.timesteps]) for tr in train_traces])
    s4_test = np.stack([np.concatenate([ts.a for ts in tr.timesteps]) for tr in test_traces])

    # Stage 5: Traces e_f norm across all timesteps
    s5_train = np.stack([np.concatenate([ts.e_f_norm for ts in tr.timesteps]) for tr in train_traces])
    s5_test = np.stack([np.concatenate([ts.e_f_norm for ts in tr.timesteps]) for tr in test_traces])

    # Stage 6: Final hidden features
    s6_train = np.stack([tr.final_features for tr in train_traces])
    s6_test = np.stack([tr.final_features for tr in test_traces])

    stages = {
        "S1_Input": (s1_train, s1_test),
        "S2_Voltage": (s2_train, s2_test),
        "S3_Spikes": (s3_train, s3_test),
        "S4_Adaptation": (s4_train, s4_test),
        "S5_Traces": (s5_train, s5_test),
        "S6_Features": (s6_train, s6_test),
    }

    results = {}
    for st_name, (tr_rep, te_rep) in stages.items():
        lin_score = ridge_probe(tr_rep, train_y, te_rep, test_y)
        mlp = ControlledNonLinearProbe(input_dim=tr_rep.shape[1], hidden_dim=16, lr=0.05, epochs=80)
        mlp.fit(tr_rep, train_y)
        nonlin_score = mlp.score(te_rep, test_y)

        results[st_name] = {
            "linear_probe": float(lin_score),
            "nonlinear_probe": float(nonlin_score),
            "probe_nonlin_gain": float(nonlin_score - lin_score),
        }

    # Direct model readout accuracy on test set
    direct_preds = np.array([tr.prediction for tr in test_traces], dtype=np.int32)
    direct_acc = float(np.mean(direct_preds == test_y))
    results["Direct_Readout"] = direct_acc

    # Diagnose information localization category
    s6_lin = results["S6_Features"]["linear_probe"]
    s6_nonlin = results["S6_Features"]["nonlinear_probe"]
    s1_lin = results["S1_Input"]["linear_probe"]
    s1_nonlin = results["S1_Input"]["nonlinear_probe"]

    if s6_nonlin >= 0.65 and direct_acc >= 0.65:
        category = "INFORMATION_PRESERVED_AND_DECODED"
    elif s6_nonlin >= 0.65 and s6_lin < 0.58 and direct_acc < 0.58:
        category = "INFORMATION_PRESENT_BUT_NOT_LINEALLY_DECODABLE"
    elif s6_lin >= 0.65 and direct_acc < 0.58:
        category = "READOUT_FAILURE"
    elif s6_nonlin < 0.55 and s1_nonlin >= 0.65:
        category = "INFORMATION_ABSENT"
    elif s1_nonlin < 0.55:
        category = "TRAINING_FAILURE"
    else:
        category = "UNDETERMINED"

    results["attribution_category"] = category
    return results
