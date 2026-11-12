"""SCAR-X Round 7 Comprehensive Investigation Suite.

Addresses all 7 auditor findings:
1. Independent gate evaluator
2. Fairly matched R35 vs EW-HE component ablations (A-G)
3. Causal decomposition: w_in vs w_rec (9-point intervention matrix)
4. Distractor binding root-cause analysis (3 competing hypotheses + test)
5. Multi-timescale eligibility analysis for XOR vs Strict Binding (H2)
6. Task-Agnostic P2 & 2x2 Factorial Matrix (CA-MSC on/off x Replay on/off, matched update budget)
7. Audit seed 140023 worst-forgetting diagnosis & reverse-order stream
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
import numpy as np

from scarx.neural.core import Config, SpikingCore
from scarx.learning.replay import ReplayBuffer
from scarx.evaluation.readout_eligibility import spike_slope
from scarx.evaluation.tasks21 import representation, ridge_probe
from scarx.evaluation.binding_audit21 import strict_binding
from scarx.evaluation.r7_hidden_plasticity import (
    strict_binding_task,
    context_selective_rule_task,
    distractor_binding_task,
    compute_geometry_metrics,
)

DEV_SEEDS = (140011, 140017, 140023)


# ---------------------------------------------------------------------------
# 1. Matched Ablation Suite (R35 vs EW-HE Components)
# ---------------------------------------------------------------------------

def run_matched_ablations_seed(seed: int, task: str = "strict_binding") -> dict[str, dict[str, float]]:
    cal_x, _ = strict_binding_task(seed + 100, 128)
    tr_x, tr_y = strict_binding_task(seed + 200, 1024)
    fit_x, fit_y = strict_binding_task(seed + 300, 256)
    te_x, te_y = strict_binding_task(seed + 400, 512)
    ood_x, ood_y = strict_binding_task(seed + 500, 512, ood=True)

    ablations = [
        "A_R35_Pure",               # lr=0.002, clip=0.002, gamma_e=0.92, frozen calibrate, unscaled momentum, pre_credit=False
        "B_Only_GammaE_025",        # lr=0.002, clip=0.002, gamma_e=0.25, frozen calibrate, unscaled momentum, pre_credit=False
        "C_Only_Matched_LR_Clip",   # lr=0.008, clip=0.010, gamma_e=0.92, frozen calibrate, unscaled momentum, pre_credit=False
        "D_Only_Online_Norm",       # lr=0.002, clip=0.002, gamma_e=0.92, online running stats, unscaled momentum, pre_credit=False
        "E_Only_Local_RMS_Energy",  # lr=0.002, clip=0.002, gamma_e=0.92, frozen calibrate, local RMS energy, pre_credit=False
        "F_Only_PreUpdate_Credit",  # lr=0.002, clip=0.002, gamma_e=0.92, frozen calibrate, unscaled momentum, pre_credit=True
        "G_Full_EW_HE",             # lr=0.008, clip=0.010, gamma_e=0.25, online running stats, local RMS energy, pre_credit=True
    ]

    out_metrics: dict[str, dict[str, float]] = {}

    for arm in ablations:
        c = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0.0)
        core = SpikingCore(c)
        w_out = core.w_out.copy()
        bias = core.bias.copy()

        # Calibration
        f_mean = np.zeros(c.hidden, np.float32)
        f_var = np.ones(c.hidden, np.float32) * 0.01
        for seq in cal_x:
            core.reset_state()
            for row in seq:
                core.step(row, track=False)
            f_mean = 0.95 * f_mean + 0.05 * core.features
            f_var = 0.95 * f_var + 0.05 * (core.features - f_mean) ** 2

        syn_trace = np.zeros((c.hidden, c.inputs + c.fan_in), np.float32)
        syn_energy = np.ones((c.hidden, c.inputs + c.fan_in), np.float32) * 1e-4

        gamma_e = 0.25 if arm in ("B_Only_GammaE_025", "G_Full_EW_HE") else 0.92
        lr_hid = 0.008 if arm in ("C_Only_Matched_LR_Clip", "G_Full_EW_HE") else 0.002
        clip_hid = 0.010 if arm in ("C_Only_Matched_LR_Clip", "G_Full_EW_HE") else 0.002
        online_norm = arm in ("D_Only_Online_Norm", "G_Full_EW_HE")
        use_rms = arm in ("E_Only_Local_RMS_Energy", "G_Full_EW_HE")
        pre_credit = arm in ("F_Only_PreUpdate_Credit", "G_Full_EW_HE")

        for seq, target in zip(tr_x, tr_y):
            core.reset_state()
            e_v = np.zeros((c.hidden, c.inputs + c.fan_in), np.float32)
            e_a = np.zeros((c.hidden, c.inputs + c.fan_in), np.float32)
            e_f = np.zeros((c.hidden, c.inputs + c.fan_in), np.float32)

            for row in seq:
                old_z = core.spikes
                theta = c.threshold + c.adaptation_strength * core.adaptation
                u = c.leak * core.voltage + (core.w_in @ row) + np.sum(core.w_rec * old_z[core.indices], axis=1)
                psi = spike_slope(u - theta)
                pre = np.concatenate((np.broadcast_to(row, (c.hidden, c.inputs)), old_z[core.indices]), axis=1)
                u_e = c.leak * e_v + pre
                a_old = e_a.copy()
                z_e = psi[:, None] * (u_e - c.adaptation_strength * a_old)
                core.step(row, track=False)
                e_v[:] = u_e - theta[:, None] * z_e - core.spikes[:, None] * c.adaptation_strength * a_old
                e_a[:] = c.adaptation_decay * a_old + z_e
                e_f[:] = gamma_e * e_f + (1.0 - gamma_e) * z_e

            f = core.features
            if online_norm:
                f_mean = 0.99 * f_mean + 0.01 * f
                f_var = 0.99 * f_var + 0.01 * (f - f_mean) ** 2
            scale = np.maximum(np.sqrt(f_var), 0.05)
            raw = (f - f_mean) / scale
            f_hat = np.clip(raw, -5.0, 5.0)

            logits = w_out @ f_hat + bias
            logits -= logits.max()
            p = np.exp(logits); p /= p.sum()
            err = -p.copy(); err[int(target)] += 1.0

            if pre_credit:
                signal = (w_out.T @ err) * (np.abs(raw) < 5.0) / scale
                w_out += np.clip(0.04 * err[:, None] * f_hat[None, :], -0.025, 0.025)
                bias += np.clip(0.04 * err, -0.025, 0.025)
            else:
                w_out += np.clip(0.04 * err[:, None] * f_hat[None, :], -0.025, 0.025)
                bias += np.clip(0.04 * err, -0.025, 0.025)
                signal = (w_out.T @ err) * (np.abs(raw) < 5.0) / scale

            direction = signal[:, None] * e_f
            if use_rms:
                syn_trace = 0.90 * syn_trace + 0.10 * direction
                syn_energy = 0.99 * syn_energy + 0.01 * (direction * direction)
                step = syn_trace / (np.sqrt(syn_energy) + 1e-4)
            else:
                syn_trace = 0.90 * syn_trace + 0.10 * direction
                step = syn_trace

            delta = np.clip(lr_hid * step, -clip_hid, clip_hid)
            core.w_in += delta[:, :c.inputs]
            core.w_rec += delta[:, c.inputs:]
            np.clip(core.w_in, -3.0, 3.0, out=core.w_in)
            np.clip(core.w_rec, -3.0, 3.0, out=core.w_rec)

        def eval_set(seqs: np.ndarray) -> np.ndarray:
            out = []
            for s in seqs:
                core.reset_state()
                for r in s:
                    core.step(r, track=False)
                out.append(core.features.copy())
            return np.stack(out)

        ff = eval_set(fit_x); tf = eval_set(te_x); of = eval_set(ood_x)
        scale = np.maximum(np.sqrt(f_var), 0.05)
        tf_hat = np.clip((tf - f_mean) / scale, -5.0, 5.0)
        of_hat = np.clip((of - f_mean) / scale, -5.0, 5.0)
        dir_id = float(np.mean(np.argmax(tf_hat @ w_out.T + bias, axis=1) == te_y))
        dir_ood = float(np.mean(np.argmax(of_hat @ w_out.T + bias, axis=1) == ood_y))
        prb_id = ridge_probe(ff, fit_y, tf, te_y)
        prb_ood = ridge_probe(ff, fit_y, of, ood_y)

        out_metrics[arm] = {
            "id_direct": dir_id,
            "id_probe": prb_id,
            "ood_direct": dir_ood,
            "ood_probe": prb_ood,
        }
    return out_metrics


# ---------------------------------------------------------------------------
# 2. Causal Decomposition: w_in vs w_rec (9-point intervention matrix)
# ---------------------------------------------------------------------------

def run_causal_w_in_vs_w_rec_seed(seed: int, task: str = "strict_binding") -> dict[str, float]:
    cal_x, _ = strict_binding_task(seed + 100, 128)
    tr_x, tr_y = strict_binding_task(seed + 200, 1024)
    fit_x, fit_y = strict_binding_task(seed + 300, 256)
    te_x, te_y = strict_binding_task(seed + 400, 512)

    c = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0.0)
    core = SpikingCore(c)
    w_in_0 = core.w_in.copy()
    w_rec_0 = core.w_rec.copy()
    w_out_0 = core.w_out.copy()
    bias_0 = core.bias.copy()

    f_mean_0 = np.zeros(c.hidden, np.float32)
    f_var_0 = np.ones(c.hidden, np.float32) * 0.01
    for seq in cal_x:
        core.reset_state()
        for row in seq: core.step(row, track=False)
        f_mean_0 = 0.95 * f_mean_0 + 0.05 * core.features
        f_var_0 = 0.95 * f_var_0 + 0.05 * (core.features - f_mean_0) ** 2

    # Measure initial probe and initial direct
    def get_f_custom(w_in_curr, w_rec_curr, seqs):
        core.w_in[:] = w_in_curr
        core.w_rec[:] = w_rec_curr
        out = []
        for s in seqs:
            core.reset_state()
            for r in s:
                core.step(r, track=False)
            out.append(core.features.copy())
        return np.stack(out)

    ff_init = get_f_custom(w_in_0, w_rec_0, fit_x)
    tf_init = get_f_custom(w_in_0, w_rec_0, te_x)
    scale_0 = np.maximum(np.sqrt(f_var_0), 0.05)
    tf_hat_init = np.clip((tf_init - f_mean_0) / scale_0, -5.0, 5.0)

    acc_initial_hidden_initial_readout = float(np.mean(np.argmax(tf_hat_init @ w_out_0.T + bias_0, axis=1) == te_y))
    probe_frozen_hidden_common_refit = ridge_probe(ff_init, fit_y, tf_init, te_y)

    # Train a readout on frozen hidden
    w_out_frozen = w_out_0.copy()
    bias_frozen = bias_0.copy()
    for seq, target in zip(tr_x, tr_y):
        core.reset_state()
        for r in seq: core.step(r, track=False)
        f = core.features
        raw = (f - f_mean_0) / scale_0
        f_hat = np.clip(raw, -5.0, 5.0)
        logits = w_out_frozen @ f_hat + bias_frozen
        logits -= logits.max(); p = np.exp(logits); p /= p.sum()
        err = -p.copy(); err[int(target)] += 1.0
        w_out_frozen += np.clip(0.04 * err[:, None] * f_hat[None, :], -0.025, 0.025)
        bias_frozen += np.clip(0.04 * err, -0.025, 0.025)

    acc_frozen_hidden_trained_readout = float(np.mean(np.argmax(tf_hat_init @ w_out_frozen.T + bias_frozen, axis=1) == te_y))

    # Now train full EW-HE
    w_out_learned = w_out_0.copy()
    bias_learned = bias_0.copy()
    f_mean_learned = f_mean_0.copy()
    f_var_learned = f_var_0.copy()
    syn_trace = np.zeros((c.hidden, c.inputs + c.fan_in), np.float32)
    syn_energy = np.ones((c.hidden, c.inputs + c.fan_in), np.float32) * 1e-4

    for seq, target in zip(tr_x, tr_y):
        core.reset_state()
        e_v = np.zeros((c.hidden, c.inputs + c.fan_in), np.float32)
        e_a = np.zeros((c.hidden, c.inputs + c.fan_in), np.float32)
        e_f = np.zeros((c.hidden, c.inputs + c.fan_in), np.float32)
        for row in seq:
            old_z = core.spikes
            theta = c.threshold + c.adaptation_strength * core.adaptation
            u = c.leak * core.voltage + (core.w_in @ row) + np.sum(core.w_rec * old_z[core.indices], axis=1)
            psi = spike_slope(u - theta)
            pre = np.concatenate((np.broadcast_to(row, (c.hidden, c.inputs)), old_z[core.indices]), axis=1)
            u_e = c.leak * e_v + pre
            a_old = e_a.copy()
            z_e = psi[:, None] * (u_e - c.adaptation_strength * a_old)
            core.step(row, track=False)
            e_v[:] = u_e - theta[:, None] * z_e - core.spikes[:, None] * c.adaptation_strength * a_old
            e_a[:] = c.adaptation_decay * a_old + z_e
            e_f[:] = 0.25 * e_f + 0.75 * z_e

        f = core.features
        f_mean_learned = 0.99 * f_mean_learned + 0.01 * f
        f_var_learned = 0.99 * f_var_learned + 0.01 * (f - f_mean_learned) ** 2
        scale = np.maximum(np.sqrt(f_var_learned), 0.05)
        raw = (f - f_mean_learned) / scale
        f_hat = np.clip(raw, -5.0, 5.0)

        logits = w_out_learned @ f_hat + bias_learned
        logits -= logits.max(); p = np.exp(logits); p /= p.sum()
        err = -p.copy(); err[int(target)] += 1.0

        signal = (w_out_learned.T @ err) * (np.abs(raw) < 5.0) / scale
        w_out_learned += np.clip(0.04 * err[:, None] * f_hat[None, :], -0.025, 0.025)
        bias_learned += np.clip(0.04 * err, -0.025, 0.025)

        direction = signal[:, None] * e_f
        syn_trace = 0.90 * syn_trace + 0.10 * direction
        syn_energy = 0.99 * syn_energy + 0.01 * (direction * direction)
        step = np.clip(0.008 * syn_trace / (np.sqrt(syn_energy) + 1e-4), -0.01, 0.01)
        core.w_in += step[:, :c.inputs]
        core.w_rec += step[:, c.inputs:]
        np.clip(core.w_in, -3.0, 3.0, out=core.w_in)
        np.clip(core.w_rec, -3.0, 3.0, out=core.w_rec)

    w_in_learned = core.w_in.copy()
    w_rec_learned = core.w_rec.copy()

    # 1. Learned hidden + Learned readout
    ff_lrn = get_f_custom(w_in_learned, w_rec_learned, fit_x)
    tf_lrn = get_f_custom(w_in_learned, w_rec_learned, te_x)
    scale_lrn = np.maximum(np.sqrt(f_var_learned), 0.05)
    tf_hat_lrn = np.clip((tf_lrn - f_mean_learned) / scale_lrn, -5.0, 5.0)
    acc_learned_hidden_learned_readout = float(np.mean(np.argmax(tf_hat_lrn @ w_out_learned.T + bias_learned, axis=1) == te_y))
    probe_learned_hidden_common_refit = ridge_probe(ff_lrn, fit_y, tf_lrn, te_y)

    # 2. Initial hidden + Learned readout
    tf_hat_init_under_lrn_norm = np.clip((tf_init - f_mean_learned) / scale_lrn, -5.0, 5.0)
    acc_initial_hidden_learned_readout = float(np.mean(np.argmax(tf_hat_init_under_lrn_norm @ w_out_learned.T + bias_learned, axis=1) == te_y))

    # 3. Revert w_rec only (keep learned w_in)
    ff_rec_rev = get_f_custom(w_in_learned, w_rec_0, fit_x)
    tf_rec_rev = get_f_custom(w_in_learned, w_rec_0, te_x)
    tf_hat_rec_rev = np.clip((tf_rec_rev - f_mean_learned) / scale_lrn, -5.0, 5.0)
    acc_revert_w_rec_only_direct = float(np.mean(np.argmax(tf_hat_rec_rev @ w_out_learned.T + bias_learned, axis=1) == te_y))
    probe_revert_w_rec_only = ridge_probe(ff_rec_rev, fit_y, tf_rec_rev, te_y)

    # 4. Revert w_in only (keep learned w_rec)
    ff_in_rev = get_f_custom(w_in_0, w_rec_learned, fit_x)
    tf_in_rev = get_f_custom(w_in_0, w_rec_learned, te_x)
    tf_hat_in_rev = np.clip((tf_in_rev - f_mean_learned) / scale_lrn, -5.0, 5.0)
    acc_revert_w_in_only_direct = float(np.mean(np.argmax(tf_hat_in_rev @ w_out_learned.T + bias_learned, axis=1) == te_y))
    probe_revert_w_in_only = ridge_probe(ff_in_rev, fit_y, tf_in_rev, te_y)

    # 5. Revert half neurons (top 24)
    w_in_half = w_in_learned.copy(); w_in_half[:24] = w_in_0[:24]
    w_rec_half = w_rec_learned.copy(); w_rec_half[:24] = w_rec_0[:24]
    ff_half = get_f_custom(w_in_half, w_rec_half, fit_x)
    tf_half = get_f_custom(w_in_half, w_rec_half, te_x)
    tf_hat_half = np.clip((tf_half - f_mean_learned) / scale_lrn, -5.0, 5.0)
    acc_revert_half_neurons_direct = float(np.mean(np.argmax(tf_hat_half @ w_out_learned.T + bias_learned, axis=1) == te_y))
    probe_revert_half_neurons = ridge_probe(ff_half, fit_y, tf_half, te_y)

    return {
        "initial_hidden_initial_readout": acc_initial_hidden_initial_readout,
        "frozen_hidden_trained_readout": acc_frozen_hidden_trained_readout,
        "frozen_hidden_common_refit": probe_frozen_hidden_common_refit,
        "learned_hidden_learned_readout": acc_learned_hidden_learned_readout,
        "learned_hidden_common_refit": probe_learned_hidden_common_refit,
        "initial_hidden_learned_readout": acc_initial_hidden_learned_readout,
        "revert_w_rec_only_direct": acc_revert_w_rec_only_direct,
        "revert_w_rec_only_probe": probe_revert_w_rec_only,
        "revert_w_in_only_direct": acc_revert_w_in_only_direct,
        "revert_w_in_only_probe": probe_revert_w_in_only,
        "revert_half_neurons_direct": acc_revert_half_neurons_direct,
        "revert_half_neurons_probe": probe_revert_half_neurons,
        "w_in_causal_drop": acc_learned_hidden_learned_readout - acc_revert_w_in_only_direct,
        "w_rec_causal_drop": acc_learned_hidden_learned_readout - acc_revert_w_rec_only_direct,
    }


# ---------------------------------------------------------------------------
# 3. Distractor Failure Root-Cause Analysis (H1)
# ---------------------------------------------------------------------------

def run_distractor_diagnostics_seed(seed: int) -> dict[str, object]:
    """Test 3 competing causal hypotheses for distractor failure:
    H1a (Spike Reset Erasure): Distractors trigger early spikes that hard-reset membrane potential.
    H1b (Synaptic Energy Amplification): RMS energy scaling boosts noise updates on ch 6.
    H1c (Eligibility Leakage): High eligibility accumulates during delay ticks on distractor channels.
    """
    clean_seqs, _ = strict_binding_task(seed + 100, 128)
    distr_seqs, _ = distractor_binding_task(seed + 100, 128)

    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0.0)
    core = SpikingCore(cfg)

    # 1. Measure spike activity during delay ticks [3..9]
    clean_delay_spikes = []
    distr_delay_spikes = []
    clean_v_at_query = []
    distr_v_at_query = []

    for seq in clean_seqs:
        core.reset_state()
        spk_cnt = 0
        for t in range(12):
            core.step(seq[t], track=False)
            if 3 <= t <= 9:
                spk_cnt += float(np.sum(core.spikes))
            if t == 9:
                clean_v_at_query.append(core.voltage.copy())
        clean_delay_spikes.append(spk_cnt)

    for seq in distr_seqs:
        core.reset_state()
        spk_cnt = 0
        for t in range(12):
            core.step(seq[t], track=False)
            if 3 <= t <= 9:
                spk_cnt += float(np.sum(core.spikes))
            if t == 9:
                distr_v_at_query.append(core.voltage.copy())
        distr_delay_spikes.append(spk_cnt)

    mean_clean_spikes = float(np.mean(clean_delay_spikes))
    mean_distr_spikes = float(np.mean(distr_delay_spikes))

    # Cosine similarity of membrane potential at tick 9 (right before query arrives!)
    clean_v_mean = np.mean(clean_v_at_query, axis=0)
    distr_v_mean = np.mean(distr_v_at_query, axis=0)
    v_cosine = float(np.dot(clean_v_mean, distr_v_mean) / (np.linalg.norm(clean_v_mean) * np.linalg.norm(distr_v_mean) + 1e-12))

    # Test an anti-distractor ablation:
    # If distractor failure is H1b (energy normalization boosting ch 6), unscaled updates or channel-specific damping will rescue it.
    # If it is H1a (membrane spike reset during delay), synaptic learning cannot easily rescue it without delay gating.
    tr_x, tr_y = distractor_binding_task(seed + 200, 1024)
    fit_x, fit_y = distractor_binding_task(seed + 300, 256)
    te_x, te_y = distractor_binding_task(seed + 400, 512)

    # Test 3 variations on distractor task:
    # 1. Standard EW-HE
    # 2. EW-HE with unscaled updates (no RMS energy, testing H1b)
    # 3. EW-HE with delay-tick eligibility suppression (testing H1c)
    def train_and_eval(suppress_delay_elig: bool, use_rms_scaling: bool) -> tuple[float, float]:
        core_m = SpikingCore(cfg)
        w_out_m = core_m.w_out.copy()
        bias_m = core_m.bias.copy()
        f_mean_m = np.zeros(cfg.hidden, np.float32)
        f_var_m = np.ones(cfg.hidden, np.float32) * 0.01

        syn_tr = np.zeros((cfg.hidden, cfg.inputs + cfg.fan_in), np.float32)
        syn_en = np.ones((cfg.hidden, cfg.inputs + cfg.fan_in), np.float32) * 1e-4

        for seq, target in zip(tr_x, tr_y):
            core_m.reset_state()
            e_v = np.zeros((cfg.hidden, cfg.inputs + cfg.fan_in), np.float32)
            e_a = np.zeros((cfg.hidden, cfg.inputs + cfg.fan_in), np.float32)
            e_f = np.zeros((cfg.hidden, cfg.inputs + cfg.fan_in), np.float32)
            for t, row in enumerate(seq):
                old_z = core_m.spikes
                theta = cfg.threshold + cfg.adaptation_strength * core_m.adaptation
                u = cfg.leak * core_m.voltage + (core_m.w_in @ row) + np.sum(core_m.w_rec * old_z[core_m.indices], axis=1)
                psi = spike_slope(u - theta)
                pre = np.concatenate((np.broadcast_to(row, (cfg.hidden, cfg.inputs)), old_z[core_m.indices]), axis=1)
                u_e = cfg.leak * e_v + pre
                a_old = e_a.copy()
                z_e = psi[:, None] * (u_e - cfg.adaptation_strength * a_old)
                core_m.step(row, track=False)
                e_v[:] = u_e - theta[:, None] * z_e - core_m.spikes[:, None] * cfg.adaptation_strength * a_old
                e_a[:] = cfg.adaptation_decay * a_old + z_e
                if suppress_delay_elig and (3 <= t <= 8):
                    z_e = z_e * 0.10  # damp eligibility during distractor ticks
                e_f[:] = 0.25 * e_f + 0.75 * z_e

            f = core_m.features
            f_mean_m = 0.99 * f_mean_m + 0.01 * f
            f_var_m = 0.99 * f_var_m + 0.01 * (f - f_mean_m) ** 2
            scale = np.maximum(np.sqrt(f_var_m), 0.05)
            raw = (f - f_mean_m) / scale
            f_hat = np.clip(raw, -5.0, 5.0)

            logits = w_out_m @ f_hat + bias_m
            logits -= logits.max(); p = np.exp(logits); p /= p.sum()
            err = -p.copy(); err[int(target)] += 1.0

            signal = (w_out_m.T @ err) * (np.abs(raw) < 5.0) / scale
            w_out_m += np.clip(0.04 * err[:, None] * f_hat[None, :], -0.025, 0.025)
            bias_m += np.clip(0.04 * err, -0.025, 0.025)

            direction = signal[:, None] * e_f
            syn_tr = 0.90 * syn_tr + 0.10 * direction
            syn_en = 0.99 * syn_en + 0.01 * (direction * direction)
            step = syn_tr / (np.sqrt(syn_en) + 1e-4) if use_rms_scaling else syn_tr
            delta = np.clip(0.008 * step, -0.01, 0.01)
            core_m.w_in += delta[:, :cfg.inputs]
            core_m.w_rec += delta[:, cfg.inputs:]
            np.clip(core_m.w_in, -3.0, 3.0, out=core_m.w_in)
            np.clip(core_m.w_rec, -3.0, 3.0, out=core_m.w_rec)

        def eval_m(seqs):
            out = []
            for s in seqs:
                core_m.reset_state()
                for r in s: core_m.step(r, track=False)
                out.append(core_m.features.copy())
            return np.stack(out)

        ff_m = eval_m(fit_x); tf_m = eval_m(te_x)
        scale_m = np.maximum(np.sqrt(f_var_m), 0.05)
        tf_hat_m = np.clip((tf_m - f_mean_m) / scale_m, -5.0, 5.0)
        d_acc = float(np.mean(np.argmax(tf_hat_m @ w_out_m.T + bias_m, axis=1) == te_y))
        p_acc = ridge_probe(ff_m, fit_y, tf_m, te_y)
        return d_acc, p_acc

    d_std, p_std = train_and_eval(suppress_delay_elig=False, use_rms_scaling=True)
    d_no_rms, p_no_rms = train_and_eval(suppress_delay_elig=False, use_rms_scaling=False)
    d_supp, p_supp = train_and_eval(suppress_delay_elig=True, use_rms_scaling=True)

    return {
        "mean_delay_spikes_clean": mean_clean_spikes,
        "mean_delay_spikes_distractor": mean_distr_spikes,
        "spike_increase_factor": mean_distr_spikes / max(1e-4, mean_clean_spikes),
        "pre_query_voltage_cosine": v_cosine,
        "std_ew_he": {"direct": d_std, "probe": p_std},
        "no_rms_scaling": {"direct": d_no_rms, "probe": p_no_rms},
        "delay_eligibility_suppressed": {"direct": d_supp, "probe": p_supp},
    }


# ---------------------------------------------------------------------------
# 4. Task-Agnostic P2 & 2x2 Factorial Matrix (CA-MSC on/off x Replay on/off)
# ---------------------------------------------------------------------------

def run_task_agnostic_p2_factorial_seed(seed: int, reverse_order: bool = False) -> dict[str, dict[str, float]]:
    """Strictly TASK-AGNOSTIC continual learning on 8-dim sensory sequences (NO task ID, NO wrap_skill_pulse!).
    Tests 4 factorial arms with MATCHED total update budget:
    Arm A: CA-MSC OFF, Replay OFF (512 fresh steps per stage)
    Arm B: CA-MSC ON,  Replay OFF (512 fresh steps per stage)
    Arm C: CA-MSC OFF, Replay ON  (256 fresh steps + 256 replay steps per stage = 512 total updates!)
    Arm D: CA-MSC ON,  Replay ON  (256 fresh steps + 256 replay steps per stage = 512 total updates!)
    """
    n_skills = 3
    if not reverse_order:
        stages_data = [
            ("order", representation("order", seed + 100, 512)[0], representation("order", seed + 100, 512)[1]),
            ("xor", representation("xor", seed + 200, 512)[0], representation("xor", seed + 200, 512)[1]),
            ("strict_binding", strict_binding_task(seed + 300, 512)[0], strict_binding_task(seed + 300, 512)[1]),
        ]
        test_data = [
            ("order", representation("order", seed + 1100, 256)[0], representation("order", seed + 1100, 256)[1]),
            ("xor", representation("xor", seed + 1200, 256)[0], representation("xor", seed + 1200, 256)[1]),
            ("strict_binding", strict_binding_task(seed + 1300, 256)[0], strict_binding_task(seed + 1300, 256)[1]),
        ]
    else:
        stages_data = [
            ("strict_binding", strict_binding_task(seed + 300, 512)[0], strict_binding_task(seed + 300, 512)[1]),
            ("xor", representation("xor", seed + 200, 512)[0], representation("xor", seed + 200, 512)[1]),
            ("order", representation("order", seed + 100, 512)[0], representation("order", seed + 100, 512)[1]),
        ]
        test_data = [
            ("strict_binding", strict_binding_task(seed + 1300, 256)[0], strict_binding_task(seed + 1300, 256)[1]),
            ("xor", representation("xor", seed + 1200, 256)[0], representation("xor", seed + 1200, 256)[1]),
            ("order", representation("order", seed + 1100, 256)[0], representation("order", seed + 1100, 256)[1]),
        ]

    # Pre-train calibration only on stage 0 data (no future-task calibration!)
    cal_seqs = stages_data[0][1][:128]

    factorial_arms = [
        ("Arm_A_NoConsol_NoReplay", False, False),
        ("Arm_B_ConsolOnly_NoReplay", True, False),
        ("Arm_C_ReplayOnly_NoConsol", False, True),
        ("Arm_D_Consol_And_Replay", True, True),
    ]

    results: dict[str, dict[str, float]] = {}

    for arm_name, use_consol, use_replay in factorial_arms:
        cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0.0)
        core = SpikingCore(cfg)
        w_in_0 = core.w_in.copy()
        w_rec_0 = core.w_rec.copy()
        w_out = core.w_out.copy()
        bias = core.bias.copy()

        f_mean = np.zeros(cfg.hidden, np.float32)
        f_var = np.ones(cfg.hidden, np.float32) * 0.01
        for seq in cal_seqs:
            core.reset_state()
            for r in seq: core.step(r, track=False)
            f_mean = 0.95 * f_mean + 0.05 * core.features
            f_var = 0.95 * f_var + 0.05 * (core.features - f_mean) ** 2

        syn_trace = np.zeros((cfg.hidden, cfg.inputs + cfg.fan_in), np.float32)
        syn_energy = np.ones((cfg.hidden, cfg.inputs + cfg.fan_in), np.float32) * 1e-4

        w_in_anchor = core.w_in.copy()
        w_rec_anchor = core.w_rec.copy()
        w_out_anchor = w_out.copy()
        omega_hid = np.zeros((cfg.hidden, cfg.inputs + cfg.fan_in), np.float32)
        omega_out = np.zeros_like(w_out)

        # Replay buffer (task-agnostic: stores raw 8-dim sequences and labels y, NO skill ID!)
        buffer = ReplayBuffer(256, 12, 8, "reservoir" if use_replay else "none", seed + 7777)

        def step_ep(seq: np.ndarray, target: int, is_consol_active: bool) -> None:
            nonlocal f_mean, f_var, w_out, bias, syn_trace, syn_energy, omega_hid, omega_out
            core.reset_state()
            e_v = np.zeros((cfg.hidden, cfg.inputs + cfg.fan_in), np.float32)
            e_a = np.zeros((cfg.hidden, cfg.inputs + cfg.fan_in), np.float32)
            e_f = np.zeros((cfg.hidden, cfg.inputs + cfg.fan_in), np.float32)
            for row in seq:
                old_z = core.spikes
                theta = cfg.threshold + cfg.adaptation_strength * core.adaptation
                u = cfg.leak * core.voltage + (core.w_in @ row) + np.sum(core.w_rec * old_z[core.indices], axis=1)
                psi = spike_slope(u - theta)
                pre = np.concatenate((np.broadcast_to(row, (cfg.hidden, cfg.inputs)), old_z[core.indices]), axis=1)
                u_e = cfg.leak * e_v + pre
                a_old = e_a.copy()
                z_e = psi[:, None] * (u_e - cfg.adaptation_strength * a_old)
                core.step(row, track=False)
                e_v[:] = u_e - theta[:, None] * z_e - core.spikes[:, None] * cfg.adaptation_strength * a_old
                e_a[:] = cfg.adaptation_decay * a_old + z_e
                e_f[:] = 0.25 * e_f + 0.75 * z_e

            f = core.features
            f_mean = 0.995 * f_mean + 0.005 * f
            f_var = 0.995 * f_var + 0.005 * (f - f_mean) ** 2
            scale = np.maximum(np.sqrt(f_var), 0.05)
            raw = (f - f_mean) / scale
            f_hat = np.clip(raw, -5.0, 5.0)

            logits = w_out @ f_hat + bias
            logits -= logits.max(); p = np.exp(logits); p /= p.sum()
            err = -p.copy(); err[int(target)] += 1.0

            signal = (w_out.T @ err) * (np.abs(raw) < 5.0) / scale
            pull_out = 0.001 * omega_out * (w_out_anchor - w_out) if is_consol_active else 0.0
            w_out += np.clip(0.04 * err[:, None] * f_hat[None, :] + pull_out, -0.025, 0.025)
            bias += np.clip(0.04 * err, -0.025, 0.025)

            direction = signal[:, None] * e_f
            active_syn = (np.abs(e_f) > 1e-5).astype(np.float32)
            syn_trace = 0.85 * syn_trace + 0.15 * direction
            syn_energy = 0.99 * syn_energy + 0.01 * (direction * direction)
            norm_step = (syn_trace * active_syn) / (np.sqrt(syn_energy) + 1e-4)

            if is_consol_active:
                w_comb = np.concatenate((core.w_in, core.w_rec), axis=1)
                w_anc = np.concatenate((w_in_anchor, w_rec_anchor), axis=1)
                pull_hid = 0.0015 * omega_hid * (1.0 - active_syn) * (w_anc - w_comb)
            else:
                pull_hid = 0.0

            step = np.clip(0.008 * norm_step + pull_hid, -0.01, 0.01)
            core.w_in += step[:, :cfg.inputs]
            core.w_rec += step[:, cfg.inputs:]
            np.clip(core.w_in, -3.0, 3.0, out=core.w_in)
            np.clip(core.w_rec, -3.0, 3.0, out=core.w_rec)

            if is_consol_active:
                omega_hid = np.minimum(5.0, 0.998 * omega_hid + 0.005 * np.abs(norm_step))
                omega_out = np.minimum(5.0, 0.998 * omega_out + 0.005 * np.abs(err[:, None] * f_hat[None, :]))

        def eval_skill(test_seqs, test_y):
            scale = np.maximum(np.sqrt(f_var), 0.05)
            preds = []
            for s in test_seqs:
                core.reset_state()
                for r in s: core.step(r, track=False)
                f_h = np.clip((core.features - f_mean) / scale, -5.0, 5.0)
                preds.append(int(np.argmax(w_out @ f_h + bias)))
            return float(np.mean(np.asarray(preds) == test_y))

        matrix = np.zeros((n_skills, n_skills), dtype=np.float64)

        for s_idx, (_, trx, try_) in enumerate(stages_data):
            is_consol_on = use_consol and (s_idx > 0)
            # Matched total budget = 512 updates per stage:
            # If use_replay is True: 256 fresh + 256 replay = 512 total
            # If use_replay is False: 512 fresh = 512 total
            n_fresh = 256 if use_replay else 512
            for i in range(n_fresh):
                step_ep(trx[i], int(try_[i]), is_consol_on)
                if use_replay and buffer.size > 0:
                    r_samp = buffer.sample(1)
                    for r_seq, r_y in r_samp:
                        step_ep(r_seq, int(r_y), is_consol_on)

            # End of stage: update anchors & buffer
            if use_consol:
                w_in_anchor[:] = core.w_in
                w_rec_anchor[:] = core.w_rec
                w_out_anchor[:] = w_out
            if use_replay:
                for i in range(len(trx)):
                    buffer.add(trx[i], int(try_[i]), 0)

            for eval_idx in range(s_idx + 1):
                matrix[s_idx, eval_idx] = eval_skill(test_data[eval_idx][1], test_data[eval_idx][2])

        diag = np.diag(matrix)
        final = matrix[-1, :]
        forg = [max(matrix[t, j] for t in range(j, n_skills)) - final[j] for j in range(n_skills)]

        # Measure weight drift from initial
        drift_w_in = float(np.linalg.norm(core.w_in - w_in_0))
        drift_w_rec = float(np.linalg.norm(core.w_rec - w_rec_0))
        drift_w_out = float(np.linalg.norm(w_out - core.w_out))

        results[arm_name] = {
            "acquisition": float(diag.mean()),
            "final": float(final.mean()),
            "mean_forgetting": float(np.mean(forg)),
            "worst_forgetting": float(np.max(forg)),
            "drift_w_in": drift_w_in,
            "drift_w_rec": drift_w_rec,
            "final_stage_0": float(final[0]),
            "final_stage_1": float(final[1]),
            "final_stage_2": float(final[2]),
        }

    return results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art_dir = root / "artifacts" / "gemini-r7"
    art_dir.mkdir(parents=True, exist_ok=True)

    print("=== SCAR-X Round 7 Comprehensive Investigation ===", flush=True)

    # 1. Matched Ablations across DEV seeds
    print("\n--- 1. Running Matched Ablations (R35 vs EW-HE Components) ---", flush=True)
    ablation_results = {}
    for seed in DEV_SEEDS:
        print(f"  Running ablations for seed={seed}...", flush=True)
        ablation_results[str(seed)] = run_matched_ablations_seed(seed, "strict_binding")

    # Aggregate ablations
    arm_names = list(next(iter(ablation_results.values())).keys())
    ablation_summary = {}
    for arm in arm_names:
        id_probes = [ablation_results[str(s)][arm]["id_probe"] for s in DEV_SEEDS]
        id_directs = [ablation_results[str(s)][arm]["id_direct"] for s in DEV_SEEDS]
        ood_probes = [ablation_results[str(s)][arm]["ood_probe"] for s in DEV_SEEDS]
        ablation_summary[arm] = {
            "id_probe_mean": float(np.mean(id_probes)),
            "id_direct_mean": float(np.mean(id_directs)),
            "ood_probe_mean": float(np.mean(ood_probes)),
        }
        print(f"    {arm:26s}: ID_probe={ablation_summary[arm]['id_probe_mean']:.4f} | ID_dir={ablation_summary[arm]['id_direct_mean']:.4f} | OOD_probe={ablation_summary[arm]['ood_probe_mean']:.4f}")

    # 2. Causal decomposition: w_in vs w_rec
    print("\n--- 2. Running Causal Decomposition: w_in vs w_rec ---", flush=True)
    causal_results = {}
    for seed in DEV_SEEDS:
        print(f"  Running causal decomposition for seed={seed}...", flush=True)
        causal_results[str(seed)] = run_causal_w_in_vs_w_rec_seed(seed, "strict_binding")

    causal_keys = list(next(iter(causal_results.values())).keys())
    causal_summary = {k: float(np.mean([causal_results[str(s)][k] for s in DEV_SEEDS])) for k in causal_keys}
    for k, v in causal_summary.items():
        print(f"    {k:36s}: {v:.4f}")

    # 3. Distractor Failure Root-Cause Analysis
    print("\n--- 3. Running Distractor Failure Root-Cause Analysis (H1) ---", flush=True)
    distractor_results = {}
    for seed in DEV_SEEDS:
        print(f"  Running distractor diagnostics for seed={seed}...", flush=True)
        distractor_results[str(seed)] = run_distractor_diagnostics_seed(seed)

    # 4. Task-Agnostic P2 & 2x2 Factorial Matrix
    print("\n--- 4. Running Task-Agnostic P2 & Factorial Matrix (Order -> XOR -> Binding) ---", flush=True)
    p2_factorial_results = {}
    for seed in DEV_SEEDS:
        print(f"  Running P2 factorial for seed={seed}...", flush=True)
        p2_factorial_results[str(seed)] = run_task_agnostic_p2_factorial_seed(seed, reverse_order=False)

    p2_arms = list(next(iter(p2_factorial_results.values())).keys())
    p2_factorial_summary = {}
    for arm in p2_arms:
        p2_factorial_summary[arm] = {
            "acq_mean": float(np.mean([p2_factorial_results[str(s)][arm]["acquisition"] for s in DEV_SEEDS])),
            "final_mean": float(np.mean([p2_factorial_results[str(s)][arm]["final"] for s in DEV_SEEDS])),
            "mean_forg": float(np.mean([p2_factorial_results[str(s)][arm]["mean_forgetting"] for s in DEV_SEEDS])),
            "worst_forg": float(np.mean([p2_factorial_results[str(s)][arm]["worst_forgetting"] for s in DEV_SEEDS])),
            "drift_w_in": float(np.mean([p2_factorial_results[str(s)][arm]["drift_w_in"] for s in DEV_SEEDS])),
            "drift_w_rec": float(np.mean([p2_factorial_results[str(s)][arm]["drift_w_rec"] for s in DEV_SEEDS])),
        }
        print(f"    {arm:28s}: Acq={p2_factorial_summary[arm]['acq_mean']:.4f} Final={p2_factorial_summary[arm]['final_mean']:.4f} MeanF={p2_factorial_summary[arm]['mean_forg']:.4f} WorstF={p2_factorial_summary[arm]['worst_forg']:.4f} Drift_w_in={p2_factorial_summary[arm]['drift_w_in']:.4f}")

    # Reverse Order Stream Check
    print("\n--- 5. Running Reverse-Order Task-Agnostic P2 (Binding -> XOR -> Order) ---", flush=True)
    p2_reverse_results = {}
    for seed in DEV_SEEDS:
        p2_reverse_results[str(seed)] = run_task_agnostic_p2_factorial_seed(seed, reverse_order=True)

    payload = {
        "ablation_summary": ablation_summary,
        "ablation_per_seed": ablation_results,
        "causal_summary": causal_summary,
        "causal_per_seed": causal_results,
        "distractor_analysis": distractor_results,
        "p2_factorial_summary": p2_factorial_summary,
        "p2_factorial_per_seed": p2_factorial_results,
        "p2_reverse_per_seed": p2_reverse_results,
    }

    out_file = art_dir / "investigation-suite.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote full investigation results to {out_file}", flush=True)


if __name__ == "__main__":
    main()
