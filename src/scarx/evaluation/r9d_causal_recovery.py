"""SCAR-X Round 9d Comprehensive Causal Information Recovery Suite.

Implements rigorous state restoration experiments with all required controls:
1. Voltage State Restoration (VSR)
2. Spike & Feature State Restoration (SSR) - updates core.features with clean spikes
3. Adaptation State Restoration (ASR)
4. Full State Restoration (FSR: voltage + spikes + adaptation + features)
5. Sham Intervention (SHAM: restoration applied at neutral non-distractor tick)
6. Random State Restoration (RAND: random state of matched norm applied)
7. Non-Noisy Timing Restoration (NNT: restoration applied before distractor)
8. Raw Oracle Decoder: validates 100% information presence in raw stimulus.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from scarx.neural.core import Config
from scarx.neural.r9d_verified_gating import VerifiedNeuralGatingALIF
from scarx.evaluation.r9d_clean_battery import verify_binding_oracle


def run_comprehensive_causal_recovery(
    model: VerifiedNeuralGatingALIF,
    clean_seqs: np.ndarray,
    noisy_seqs: np.ndarray,
    labels: np.ndarray,
    distractor_ticks: tuple[int, ...] = (3, 7, 8),
) -> dict[str, Any]:
    n = len(clean_seqs)
    c = model.config

    counts = {
        "clean": 0,
        "noisy": 0,
        "vsr": 0,
        "ssr": 0,
        "asr": 0,
        "fsr": 0,
        "sham": 0,
        "rand": 0,
        "nnt": 0,
        "oracle": 0,
    }

    # Verify oracle decoder on raw inputs
    for i in range(n):
        if verify_binding_oracle(noisy_seqs[i]) == labels[i]:
            counts["oracle"] += 1

    rng_rand = np.random.default_rng(model.config.seed + 31415)

    for i in range(n):
        c_seq = clean_seqs[i]
        n_seq = noisy_seqs[i]
        y_true = int(labels[i])

        # 1. Clean Run: record full trajectory of all states
        model.reset_episode()
        v_traj = []
        z_traj = []
        a_traj = []
        f_traj = []
        for row in c_seq:
            model.step(row, track=True)
            v_traj.append(model.core.voltage.copy())
            z_traj.append(model.core.spikes.copy())
            a_traj.append(model.core.adaptation.copy())
            f_traj.append(model.core.features.copy())

        p_clean = model.probabilities_from_features(model.core.features)
        if int(np.argmax(p_clean)) == y_true:
            counts["clean"] += 1

        # 2. Noisy Baseline Run
        model.reset_episode()
        for row in n_seq:
            model.step(row, track=True)
        p_noisy = model.probabilities_from_features(model.core.features)
        if int(np.argmax(p_noisy)) == y_true:
            counts["noisy"] += 1

        # 3. Intervention A: Voltage State Restoration (VSR)
        model.reset_episode()
        for t, row in enumerate(n_seq):
            model.step(row, track=True)
            if t in distractor_ticks:
                model.core.voltage[:] = v_traj[t]
        p_vsr = model.probabilities_from_features(model.core.features)
        if int(np.argmax(p_vsr)) == y_true:
            counts["vsr"] += 1

        # 4. Intervention B: Spike & Feature State Restoration (SSR)
        model.reset_episode()
        for t, row in enumerate(n_seq):
            model.step(row, track=True)
            if t in distractor_ticks:
                model.core.spikes[:] = z_traj[t]
                model.core.features[:] = f_traj[t]
        p_ssr = model.probabilities_from_features(model.core.features)
        if int(np.argmax(p_ssr)) == y_true:
            counts["ssr"] += 1

        # 5. Intervention C: Adaptation State Restoration (ASR)
        model.reset_episode()
        for t, row in enumerate(n_seq):
            model.step(row, track=True)
            if t in distractor_ticks:
                model.core.adaptation[:] = a_traj[t]
        p_asr = model.probabilities_from_features(model.core.features)
        if int(np.argmax(p_asr)) == y_true:
            counts["asr"] += 1

        # 6. Intervention D: Full State Restoration (FSR: v, z, a, f)
        model.reset_episode()
        for t, row in enumerate(n_seq):
            model.step(row, track=True)
            if t in distractor_ticks:
                model.core.voltage[:] = v_traj[t]
                model.core.spikes[:] = z_traj[t]
                model.core.adaptation[:] = a_traj[t]
                model.core.features[:] = f_traj[t]
        p_fsr = model.probabilities_from_features(model.core.features)
        if int(np.argmax(p_fsr)) == y_true:
            counts["fsr"] += 1

        # 7. Intervention E: Sham Control (applied at neutral tick 1 where no distractor is present)
        model.reset_episode()
        for t, row in enumerate(n_seq):
            model.step(row, track=True)
            if t == 1:
                model.core.voltage[:] = v_traj[1]
                model.core.spikes[:] = z_traj[1]
                model.core.adaptation[:] = a_traj[1]
                model.core.features[:] = f_traj[1]
        p_sham = model.probabilities_from_features(model.core.features)
        if int(np.argmax(p_sham)) == y_true:
            counts["sham"] += 1

        # 8. Intervention F: Random State Control (random perturbation of matched norm)
        model.reset_episode()
        for t, row in enumerate(n_seq):
            model.step(row, track=True)
            if t in distractor_ticks:
                rand_v = rng_rand.normal(0, np.std(v_traj[t]) + 1e-4, c.hidden).astype(np.float32)
                model.core.voltage[:] = rand_v
        p_rand = model.probabilities_from_features(model.core.features)
        if int(np.argmax(p_rand)) == y_true:
            counts["rand"] += 1

        # 9. Intervention G: Non-Noisy Timing Restoration (at tick 0 before any distractor)
        model.reset_episode()
        for t, row in enumerate(n_seq):
            model.step(row, track=True)
            if t == 0:
                model.core.voltage[:] = v_traj[0]
        p_nnt = model.probabilities_from_features(model.core.features)
        if int(np.argmax(p_nnt)) == y_true:
            counts["nnt"] += 1

    clean_acc = counts["clean"] / n
    noisy_acc = counts["noisy"] / n
    gap = max(1e-4, clean_acc - noisy_acc)

    return {
        "n_samples": n,
        "oracle_raw_acc": counts["oracle"] / n,
        "clean_direct_acc": clean_acc,
        "noisy_direct_acc": noisy_acc,
        "vsr_direct_acc": counts["vsr"] / n,
        "vsr_recovery": (counts["vsr"] / n - noisy_acc) / gap,
        "ssr_direct_acc": counts["ssr"] / n,
        "ssr_recovery": (counts["ssr"] / n - noisy_acc) / gap,
        "asr_direct_acc": counts["asr"] / n,
        "asr_recovery": (counts["asr"] / n - noisy_acc) / gap,
        "fsr_direct_acc": counts["fsr"] / n,
        "fsr_recovery": (counts["fsr"] / n - noisy_acc) / gap,
        "sham_direct_acc": counts["sham"] / n,
        "random_direct_acc": counts["rand"] / n,
        "nnt_direct_acc": counts["nnt"] / n,
    }
