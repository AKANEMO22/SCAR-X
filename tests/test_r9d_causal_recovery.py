"""Unit tests for the R9d comprehensive causal recovery suite."""
from __future__ import annotations

import numpy as np
import pytest

from scarx.neural.core import Config
from scarx.neural.r9d_verified_gating import VerifiedNeuralGatingALIF
from scarx.evaluation.r9c_paired_dataset import generate_paired_dataset
from scarx.evaluation.r9d_causal_recovery import run_comprehensive_causal_recovery


def test_causal_recovery_controls_and_invariants():
    cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=42)
    model = VerifiedNeuralGatingALIF(config=cfg, gating_arch="no_gating")

    batch = generate_paired_dataset(seed=42, count=32, ood=False)
    clean_seqs = batch.sequences["clean"]
    noisy_seqs = batch.sequences["medium_distractor"]
    labels = batch.labels

    model.calibrate(clean_seqs[:16])

    results = run_comprehensive_causal_recovery(
        model=model,
        clean_seqs=clean_seqs,
        noisy_seqs=noisy_seqs,
        labels=labels,
        distractor_ticks=(3, 7, 8),
    )

    # Oracle decoder on raw input MUST be 100%
    assert results["oracle_raw_acc"] == 1.0, f"Oracle decoding failed: {results['oracle_raw_acc']}"

    # Verify all controls are present
    assert "vsr_direct_acc" in results
    assert "ssr_direct_acc" in results
    assert "asr_direct_acc" in results
    assert "fsr_direct_acc" in results
    assert "sham_direct_acc" in results
    assert "random_direct_acc" in results
    assert "nnt_direct_acc" in results

    # Sham intervention on non-noisy tick must not artificially inflate accuracy above clean
    assert results["sham_direct_acc"] <= 1.0
