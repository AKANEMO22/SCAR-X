"""Unit tests for the R9c causal diagnostics and multi-layer tracing suite."""
from __future__ import annotations

import numpy as np
import pytest

from scarx.neural.core import Config
from scarx.neural.r9b_factorial_eprop import FactorialMTSGEProp
from scarx.evaluation.r9c_paired_dataset import generate_paired_dataset
from scarx.evaluation.r9c_causal_diagnostics import (
    ControlledNonLinearProbe,
    trace_episode,
    run_causal_restoration_experiment,
    evaluate_multi_layer_decodability,
)


def get_test_model():
    cfg = Config(neuron="alif", hidden=48, fan_in=8, inputs=8, outputs=2, seed=42)
    model = FactorialMTSGEProp(
        config=cfg,
        decay_mode="uniform_fast",
        gating_mode="none",
        reset_mode="stopped_reset",
        control_mode="learned",
    )
    # Calibrate with small noise
    rng = np.random.default_rng(42)
    calib = rng.normal(0, 0.1, (10, 12, 8)).astype(np.float32)
    model.calibrate(calib)
    return model


def test_controlled_nonlinear_probe():
    rng = np.random.default_rng(101)
    # XOR dataset to verify non-linear learning capability
    x = rng.normal(0, 0.5, (100, 2)).astype(np.float32)
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(np.int32)

    probe = ControlledNonLinearProbe(input_dim=2, hidden_dim=16, lr=0.1, epochs=100)
    probe.fit(x, y)
    acc = probe.score(x, y)
    assert acc > 0.70, f"Expected XOR probe accuracy > 0.70, got {acc}"


def test_trace_episode():
    model = get_test_model()
    seq = np.random.default_rng(42).normal(0, 0.1, (12, 8)).astype(np.float32)
    trace = trace_episode(model, seq, sample_id=0, label=1)

    assert len(trace.timesteps) == 12
    assert trace.final_features.shape == (48,)
    assert trace.logits.shape == (2,)
    assert trace.prediction in (0, 1)

    # Check timestep trace
    ts0 = trace.timesteps[0]
    assert ts0.u_pre.shape == (48,)
    assert ts0.v_post.shape == (48,)
    assert ts0.z.shape == (48,)
    assert ts0.a.shape == (48,)
    assert ts0.e_f_norm.shape == (48,)


def test_causal_restoration_experiment():
    model = get_test_model()
    batch = generate_paired_dataset(seed=55, count=16, ood=False)
    clean_seqs = batch.sequences["clean"]
    noisy_seqs = batch.sequences["medium_distractor"]
    labels = batch.labels

    results = run_causal_restoration_experiment(
        model=model,
        clean_seqs=clean_seqs,
        noisy_seqs=noisy_seqs,
        labels=labels,
        distractor_ticks=(3, 7, 8),
    )

    assert results["n_samples"] == 16
    assert 0.0 <= results["clean_direct_acc"] <= 1.0
    assert 0.0 <= results["noisy_direct_acc"] <= 1.0
    assert 0.0 <= results["vsr_direct_acc"] <= 1.0
    assert 0.0 <= results["fsr_direct_acc"] <= 1.0
    assert results["mean_voltage_perturbation"] >= 0.0


def test_evaluate_multi_layer_decodability():
    model = get_test_model()
    batch_train = generate_paired_dataset(seed=10, count=20, ood=False)
    batch_test = generate_paired_dataset(seed=20, count=20, ood=False)

    clean_train = batch_train.sequences["clean"]
    clean_test = batch_test.sequences["clean"]

    train_traces = [trace_episode(model, clean_train[i], i, batch_train.labels[i]) for i in range(20)]
    test_traces = [trace_episode(model, clean_test[i], i, batch_test.labels[i]) for i in range(20)]

    decodability = evaluate_multi_layer_decodability(
        train_traces=train_traces,
        test_traces=test_traces,
        raw_train_x=clean_train,
        raw_test_x=clean_test,
    )

    assert "S1_Input" in decodability
    assert "S2_Voltage" in decodability
    assert "S6_Features" in decodability
    assert "Direct_Readout" in decodability
    assert decodability["attribution_category"] in (
        "INFORMATION_PRESERVED_AND_DECODED",
        "INFORMATION_PRESENT_BUT_NOT_LINEALLY_DECODABLE",
        "READOUT_FAILURE",
        "INFORMATION_ABSENT",
        "TRAINING_FAILURE",
        "UNDETERMINED",
    )
