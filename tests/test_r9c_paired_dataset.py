"""Unit tests for the R9c paired dataset generator."""
from __future__ import annotations

import numpy as np
import pytest
from scarx.evaluation.r9c_paired_dataset import (
    PAIRED_CONDITIONS,
    generate_paired_dataset,
)


def test_paired_dataset_shapes_and_keys():
    batch = generate_paired_dataset(seed=42, count=64, ood=False)
    assert len(batch.metadata) == 64
    assert len(batch.labels) == 64
    assert set(batch.sequences.keys()) == set(PAIRED_CONDITIONS)

    for cond, seqs in batch.sequences.items():
        assert seqs.shape == (64, 12, 8)
        assert np.all(np.isfinite(seqs))


def test_paired_dataset_bit_identical_stimuli():
    batch = generate_paired_dataset(seed=123, count=50, ood=False)
    clean = batch.sequences["clean"]

    for cond in ("weak_distractor", "medium_distractor", "strong_distractor", "swapped_channel", "shifted_time"):
        seqs = batch.sequences[cond]
        for i, meta in enumerate(batch.metadata):
            p1, p2 = meta.positions
            # Stimulus positions must be bit-identical on stimulus channels 0..3
            np.testing.assert_array_equal(seqs[i, p1, :4], clean[i, p1, :4])
            np.testing.assert_array_equal(seqs[i, p2, :4], clean[i, p2, :4])
            # Query tick 10 must be bit-identical on query channels 4..5
            np.testing.assert_array_equal(seqs[i, 10, :6], clean[i, 10, :6])
            # Marker tick 11 must be bit-identical across all 8 channels
            np.testing.assert_array_equal(seqs[i, 11, :], clean[i, 11, :])


def test_paired_dataset_proportional_amplitudes():
    batch = generate_paired_dataset(seed=999, count=30, ood=False)
    clean = batch.sequences["clean"]
    weak = batch.sequences["weak_distractor"]
    med = batch.sequences["medium_distractor"]
    strong = batch.sequences["strong_distractor"]

    diff_w = weak - clean
    diff_m = med - clean
    diff_s = strong - clean

    # For standard ticks where distractor occurs on channel 6
    for i, meta in enumerate(batch.metadata):
        for dt in (3, 7, 8):
            if dt not in meta.positions and dt < 10:
                val_w = diff_w[i, dt, 6]
                val_m = diff_m[i, dt, 6]
                val_s = diff_s[i, dt, 6]
                if abs(val_m) > 0.01:
                    # Signs must match
                    assert np.sign(val_w) == np.sign(val_m) == np.sign(val_s)
                    # Amplitudes must be exactly 0.4, 1.2, 2.5
                    assert pytest.approx(abs(val_w), abs=1e-5) == 0.4
                    assert pytest.approx(abs(val_m), abs=1e-5) == 1.2
                    assert pytest.approx(abs(val_s), abs=1e-5) == 2.5


def test_paired_dataset_determinism():
    batch1 = generate_paired_dataset(seed=777, count=20, ood=False)
    batch2 = generate_paired_dataset(seed=777, count=20, ood=False)

    np.testing.assert_array_equal(batch1.labels, batch2.labels)
    for cond in PAIRED_CONDITIONS:
        np.testing.assert_array_equal(batch1.sequences[cond], batch2.sequences[cond])
    assert [m.stimulus_hash for m in batch1.metadata] == [m.stimulus_hash for m in batch2.metadata]
