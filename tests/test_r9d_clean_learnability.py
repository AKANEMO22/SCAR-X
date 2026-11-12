"""Unit tests verifying clean learnability battery invariants."""
from __future__ import annotations

import numpy as np
import pytest

from scarx.evaluation.r9d_clean_battery import (
    verify_binding_oracle,
    generate_clean_verified_task_data,
    evaluate_clean_learnability,
)


def test_oracle_label_verification():
    x, y = generate_clean_verified_task_data("strict_binding", seed=42, count=64)
    assert len(x) == 64
    assert len(y) == 64
    for i in range(64):
        assert verify_binding_oracle(x[i]) == y[i]


def test_label_balance_and_shapes():
    for task in ("strict_binding", "xor", "order"):
        x, y = generate_clean_verified_task_data(task, seed=101, count=100)
        assert x.shape == (100, 12, 8)
        assert y.shape == (100,)
        # Exactly 50 0s and 50 1s
        assert np.sum(y == 0) == 50
        assert np.sum(y == 1) == 50


def test_zero_train_test_overlap():
    train_x, _ = generate_clean_verified_task_data("strict_binding", seed=10, count=50)
    test_x, _ = generate_clean_verified_task_data("strict_binding", seed=20, count=50)

    train_hashes = {hash(x.tobytes()) for x in train_x}
    test_hashes = {hash(x.tobytes()) for x in test_x}
    assert len(train_hashes.intersection(test_hashes)) == 0


def test_clean_battery_fast():
    # Fast check on seed 140011 with small training budget
    res = evaluate_clean_learnability(arch="no_gating", seeds=[140011], n_train=128, n_test=64)
    assert "strict_binding" in res
    assert "xor" in res
    assert "order" in res
    assert res["xor"]["mean_direct_acc"] > 0.70
    assert res["order"]["mean_direct_acc"] > 0.85
