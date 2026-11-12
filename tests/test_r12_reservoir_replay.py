"""SCAR-X Round 12: Unit Tests for Reservoir Replay Buffer.

Validates:
1. Capacity limit enforcement for M in {16, 64}.
2. Statistical uniformity of reservoir sampling across streaming inputs.
3. RNG determinism and reproducibility across runs.
4. Protection against future sample leakage.
5. Correct pre-insertion replay draw mechanics.
"""
import pytest
import numpy as np

from scarx.neural.r12_reservoir_replay import ReservoirReplayBuffer


def test_capacity_enforcement():
    for cap in (16, 64):
        buf = ReservoirReplayBuffer(capacity=cap, seed=123)
        assert len(buf) == 0
        assert buf.memory_bytes == 0

        # Feed 200 items
        for i in range(200):
            x = np.ones((20, 8), dtype=np.float32) * i
            buf.insert(x, i % 2)
            assert len(buf) <= cap

        assert len(buf) == cap
        assert buf.total_seen == 200
        # Memory overhead must be < 100 KB
        assert buf.memory_bytes < 100 * 1024


def test_no_leakage_and_pre_insertion_draw():
    buf = ReservoirReplayBuffer(capacity=16, seed=456)
    # Empty buffer returns None
    assert buf.sample_replay() is None

    # First item
    x1 = np.full((20, 8), 1.0, dtype=np.float32)
    assert buf.sample_replay() is None  # Drawn before insertion -> None

    buf.insert(x1, 1)
    assert len(buf) == 1

    # Second item
    x2 = np.full((20, 8), 2.0, dtype=np.float32)
    rep_x, rep_y = buf.sample_replay()  # Must strictly be x1, not x2!
    assert np.allclose(rep_x, 1.0)
    assert rep_y == 1

    buf.insert(x2, 0)
    assert len(buf) == 2


def test_rng_determinism():
    buf1 = ReservoirReplayBuffer(capacity=16, seed=789)
    buf2 = ReservoirReplayBuffer(capacity=16, seed=789)

    for i in range(100):
        x = np.random.randn(20, 8).astype(np.float32)
        rep1 = buf1.sample_replay()
        rep2 = buf2.sample_replay()
        if rep1 is not None and rep2 is not None:
            assert np.allclose(rep1[0], rep2[0])
            assert rep1[1] == rep2[1]
        
        acc1 = buf1.insert(x, i % 2)
        acc2 = buf2.insert(x, i % 2)
        assert acc1 == acc2

    # Final contents match exactly
    c1_x, c1_y = buf1.get_contents_copy()
    c2_x, c2_y = buf2.get_contents_copy()
    assert len(c1_x) == len(c2_x)
    for a, b in zip(c1_x, c2_x):
        assert np.allclose(a, b)
    assert c1_y == c2_y


def test_statistical_uniformity():
    """Verify that every element in a stream of N items has probability M/N of retention."""
    N = 200
    M = 20
    trials = 2000
    expected_prob = M / N  # 0.10

    counts = np.zeros(N, dtype=np.int32)
    for t in range(trials):
        buf = ReservoirReplayBuffer(capacity=M, seed=1000 + t)
        for i in range(N):
            x = np.array([[i]], dtype=np.float32)
            buf.insert(x, i)
        
        for x_item, y_val in zip(*buf.get_contents_copy()):
            counts[y_val] += 1

    empirical_probs = counts / trials
    # Every item's empirical retention probability should be close to 0.10 (+/- 0.035 with 99.9% confidence)
    max_dev = np.max(np.abs(empirical_probs - expected_prob))
    mean_dev = np.mean(np.abs(empirical_probs - expected_prob))
    assert max_dev < 0.035, f"Excessive reservoir bias: max dev = {max_dev:.4f}"
    assert mean_dev < 0.015, f"Excessive reservoir bias: mean dev = {mean_dev:.4f}"
