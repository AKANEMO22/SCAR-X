"""SCAR-X Round 9 Benchmark Integrity Assertions & Scaled Task Generators.

Strict invariants:
1. Input dimension strictly == 8 channels (zero implicit cue channels).
2. Spikes strictly binary {0.0, 1.0}.
3. Zero task boundary or oracle signals during continual evaluation.
4. Exactly matched update budgets across conditions.
5. Scaled multi-key associative retrieval (2, 4, 8, 16 keys).
6. Ceiling-stress binding tasks (eliminating the direct_accuracy >= 95% bypass).
"""
from __future__ import annotations

import numpy as np
from ..neural.core import Config, SpikingCore


def assert_input_dimension(x: np.ndarray, expected_d: int = 8) -> None:
    if x.ndim not in (2, 3):
        raise AssertionError(f"Expected 2D or 3D input array, got shape {x.shape}")
    d = x.shape[-1]
    if d != expected_d:
        raise AssertionError(
            f"Benchmark integrity violation: input dimension {d} != expected {expected_d}. "
            "Implicit task cue channels are strictly prohibited!"
        )


def assert_spikes_binary(spikes: np.ndarray) -> None:
    if not np.isin(spikes, [0.0, 1.0]).all():
        raise AssertionError("Spike outputs must be strictly binary {0.0, 1.0}")


def generate_strict_binding(seed: int, count: int, ood: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Key-value relational binding over temporal delay. Strictly 8 input channels."""
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 2
    x = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    for i in range(count):
        query = int(rng.integers(2))
        a = int(y[i]) ^ query
        values = (a, 1 - a)
        order = rng.permutation(2)
        positions = (0, 7) if ood else (int(rng.integers(1, 3)), int(rng.integers(4, 7)))
        for position, key in zip(positions, order):
            x[i, position, key] += 1.5
            x[i, position, 2 + values[key]] += 1.5
        x[i, 10, 4 + query] += 1.5
        x[i, 11, 7] += 1.0
    order = rng.permutation(count)
    x_out, y_out = x[order], y[order].astype(np.int32)
    assert_input_dimension(x_out, 8)
    return x_out, y_out


def generate_distractor_binding(
    seed: int, count: int, ood: bool = False, distractor_amp: float = 1.2
) -> tuple[np.ndarray, np.ndarray]:
    """Relational binding under mid-delay distractor pulses. Strictly 8 input channels."""
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
                x[i, dt, 6] += float(rng.choice([-distractor_amp, distractor_amp]))
        x[i, 10, 4 + query] += 1.5
        x[i, 11, 7] += 1.0
    order = rng.permutation(count)
    x_out, y_out = x[order], y[order].astype(np.int32)
    assert_input_dimension(x_out, 8)
    return x_out, y_out


def generate_xor_representation(seed: int, count: int, ood: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Non-linear XOR parity across 12 ticks. Strictly 8 input channels."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    y = np.arange(count) % 2
    for i in range(count):
        p1 = 0 if ood else int(rng.integers(1, 3))
        p2 = 8 if ood else int(rng.integers(4, 7))
        a = int(rng.integers(2))
        b = a ^ int(y[i])
        x[i, p1, a] += 1.5
        x[i, p2, 2 + b] += 1.5
        x[i, 11, 7] += 1.0
    order = rng.permutation(count)
    x_out, y_out = x[order], y[order].astype(np.int32)
    assert_input_dimension(x_out, 8)
    return x_out, y_out


def generate_order_representation(seed: int, count: int, ood: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Temporal sequence ordering across 12 ticks. Strictly 8 input channels."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    y = np.arange(count) % 2
    for i in range(count):
        p1 = 0 if ood else int(rng.integers(1, 3))
        p2 = 8 if ood else int(rng.integers(4, 7))
        a = int(y[i])
        b = 1 - a
        x[i, p1, a] += 1.5
        x[i, p2, b] += 1.5
        x[i, 11, 7] += 1.0
    order = rng.permutation(count)
    x_out, y_out = x[order], y[order].astype(np.int32)
    assert_input_dimension(x_out, 8)
    return x_out, y_out


def generate_ceiling_stress_binding(
    seed: int, count: int, delay_ticks: int = 16, distractor_amp: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Ceiling-breaking stress binding task. Longer delay (16 ticks) + heavy distractors."""
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 2
    seq_len = 4 + delay_ticks + 2
    x = rng.normal(0, 0.12, (count, seq_len, 8)).astype(np.float32)

    for i in range(count):
        query = int(rng.integers(2))
        a = int(y[i]) ^ query
        values = (a, 1 - a)
        # Presentation in first 4 ticks
        for key in range(2):
            pos = key * 2
            x[i, pos, key] += 1.5
            x[i, pos + 1, 2 + values[key]] += 1.5

        # Delay ticks with heavy intermittent distractor on channel 6
        for dt in range(4, 4 + delay_ticks):
            if rng.random() < 0.4:
                x[i, dt, 6] += float(rng.choice([-distractor_amp, distractor_amp]))

        # Query tick
        q_pos = 4 + delay_ticks
        x[i, q_pos, 4 + query] += 1.5
        # Response prompt tick
        x[i, seq_len - 1, 7] += 1.0

    order = rng.permutation(count)
    x_out, y_out = x[order], y[order].astype(np.int32)
    assert_input_dimension(x_out, 8)
    return x_out, y_out


def generate_multikey_associative_r9(
    seed: int, count: int, n_keys: int = 2, delay: int = 4, distractor: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled key-value associative retrieval across n_keys in {2, 4, 8, 16}. Strictly 8 channels."""
    if n_keys not in (2, 4, 8, 16):
        raise ValueError("n_keys must be 2, 4, 8, or 16")

    rng = np.random.default_rng(seed)
    seq_len = 2 * n_keys + delay + 2
    x = rng.normal(0, 0.05, (count, seq_len, 8)).astype(np.float32)
    y = np.arange(count) % 2

    for i in range(count):
        vals = [int(rng.integers(2)) for _ in range(n_keys)]
        query_k = int(rng.integers(n_keys))
        vals[query_k] = int(y[i])

        # Write phase: 2 ticks per key
        for k in range(n_keys):
            tick = 2 * k
            if n_keys == 2:
                x[i, tick, k] += 1.5
                x[i, tick + 1, 2 + vals[k]] += 1.5
            else:
                # Binary key coding across channels 0..3 (supports up to 16 keys)
                for bit in range(4):
                    if (k >> bit) & 1:
                        x[i, tick, bit] += 1.5
                    else:
                        x[i, tick, bit] -= 1.5
                # Value pulse on channel 4
                x[i, tick + 1, 4] += 1.5 if vals[k] == 1 else -1.5

        # Delay phase
        delay_start = 2 * n_keys
        if distractor:
            for dt in range(delay_start, delay_start + delay):
                if rng.random() < 0.4:
                    x[i, dt, 6] += float(rng.choice([-1.0, 1.0]))

        # Query phase
        q_tick = delay_start + delay
        if n_keys == 2:
            x[i, q_tick, 4 + query_k] += 1.5
        else:
            for bit in range(4):
                if (query_k >> bit) & 1:
                    x[i, q_tick, bit] += 1.5
                else:
                    x[i, q_tick, bit] -= 1.5

        # Response marker
        x[i, seq_len - 1, 7] += 1.0

    order = rng.permutation(count)
    x_out, y_out = x[order], y[order].astype(np.int32)
    assert_input_dimension(x_out, 8)
    return x_out, y_out
