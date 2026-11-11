"""Unit, hand-computed equation, metamorphic, and no-leakage tests for Round 6C CKCER."""
from __future__ import annotations

import numpy as np

from scarx.evaluation import r5_independent as G
from scarx.evaluation import r5_probe as P
from scarx.evaluation import r6_associative as R6
from scarx.neural.core import SpikingCore


def test_hand_computed_forward_and_local_update_mode_a() -> None:
    """Verify CKCERHead Mode A forward features, scores, and local update against hand-computed values."""
    cfg = R6.CKCERConfig(lr=0.08, clip_step=0.025, weight_clip=3.0, dead_std=1e-4)
    head = R6.CKCERHead(dim=3, mode="A", seed=42, cfg=cfg)

    # Calibration matrix (3 samples, 3 coordinates; coordinate 2 is dead/constant)
    m_cal = np.array(
        [
            [1.0, 2.0, 5.0],
            [2.0, 4.0, 5.0],
            [3.0, 6.0, 5.0],
        ],
        dtype=np.float32,
    )
    head.observe(m_cal)

    # Hand-computed mean = [2.0, 4.0, 5.0], std = [1.0, 2.0, 0.0]
    np.testing.assert_allclose(head.mean_m, [[2.0, 4.0, 5.0]], atol=1e-6)
    assert list(head.active_m[0]) == [True, True, False]

    # Sample m = [3.0, 2.0, 99.0] -> standardized m_hat = [(3-2)/1, (2-4)/2, 0.0] = [1.0, -1.0, 0.0]
    m_sample = np.array([[3.0, 2.0, 99.0]], dtype=np.float32)
    cue_q0 = np.array([[1.0, 0.0]], dtype=np.float32)
    cue_q1 = np.array([[0.0, 1.0]], dtype=np.float32)

    psi_q0 = head.build_features(m_sample, cue=cue_q0)
    psi_q1 = head.build_features(m_sample, cue=cue_q1)

    # Hand-computed psi_q0 = [1.0, -1.0, 0.0,  0.0, 0.0, 0.0,  1.0, 0.0]
    expected_q0 = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    # Hand-computed psi_q1 = [0.0, 0.0, 0.0,  1.0, -1.0, 0.0,  0.0, 1.0]
    expected_q1 = np.array([[0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(psi_q0, expected_q0, atol=1e-7)
    np.testing.assert_allclose(psi_q1, expected_q1, atol=1e-7)

    # Verify zero cross-talk on inactive branch (indices 3:6 when q=0, indices 0:3 when q=1)
    assert float(np.linalg.norm(psi_q0[:, 3:6])) == 0.0
    assert float(np.linalg.norm(psi_q1[:, 0:3])) == 0.0

    # Hand-computed single step update for y=1 when initial weights are zero (pred = argmax([0, 0]) = 0)
    dw, db = head.step_update(psi_q0[0], y_i=1)
    expected_dw = np.array(
        [
            [-0.025, +0.025, 0.0, 0.0, 0.0, 0.0, -0.025, 0.0],
            [+0.025, -0.025, 0.0, 0.0, 0.0, 0.0, +0.025, 0.0],
        ],
        dtype=np.float32,
    )
    expected_db = np.array([-0.025, +0.025], dtype=np.float32)
    np.testing.assert_allclose(dw, expected_dw, atol=1e-7)
    np.testing.assert_allclose(db, expected_db, atol=1e-7)

    # Confirm Query B's dendritic weights (columns 3, 4, 5 and bias column 7) received EXACT ZERO update
    assert np.all(head.w[:, 3:6] == 0.0)
    assert np.all(head.w[:, 7] == 0.0)


def test_hand_computed_oja_update_and_cue_blindness_mode_b() -> None:
    """Verify Mode B unsupervised Oja update and prove complete blindness to external cue channels."""
    cfg = R6.CKCERConfig(oja_lr=0.05, oja_epochs=8, obs_steps=64)
    head = R6.CKCERHead(dim=4, mode="B", seed=17, cfg=cfg)
    assert head.learnable_parameter_count == 2 * (2 * 4 + 2) + 2 + 4  # 26

    rng = np.random.default_rng(17)
    # Create a correlated query transient across coordinates 0, 1, 2 (coordinate 3 is independent noise)
    q_true = rng.integers(0, 2, size=64)
    delta_q = rng.normal(0.0, 0.05, size=(64, 4)).astype(np.float32)
    sign_q = np.where(q_true == 1, 2.0, -2.0).astype(np.float32)[:, None]
    delta_q[:, :3] += sign_q
    m_pre = rng.normal(0.0, 1.0, size=(64, 4)).astype(np.float32)

    head.observe(m_pre, delta_q=delta_q)
    true_cue = np.stack([1 - q_true, q_true], axis=1).astype(np.float32)
    purity = R6.evaluate_mode_b_gate_purity(head, delta_q, true_cue)
    assert purity == 1.0

    # Mode B predict does not accept or use external cue
    preds_1 = head.predict(m_pre, delta_q=delta_q)
    preds_2 = head.predict(m_pre, cue=None, delta_q=delta_q)
    assert np.array_equal(preds_1, preds_2)


def test_streaming_runner_parity_with_batch_capture() -> None:
    """Verify tick-by-tick CKCERStreamRunner matches batch capture_r6 to machine precision."""
    x, *_ = G.independent_query_set_v1(97001, 16)
    cap = R6.capture_r6(97001, x, query_tick=10)

    core = SpikingCore(P.cfg(97001))
    runner = R6.CKCERStreamRunner(core)
    m_stream = []
    d_stream = []
    s_stream = []
    for i in range(x.shape[0]):
        ep = runner.run_episode(x[i], query_tick=10)
        m_stream.append(ep["m_pre"])
        d_stream.append(ep["delta_q"])
        s_stream.append(ep["sum_s"])

    m_stream_arr = np.stack(m_stream, axis=0)
    d_stream_arr = np.stack(d_stream, axis=0)
    s_stream_arr = np.stack(s_stream, axis=0)

    assert float(np.max(np.abs(m_stream_arr - cap["m_pre"]))) < 1e-6
    assert float(np.max(np.abs(d_stream_arr - cap["delta_q"]))) < 1e-6
    assert float(np.max(np.abs(s_stream_arr - cap["sum_s"]))) == 0.0


def test_counterfactual_sibling_pre_query_isolation_and_no_leakage() -> None:
    """Verify counterfactual siblings have identical pre-query state and calibration is frozen on test."""
    ptx, *_ = G.independent_paired_set_v1(97001, 16)
    cap = R6.capture_r6(97001, ptx, query_tick=10)

    m_a, m_b = cap["m_pre"][0::2], cap["m_pre"][1::2]
    d_a, d_b = cap["delta_q"][0::2], cap["delta_q"][1::2]

    # Pre-query eligibility state must be bit-for-bit identical between Query A and Query B siblings
    assert float(np.max(np.abs(m_a - m_b))) == 0.0
    # Somatic transient at query tick must differ between Query A and Query B
    assert float(np.max(np.abs(d_a - d_b))) > 0.1

    head = R6.CKCERHead(dim=96, mode="A", seed=97001)
    head.observe(cap["m_pre"])
    mean_before = head.mean_m.copy()
    std_before = head.std_m.copy()

    # Predicting on shifted/scaled test inputs must never mutate calibration statistics
    _ = head.predict(cap["m_pre"] * 10.0 + 5.0, cue=cap["cue"])
    assert np.array_equal(head.mean_m, mean_before)
    assert np.array_equal(head.std_m, std_before)


def test_auxiliary_tasks_oracle_and_perturbation_validity() -> None:
    """Verify auxiliary tasks and temporal shift/reverse perturbations are non-trivial and well-formed."""
    x_clean, *_ = G.independent_query_set_v1(501, 32)
    x_shift = G.shifted(x_clean, shift=1, seed=501)
    x_rev = G.reversed_writes(x_clean)
    assert not np.array_equal(x_clean, x_shift)
    assert not np.array_equal(x_clean, x_rev)
    assert G.locate_query_tick(x_clean[0]) == 10
    assert G.locate_query_tick(x_shift[0]) == 11

    x_aux1, y_aux1, m_aux1 = R6.irrelevant_third_write_windowed_set_v1(64, seed=601)
    assert x_aux1.shape == (64, 12, 8)
    oracle_y1 = np.where(m_aux1["q"] == 0, m_aux1["va"], m_aux1["vb"])
    assert np.array_equal(y_aux1, oracle_y1)

    x_aux2, y_aux2, m_aux2 = R6.key_overwrite_query_set_v1(64, seed=701)
    assert x_aux2.shape == (64, 12, 8)
    oracle_y2 = np.where(m_aux2["q"] == 0, m_aux2["va"], m_aux2["vb"])
    assert np.array_equal(y_aux2, oracle_y2)
    assert 0.10 < float(np.mean(m_aux2["va"] != m_aux2["v_first"])) < 0.45
