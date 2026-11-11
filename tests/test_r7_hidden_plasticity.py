"""P0 Numerical & Architectural Verification Tests for Round 7 Hidden Plasticity (`EW-HE` / `CG-MHE` / `CA-MSC`)."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from scarx.neural.core import Config, SpikingCore
from scarx.evaluation.readout_eligibility import spike_slope
from scarx.evaluation.r7_hidden_plasticity import (
    EventWindowedHomeostaticEligibility,
    generate_p1_task,
)


def test_p0_1_exact_forward_parity_with_spiking_core() -> None:
    """Verify EventWindowedHomeostaticEligibility forward pass is bit-identical to SpikingCore."""
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=140011, anchor_strength=0.0)
    core = SpikingCore(cfg)
    model = EventWindowedHomeostaticEligibility(cfg, "learned_ew_he")

    seqs, _ = generate_p1_task("strict_binding", 140011, 16)
    for seq in seqs:
        core.reset_state()
        model.reset_episode()
        for row in seq:
            core.step(row, track=False)
            model.step(row, track=True)
            assert np.array_equal(core.voltage, model.core.voltage)
            assert np.array_equal(core.adaptation, model.core.adaptation)
            assert np.array_equal(core.spikes, model.core.spikes)
            assert np.array_equal(core.features, model.core.features)


def test_p0_2_anchored_finite_difference_sensitivity() -> None:
    """Verify own-neuron analytical sensitivity e_f (gamma_e=0.25) matches central finite differences
    of the anchored smooth own-neuron surrogate on a fixed presynaptic tape."""
    from scarx.evaluation.eligibility_validation import fixed_presynaptic_tape

    def scalar_anchored_ew_he(
        weights: np.ndarray,
        presyn: np.ndarray,
        observed: np.ndarray,
        baseline_weights: np.ndarray,
        config: Config,
        gamma_e: float = 0.25,
    ) -> float:
        c = config
        v = a = 0.0
        margins = []
        for x, z in zip(presyn, observed):
            u = c.leak * v + float(np.dot(baseline_weights, x))
            theta = c.threshold + c.adaptation_strength * a
            margins.append((u - theta) / 0.4)
            v = u - float(z) * theta
            a = c.adaptation_decay * a + float(z)
        v = a = f = 0.0
        for x, z, base in zip(presyn, observed, margins):
            u = c.leak * v + float(np.dot(weights, x))
            theta = c.threshold + c.adaptation_strength * a
            actual = 1.0 / (1.0 + np.exp(-np.clip((u - theta) / 0.4, -20.0, 20.0)))
            anchor = 1.0 / (1.0 + np.exp(-np.clip(base, -20.0, 20.0)))
            local = float(z) + actual - anchor
            v = u - local * theta
            a = c.adaptation_decay * a + local
            f = gamma_e * f + (1.0 - gamma_e) * local
        return f

    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=140011, anchor_strength=0.0)
    model = EventWindowedHomeostaticEligibility(cfg, "learned_ew_he")
    seqs, _ = generate_p1_task("strict_binding", 140011, 1)
    seq = seqs[0]

    presyn, spikes, baseline = fixed_presynaptic_tape(model.core, seq)
    model.reset_episode()
    for row in seq:
        model.step(row, track=True)

    for name in ("voltage", "adaptation", "spikes", "features"):
        np.testing.assert_array_equal(getattr(model.core, name), getattr(baseline, name))

    eps = 1e-5
    for j in (0, 7, 19, 37):
        weights = np.concatenate((model.core.w_in[j], model.core.w_rec[j])).astype(np.float64)
        numeric = []
        for i in range(len(weights)):
            plus = weights.copy()
            minus = weights.copy()
            plus[i] += eps
            minus[i] -= eps
            val_p = scalar_anchored_ew_he(plus, presyn[:, j], spikes[:, j], weights, cfg, gamma_e=0.25)
            val_m = scalar_anchored_ew_he(minus, presyn[:, j], spikes[:, j], weights, cfg, gamma_e=0.25)
            numeric.append((val_p - val_m) / (2.0 * eps))
        numeric_arr = np.asarray(numeric, dtype=np.float64)
        trace_arr = model.e_f[j].astype(np.float64)
        np.testing.assert_allclose(trace_arr, numeric_arr, atol=1e-5, rtol=1e-3)


def test_p0_3_zero_signal_equals_frozen_hidden_bit_identical() -> None:
    """Verify condition='zero_signal' leaves hidden weights untouched and matches 'frozen_hidden' bit-for-bit."""
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=140011, anchor_strength=0.0)
    cal_x, _ = generate_p1_task("strict_binding", 140111, 128)
    tr_x, tr_y = generate_p1_task("strict_binding", 140211, 64)
    te_x, _ = generate_p1_task("strict_binding", 140311, 64)

    m_frozen = EventWindowedHomeostaticEligibility(cfg, "frozen_hidden")
    m_zero = EventWindowedHomeostaticEligibility(cfg, "zero_signal")
    m_frozen.calibrate(cal_x)
    m_zero.calibrate(cal_x)

    for seq, y in zip(tr_x, tr_y):
        m_frozen.learn_episode(seq, int(y))
        m_zero.learn_episode(seq, int(y))

    assert np.array_equal(m_zero.core.w_in, m_zero.w_in_0)
    assert np.array_equal(m_zero.core.w_rec, m_zero.w_rec_0)
    assert np.array_equal(m_frozen.core.w_in, m_zero.core.w_in)
    assert np.array_equal(m_frozen.core.w_rec, m_zero.core.w_rec)
    assert np.array_equal(m_frozen.core.w_out, m_zero.core.w_out)
    assert np.array_equal(m_frozen.core.bias, m_zero.core.bias)
    assert np.array_equal(m_frozen.predict_many(te_x), m_zero.predict_many(te_x))


def test_p0_4_bounded_memory_and_zero_tape_growth() -> None:
    """Verify memory footprint is strictly bounded (< 100 KB) and does not grow with episode count."""
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=140011, anchor_strength=0.0)
    model = EventWindowedHomeostaticEligibility(cfg, "learned_ew_he")
    cal_x, _ = generate_p1_task("xor", 140111, 128)
    tr_x, tr_y = generate_p1_task("xor", 140211, 64)
    model.calibrate(cal_x)
    bytes_before = model.memory_bytes()
    for seq, y in zip(tr_x, tr_y):
        model.learn_episode(seq, int(y))
    bytes_after = model.memory_bytes()
    assert bytes_before == bytes_after
    assert bytes_after < 100_000


def test_p0_5_checkpoint_save_load_exact_continuation(tmp_path: Path) -> None:
    """Verify saving and loading mid-stream produces bit-identical continuation across future updates."""
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=140011, anchor_strength=0.0)
    cal_x, _ = generate_p1_task("strict_binding", 140111, 128)
    tr_x, tr_y = generate_p1_task("strict_binding", 140211, 32)
    te_x, _ = generate_p1_task("strict_binding", 140311, 32)

    m1 = EventWindowedHomeostaticEligibility(cfg, "learned_ew_he")
    m1.calibrate(cal_x)
    for seq, y in zip(tr_x[:16], tr_y[:16]):
        m1.learn_episode(seq, int(y))

    ckpt_path = tmp_path / "r7_ew_he.npz"
    m1.save(ckpt_path)
    m2 = EventWindowedHomeostaticEligibility.load(ckpt_path)

    for seq, y in zip(tr_x[16:], tr_y[16:]):
        loss1 = m1.learn_episode(seq, int(y))
        loss2 = m2.learn_episode(seq, int(y))
        assert loss1 == loss2

    assert np.array_equal(m1.core.w_in, m2.core.w_in)
    assert np.array_equal(m1.core.w_rec, m2.core.w_rec)
    assert np.array_equal(m1.core.w_out, m2.core.w_out)
    assert np.array_equal(m1.core.bias, m2.core.bias)
    assert np.array_equal(m1.syn_trace, m2.syn_trace)
    assert np.array_equal(m1.syn_energy, m2.syn_energy)
    assert np.array_equal(m1.predict_many(te_x), m2.predict_many(te_x))
