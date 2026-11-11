import io
import json
import zipfile

import numpy as np
import pytest

from scarx.evaluation.dc_alif_eligibility import (
    DCALIFBPTTControl,
    DCALIFEligibility,
    dc_alif_finite_difference_check,
)
from scarx.evaluation.readout_eligibility import ReadoutEligibility
from scarx.evaluation.tasks21 import representation
from scarx.evaluation.visual_words import VisualWords
from scarx.neural.core import Config
from scarx.neural.unified import UnifiedCore


def make(condition: str = "correct", usefulness_beta: float = 0.0) -> DCALIFEligibility:
    m = DCALIFEligibility(
        Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=35101),
        condition,
        usefulness_beta=usefulness_beta,
    )
    calibration, _ = representation("xor", 15101, 128)
    m.calibrate(calibration)
    return m


def test_dc_alif_eligibility_matches_forward_finite_differences_on_fixed_presynaptic_tape() -> None:
    m = make()
    x, _ = representation("xor", 15103, 1)
    r = dc_alif_finite_difference_check(m, x[0])
    assert r["finite_difference_synapses"] == 64
    assert r["forward_state_exact"]
    assert r["max_absolute_error_e_f"] < 1e-5
    assert r["max_absolute_error_e_d"] < 1e-5


def test_zero_frozen_match_while_correct_shuffled_no_gate_hidden_change_and_rng_matches() -> None:
    models = [make(c) for c in ("correct", "no_gate", "shuffled", "zero", "frozen")]
    before = [m.core.w_in.copy() for m in models]
    x, y = representation("xor", 15107, 8)
    for seq, label in zip(x, y):
        for m in models:
            m.learn(seq, int(label))
    for index in (0, 1, 2):
        assert np.any(models[index].core.w_in != before[index]) and models[index].hidden_updates == 8
    for index in (3, 4):
        np.testing.assert_array_equal(models[index].core.w_in, before[index])
        assert models[index].hidden_updates == 0
    np.testing.assert_array_equal(models[3].core.w_out, models[4].core.w_out)
    np.testing.assert_array_equal(models[3].core.bias, models[4].core.bias)
    assert models[3].core.dumps() == models[4].core.dumps()
    assert all(m.signal_rng.bit_generator.state == models[0].signal_rng.bit_generator.state for m in models)
    assert not np.array_equal(models[0].core.w_in, models[2].core.w_in)
    assert not np.array_equal(models[0].core.w_in, models[1].core.w_in)


def test_partial_segment_optimizer_rng_checkpoint_exact_and_other_loaders_reject() -> None:
    m = make(usefulness_beta=4.0)
    x, y = representation("xor", 15109, 2)
    m.begin(True)
    for row in x[0, :5]:
        m.step(row)
    payload = m.dumps()
    copy_m = DCALIFEligibility.loads(payload)
    assert copy_m.dumps() == payload
    for row in x[0, 5:]:
        np.testing.assert_array_equal(m.step(row), copy_m.step(row))
    assert m.accumulate(int(y[0])) == copy_m.accumulate(int(y[0]))
    m.finish()
    copy_m.finish()
    assert m.dumps() == copy_m.dumps()
    assert m.learn(x[1], int(y[1])) == copy_m.learn(x[1], int(y[1])) and m.dumps() == copy_m.dumps()

    for loader in (UnifiedCore.loads, VisualWords.loads, ReadoutEligibility.loads):
        with pytest.raises(ValueError):
            loader(payload)

    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    meta = json.loads(parts["metadata.json"])
    meta["condition"] = "frozen"
    parts["metadata.json"] = json.dumps(meta).encode()
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as z:
        for name, value in parts.items():
            z.writestr(name, value)
    with pytest.raises(ValueError, match="checksum"):
        DCALIFEligibility.loads(bad.getvalue())


def test_no_target_forward_trace_memory_constant_and_learning_error_rolls_back_full_state() -> None:
    m = make(usefulness_beta=4.0)
    x, y = representation("xor", 15111, 2)
    before = m.dumps()
    with pytest.raises(TypeError):
        m.features(x[0], target=0)  # type: ignore[call-arg]
    assert m.dumps() == before
    n = m.memory_bytes()
    m.begin(True)
    for row in np.tile(x[0], (10, 1)):
        m.step(row)
    assert m.memory_bytes() == n and m.steps_in_segment == 120
    m.features(x[0])
    before = m.dumps()

    def fail(direction: np.ndarray) -> None:
        raise MemoryError("Injected after readout update")

    m._apply_direction = fail  # type: ignore[method-assign]
    with pytest.raises(MemoryError):
        m.learn(x[1], int(y[1]))
    assert m.dumps() == before and m.core.w_out is m.head.weights and m.core.bias is m.head.bias


def test_error_signal_stays_at_its_postsynaptic_row_and_clip_derivative_can_cut_feedback() -> None:
    m = make()
    m.head.weights.fill(0.0)
    m.head.weights[:, 0] = [0.5, -0.5]
    x, _ = representation("xor", 15113, 1)
    m.begin(True)
    for row in x[0]:
        m.step(row)
    m.accumulate(0)
    assert np.any(m.bank[0]) and not np.any(m.bank[1:])
    n = make()
    n.begin(True)
    for row in x[0]:
        n.step(row)
    n.head.mean[:] = -100.0
    n.accumulate(0)
    assert not np.any(n.bank)


def test_isolated_dc_alif_bptt_control_has_matched_frozen_head_and_exact_continuation() -> None:
    pytest.importorskip("torch")
    import torch

    torch.set_num_threads(1)
    a = make("frozen")
    b = DCALIFEligibility.loads(a.dumps())
    frozen = DCALIFBPTTControl(b)
    x, y = representation("xor", 15119, 4)
    for seq, label in zip(x, y):
        assert a.learn(seq, int(label)) == frozen.learn(seq, int(label))
    assert a.dumps() == b.dumps() and frozen.backward_passes == 0

    m = make()
    control = DCALIFBPTTControl(m)
    initial = m.core.w_in.copy()
    for seq, label in zip(x, y):
        control.learn(seq, int(label))
    assert np.any(initial != m.core.w_in) and control.backward_passes == 4 and control.max_graph_feature_error <= 1e-6
    copy_c = DCALIFBPTTControl.loads(control.dumps())
    assert copy_c.dumps() == control.dumps()
    assert control.learn(x[0], int(y[0])) == copy_c.learn(x[0], int(y[0])) and control.dumps() == copy_c.dumps()
    with pytest.raises(ValueError):
        DCALIFEligibility.loads(control.dumps())
    with pytest.raises(ValueError):
        ReadoutEligibility.loads(control.dumps())
