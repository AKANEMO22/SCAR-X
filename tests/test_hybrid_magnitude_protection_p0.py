import numpy as np
import io
import zipfile
import json
import hashlib
import pytest
from scarx.evaluation.magnitude_protection_p0 import MagnitudeProtectionP0
from scarx.hybrid.learner import LocalALIF, features

SEQUENCE = features("Owned numeric P0, not an accuracy task", (.4, 1, 0))
EVIDENCE = {"kind": "machine_verifier", "id": "owned-unit-counterfactual-only"}


def test_initial_zeroimportance_step_preserves_original_credit_with_FP32_tolerance():
    a = MagnitudeProtectionP0()
    b = LocalALIF().to_device("gpu")
    a.learn(SEQUENCE, 2, evidence=EVIDENCE)
    b.learn(SEQUENCE, 2, evidence=EVIDENCE)
    for name in ("w_in", "w_rec", "w_out", "bias"):
        np.testing.assert_allclose(a.base.host(getattr(a.base.core, name)),
            b.host(getattr(b.core, name)), rtol=0, atol=3e-7)
    assert a.base.xp.count_nonzero(a.base.importance) > 0
    assert a.base.xp.count_nonzero(a.output_importance) > 0


def test_existing_importance_actually_attenuates_output_updates():
    a = MagnitudeProtectionP0()
    b = MagnitudeProtectionP0()
    b.output_importance.fill(20)
    before = a.base.core.w_out.copy()
    a.learn(SEQUENCE, 2, evidence=EVIDENCE)
    b.learn(SEQUENCE, 2, evidence=EVIDENCE)
    da = float(a.base.xp.linalg.norm(a.base.core.w_out - before))
    db = float(b.base.xp.linalg.norm(b.base.core.w_out - before))
    assert da > 0 and 0 < db < .02 * da


@pytest.mark.parametrize("mode", ["frozen", "readout_only"])
def test_protection_respects_original_frozen_and_readout_only_modes(mode):
    a = MagnitudeProtectionP0(mode=mode)
    initial_in, initial_rec = a.base.core.w_in.copy(), a.base.core.w_rec.copy()
    initial_out = a.base.core.w_out.copy()
    a.learn(SEQUENCE, 2, evidence=EVIDENCE)
    np.testing.assert_array_equal(a.base.host(initial_in), a.base.host(a.base.core.w_in))
    np.testing.assert_array_equal(a.base.host(initial_rec), a.base.host(a.base.core.w_rec))
    if mode == "frozen":
        np.testing.assert_array_equal(a.base.host(initial_out), a.base.host(a.base.core.w_out))
        assert a.base.updates == 0 and not a.base.xp.count_nonzero(a.output_importance)


def test_separate_checkpoint_and_actual_next_update_are_exact():
    a = MagnitudeProtectionP0()
    a.learn(SEQUENCE, 2, evidence=EVIDENCE)
    raw = a.dumps()
    b = MagnitudeProtectionP0.loads(raw)
    assert b.dumps() == raw
    assert b.base.device == "gpu"
    for model in (a, b):
        model.learn(SEQUENCE, 1, reward=-1, evidence=EVIDENCE)
        model.consolidate()
    assert a.dumps() == b.dumps()
    with pytest.raises((ValueError, KeyError)):
        LocalALIF.loads(raw)


def test_wrong_algorithm_and_extra_archive_member_are_rejected_before_loading():
    raw = MagnitudeProtectionP0(device="cpu").dumps()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        parts = {n:archive.read(n) for n in archive.namelist()}
    meta = json.loads(parts["metadata.json"])
    meta["constants"]["anchor_period"] = 1
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, value in parts.items():
            archive.writestr(name, json.dumps(meta).encode() if name == "metadata.json" else value)
    with pytest.raises(ValueError):
        MagnitudeProtectionP0.loads(output.getvalue(), device="cpu")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, value in parts.items():
            archive.writestr(name, value)
        archive.writestr("extra", b"x")
    with pytest.raises(ValueError):
        MagnitudeProtectionP0.loads(output.getvalue(), device="cpu")


def test_float64_importance_is_not_silently_cast_to_original_FP32():
    raw = MagnitudeProtectionP0(device="cpu").dumps()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        parts = {name:archive.read(name) for name in archive.namelist()}
    extra = io.BytesIO()
    np.savez(extra, output_importance=np.zeros((8, 128), np.float64),
             bias_importance=np.zeros(8, np.float32))
    parts["protection.npz"] = extra.getvalue()
    meta = json.loads(parts["metadata.json"])
    meta["extra_sha256"] = hashlib.sha256(parts["protection.npz"]).hexdigest()
    parts["metadata.json"] = json.dumps(meta).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, value in parts.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError):
        MagnitudeProtectionP0.loads(output.getvalue(), device="cpu")
