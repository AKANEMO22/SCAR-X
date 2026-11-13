"""Contract supervision and checkpoint purpose; not real adapter-utility proof."""

import copy
import pytest
from scarx.evaluation.adapter_policy import FIELDS, oracle, sequence, LifecycleALIF


def state():
    return dict(zip(FIELDS, (0, 1, 0, 0, 0.3, 0, 1, 0, 0, 300, 9, 100, 1, 1, 0, 1)))


def test_same_request_with_fault_or_missing_proof_cannot_request_training():
    s = state()
    assert oracle(s) == "propose_adapter"
    assert oracle(s | {"pipeline_fault": 1}) == "repair"
    assert oracle(s | {"verified": 0}) == "verify"
    assert oracle(s | {"ephemeral": 1, "memory_available": 1}) == "memory"
    assert oracle(s | {"headroom_mb": 149}) == "verify"
    assert oracle(s | {"idle": 0}) == "verify"


def test_reuse_requires_compatibility_provenance_gain_and_qualification():
    s = state() | {
        "failure_count": 0,
        "verified_examples": 0,
        "adapter_present": 1,
        "qualified": 1,
        "adapter_gain_pp": 6,
    }
    assert oracle(s) == "reuse_adapter"
    for change in (
        {"compatible": 0},
        {"qualified": 0},
        {"provenance": 0},
        {"adapter_gain_pp": 4},
        {"regression_pp": 4},
    ):
        assert oracle(s | change) == "verify"


def test_policy_snapshot_exact_next_update_and_not_a_serving_controller():
    m = LifecycleALIF(270113)
    row = {
        "id": "fixture",
        "prompt": "Format the next records",
        "state": state(),
        "target": "propose_adapter",
    }
    m.learn(row)
    data = m.dumps()
    other = LifecycleALIF.loads(data)
    assert m.learn(row) == other.learn(row)
    assert m.dumps() == other.dumps()
    from scarx.hybrid.learner import LocalALIF

    generic = LocalALIF().dumps({"goals": {}})
    with pytest.raises(ValueError):
        LifecycleALIF.loads(generic)


def test_policy_rejects_unmeasured_nonfinite_fields():
    with pytest.raises(ValueError):
        oracle({"verified": 1})
    with pytest.raises(ValueError):
        sequence("x", state() | {"base_accuracy": float("nan")})
