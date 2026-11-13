import copy
import numpy as np
import pytest
from scarx.evaluation.hybrid_incident_encoder import observation_sequence


def observation():
    return {
        "tool": "read_file", "reported_status": "error", "error_type": "missing",
        "empty_result": False, "spillover": False, "streak": 3,
        "returned_epoch": 100.0,
        "probe": {"kind": "requested_path_binding", "read_ok": True,
                  "reported_not_found": False, "original_path_differs": True},
    }


def test_observed_probe_changes_input_and_history_order_is_preserved():
    a = observation()
    b = copy.deepcopy(a)
    b["probe"].update(read_ok=False, reported_not_found=True)
    x = observation_sequence("Check actual failure", [a, b], decision_epoch=200)
    y = observation_sequence("Check actual failure", [b, a], decision_epoch=200)
    assert x.shape == (16, 64) and x.dtype == np.float32
    assert not np.array_equal(x, y)
    np.testing.assert_array_equal(x[:8], y[8:])
    np.testing.assert_array_equal(x[8:], y[:8])


@pytest.mark.parametrize("key", ["gold", "cause", "future_success", "scope", "id"])
def test_gold_labels_future_and_identity_cannot_enter_encoder(key):
    row = observation()
    row[key] = "forbidden"
    with pytest.raises(ValueError):
        observation_sequence("Check actual failure", [row], decision_epoch=200)


def test_missing_probe_and_failed_probe_have_distinct_inputs():
    a = observation()
    a["probe"] = None
    b = observation()
    b["probe"].update(read_ok=False)
    assert not np.array_equal(
        observation_sequence("Check actual failure", [a], decision_epoch=200),
        observation_sequence("Check actual failure", [b], decision_epoch=200),
    )


def test_bounds_reject_unbounded_or_non_boolean_state():
    with pytest.raises(ValueError):
        observation_sequence("Check actual failure", [observation()] * 9, decision_epoch=200)
    row = observation()
    row["probe"]["read_ok"] = 1
    with pytest.raises(ValueError):
        observation_sequence("Check actual failure", [row], decision_epoch=200)


@pytest.mark.parametrize("epoch", [201, -1, float("nan"), float("inf"), True])
def test_future_or_invalid_observation_time_is_rejected(epoch):
    row = observation()
    row["returned_epoch"] = epoch
    with pytest.raises(ValueError):
        observation_sequence("Check actual failure", [row], decision_epoch=200)


def test_backwards_history_is_rejected_without_encoding_absolute_time():
    a, b = observation(), observation()
    b["returned_epoch"] = 99
    with pytest.raises(ValueError):
        observation_sequence("Check actual failure", [a, b], decision_epoch=200)
    np.testing.assert_array_equal(
        observation_sequence("Check actual failure", [a], decision_epoch=200),
        observation_sequence("Check actual failure", [b], decision_epoch=200),
    )
