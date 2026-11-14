import copy
import numpy as np
from scarx.evaluation.predicate_retention_dev import dataset, pair_sequence
from scarx.evaluation.predicate_binding_diagnostic import build_pairs, metrics


def test_counterfactuals_change_only_query_principal_or_arguments_not_record():
    data = dataset()
    original = copy.deepcopy(data["final_DEV"])
    pairs, excluded = build_pairs(original)
    assert original == data["final_DEV"]
    assert pairs and excluded
    for pair in pairs:
        a, b = pair["include"], pair["exclude"]
        assert a["record"] == b["record"]
        assert a["target"] == 1 and b["target"] == 0
        assert a["api"] == b["api"] and a["clock"] == b["clock"]
        if pair["contrast"] == "owner":
            assert a["args"] == b["args"] and a["owner"] != b["owner"]
        else:
            assert a["owner"] == b["owner"] and a["args"] != b["args"]
        changed = dict(a, id="not-an-input", target=0, pair_id="not-an-input")
        np.testing.assert_array_equal(pair_sequence(a), pair_sequence(changed))


def test_contrast_metric_does_not_count_always_positive_as_relationship_skill():
    base = dict(include_probability=.6, exclude_probability=.6, encoded_equal=False)
    always_yes = dict(base, include_prediction=1, exclude_prediction=1)
    good = dict(base, include_prediction=1, exclude_prediction=0)
    result = metrics([always_yes, good])
    assert result["both_correct"] == .5 and result["prediction_flip_rate"] == .5
    assert result["mean_probability_contrast"] == 0
