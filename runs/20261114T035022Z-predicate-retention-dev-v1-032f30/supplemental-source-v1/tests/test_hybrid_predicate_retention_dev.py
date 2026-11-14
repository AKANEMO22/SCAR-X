import copy
import numpy as np
from scarx.evaluation.predicate_retention_dev import (
    dataset, make_case, pair_sequence, FAMILIES, independent_label, forgetting)


def test_queries_are_dual_verified_and_encoder_has_no_stage_or_label_input():
    for family in FAMILIES:
        for target in (0, 1):
            row = make_case(family, 0, "unit", target)
            assert independent_label(row) == target
            sequence = pair_sequence(row)
            assert sequence.shape == (16, 64) and sequence.dtype == np.float32
            changed = copy.deepcopy(row)
            changed.update(target=1-target, id="another", stage=999, future_label=1)
            np.testing.assert_array_equal(sequence, pair_sequence(changed))


def test_data_splits_entities_and_examples_are_disjoint_balanced():
    d = dataset()
    split_rows = [sum(d["train_stages"], []), d["probes"], d["final_DEV"]]
    ids = [{r["id"] for r in split} for split in split_rows]
    names = [{r["owner"] for r in split} for split in split_rows]
    assert len(d["train_stages"]) == 20 and all(len(r) == 16 for r in d["train_stages"])
    assert all(sum(r["target"] for r in stage) == 8 for stage in d["train_stages"])
    for a, b in ((0,1), (0,2), (1,2)):
        assert not ids[a] & ids[b] and not names[a] & names[b]
    assert len(d["probes"]) == 200 and len(d["final_DEV"]) == 400


def test_no_acquisition_is_not_a_retention_success_claim():
    flat = [{"by_family": dict.fromkeys(FAMILIES, .5)} for _ in range(20)]
    assert forgetting(flat)["mean_pp"] == 0
    # Separate acquisition/final>=65 gate is required; no false success here.
    flat[-1]["by_family"][FAMILIES[0]] = .4
    assert not forgetting(flat)["within_registered_limits"]
