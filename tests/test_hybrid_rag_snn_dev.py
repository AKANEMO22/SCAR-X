"""Contract/split and attribution regressions, not model-quality evidence."""

import copy
import numpy as np
from scarx.evaluation.hybrid_rag_snn_dev import (
    dataset,
    select_rule,
    select_bm25,
    pair_features,
    forgetting,
    FAMILIES,
)


def test_event_gold_is_supported_and_rejects_unverified_newer_value():
    d = dataset()
    for case in d["test"] + d["probes"]:
        gold = [r for r in case["docs"] if r["id"] == case["gold_id"]]
        if gold:
            assert (
                gold[0]["status"] == "approved" and gold[0]["entity"] == case["entity"]
            )
            assert gold[0]["value"] == case["gold"]
            if case["cutoff"]:
                assert gold[0]["time"] < case["cutoff"]
        else:
            assert case["gold"] is None
        assert select_rule(case) == case["gold_id"]


def test_three_splits_have_disjoint_entities_and_train_balanced_without_test_feedback():
    d = dataset()
    train = {i["case"]["entity"] for stage in d["train_stages"] for i in stage}
    test = {c["entity"] for c in d["test"]}
    probe = {c["entity"] for c in d["probes"]}
    assert not (train & test or train & probe or test & probe)
    assert len(d["train_stages"]) == 20 and len(d["test"]) == 48
    assert all(
        len(stage) == 24 and sum(i["target"] for i in stage) == 12
        for stage in d["train_stages"]
    )
    assert all(sum(c["family"] == f for c in d["test"]) == 12 for f in FAMILIES)


def test_candidate_order_is_irrelevant_for_rule_and_fts_only_reads_text():
    case = dataset()["test"][0]
    a = select_rule(case)
    case["docs"].reverse()
    assert select_rule(case) == a
    bm = select_bm25(case)
    assert bm is None or bm in {r["id"] for r in case["docs"]}


def test_current_encoder_ignores_provenance_metadata_and_labels_explicitly():
    case = dataset()["test"][0]
    doc = case["docs"][0]
    a = pair_features(case, doc)
    altered = copy.deepcopy(doc)
    altered.update(status="unverified", time=999, value=-1, entity="different")
    b = pair_features(dict(case, family="not-an-input", gold=9999), altered)
    np.testing.assert_array_equal(a, b)
    # This is a real limitation: metadata-invariant features cannot distinguish
    # contradictory metadata without a corresponding text change.


def test_forgetting_gate_is_joint_and_never_rewards_no_acquisition():
    curves = [
        {"families": dict.fromkeys(FAMILIES, 0.0)},
        {"families": dict.fromkeys(FAMILIES, 0.8)},
        {"families": dict.fromkeys(FAMILIES, 0.7)},
    ]
    result = forgetting(curves)
    assert 9.9 < result["mean_pp"] < 10.1 and not result["within_registered_limits"]
    flat = forgetting([curves[0]] * 20)
    assert flat["mean_pp"] == 0  # Passing retention alone says nothing about learning.
