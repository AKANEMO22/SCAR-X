from scarx.evaluation.basic_knowledge_eval import grade, reference_case, summarize


def test_typed_numeric_rubric_rejects_substrings_wrong_values_and_unearned_abstention():
    fact={"known_domain":True,"value_type":"number","value":.25}
    row={"canonical_answer":"0.25.","language":"en"}
    assert grade("`0,25`",row,fact)["typed_semantic_correct"]
    for answer in ("0.5", "0.25 or 0.5", "The answer is 0.25.", "I do not know."):
        assert grade(answer,row,fact)["typed_semantic_correct"] is False
    assert grade("0.25",row,fact)["exact_normalized_correct"] is False


def test_unknown_is_not_scored_as_a_numeric_fact_and_has_separate_denominator():
    unknown={"canonical_answer":"Tôi chưa biết.","language":"vi"}
    g=grade("Tôi chưa biết.",unknown,{"known_domain":False})
    assert g["typed_semantic_correct"] is None and g["registered_abstention"]
    rows=[{"split":"unknown_development","grade":g},
          {"split":"development","grade":{"exact_normalized_correct":False,"typed_semantic_correct":False}}]
    result=summarize(rows)
    assert result["known_exact_accuracy"]==0 and result["unknown_registered_abstention_rate"]==1
    assert result["production_ready"] is False


def test_qwen_subset_never_includes_final_split_or_other_variants():
    row={"fact_id":"config_hidden","id":"config_hidden_p05","split":"development"}
    assert reference_case(row)
    assert not reference_case({**row,"split":"heldout_confirmation"})
    assert not reference_case({**row,"id":"config_hidden_p07"})
