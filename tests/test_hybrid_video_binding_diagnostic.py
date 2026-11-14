from scarx.evaluation.video_binding_diagnostic import explicit_rank, name_observed


def test_rank_parser_does_not_turn_citation_or_ambiguous_values_into_an_answer():
    assert explicit_rank("Adam Hall xếp thứ 9 [E1].") == 9
    assert explicit_rank("15") == 15
    assert explicit_rank("xếp thứ 9 hoặc thứ 15") is None
    assert explicit_rank("[E9]") is None
    assert explicit_rank("Ninth.") == 9
    assert explicit_rank("xếp thứ chín") == 9


def test_unresolved_phonetic_name_is_not_an_exact_entity_binding():
    assert name_observed("Adam Hall", "Adam Hall đứng thứ 9.")
    assert not name_observed("Adam Hall", "Aramhorn đứng thứ 9.")
    assert not name_observed("Adam Hall", "Adam Hallucination đứng thứ 9.")
