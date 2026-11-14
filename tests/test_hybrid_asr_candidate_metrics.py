import pytest
from scarx.evaluation.asr_candidate_metrics import word_error, paired_language


def test_normalization_preserves_accent_meaning_and_counts_edits():
    assert word_error("CHÀO, bạn!", "chào bạn")["word_edits"] == 0
    assert word_error("chao ban", "chào bạn")["word_edits"] == 2
    assert word_error("", "one two")["word_edits"] == 2
    assert word_error("one extra two", "one two")["word_edits"] == 1
    with pytest.raises(ValueError):
        word_error("anything", "")


def test_language_measure_is_micro_not_unweighted_average_of_WER():
    rows = [
        {"base": {"word_edits": 1, "reference_words": 1},
         "small": {"word_edits": 0, "reference_words": 1}},
        {"base": {"word_edits": 0, "reference_words": 9},
         "small": {"word_edits": 0, "reference_words": 9}},
    ]
    result = paired_language(rows, bootstrap=100)
    assert result["base_micro_WER"] == .1 and result["small_micro_WER"] == 0
    assert result["paired_utterance_delta_CI95"][1] <= 0
    assert paired_language(rows, bootstrap=100) == result


def test_reference_mismatch_cannot_create_false_paired_improvement():
    with pytest.raises(ValueError):
        paired_language([{"base": {"word_edits": 1, "reference_words": 2},
                         "small": {"word_edits": 0, "reference_words": 3}}])
