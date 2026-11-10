import numpy as np
import pytest
from scarx.evaluation.direct_words import DirectWords
from scarx.evaluation.direct_word_data import TARGETS
from scarx.neural.language import tokenize
from scarx.evaluation.word_training_audit import protected
from scarx.evaluation.credit_alignment import (
    Applier, case_direction, measure_case, norm_matched, permuted, heldout_direction,
)

VOCAB = ['<eos>', '<bos>'] + sorted({t for target in TARGETS.values() for t in tokenize(target)})


def model(condition='correct'):
    m = DirectWords(VOCAB, 36313, condition)
    m.calibrate(['hello', 'chào bạn', 'bạn là ai', 'thanks'])
    m.learn('hello', 'Hello.')
    return m


def test_direction_shape_and_source_integrity():
    m = model()
    before = protected(m)
    tokens = m.target_tokens('Hello.')
    direction = case_direction(m, 'hello', tokens)
    assert direction.shape == m.learner.bank.shape
    assert np.isfinite(direction).all()
    # Reading the direction must never touch the source checkpoint.
    assert protected(m) == before


def test_zero_op_is_deterministic_and_leaves_no_residue():
    m = model()
    applier = Applier(m)
    before = protected(m)
    first, applied_first = applier.measure('hello', 'Hello.', None)
    second, applied_second = applier.measure('hello', 'Hello.', None)
    assert first == second and applied_first == applied_second == 0.0
    assert protected(m) == before


def test_apply_is_reversible_so_cases_do_not_contaminate_each_other():
    m = model()
    applier = Applier(m)
    tokens = m.target_tokens('Hello.')
    direction = case_direction(m, 'hello', tokens)
    baseline, _ = applier.measure('hello', 'Hello.', None)
    _, applied = applier.measure('hello', 'Hello.', direction)
    after_apply, applied_after = applier.measure('hello', 'Hello.', None)
    assert after_apply == baseline  # restore is exact, not approximate
    assert applied_after == 0.0


def test_applied_norm_is_measured_on_the_real_weights_not_silently_zero():
    # Regression: snapshotting after the update made every applied norm 0.0 and
    # made norm-matched controls unverifiable. The applied norm must be positive
    # for a nonzero direction and exactly zero only for a genuine no-op.
    m = model()
    applier = Applier(m)
    direction = case_direction(m, 'hello', m.target_tokens('Hello.'))
    assert np.linalg.norm(direction.astype(np.float64)) > 0
    _, applied_none = applier.measure('hello', 'Hello.', None)
    _, applied_direction = applier.measure('hello', 'Hello.', direction)
    assert applied_none == 0.0
    assert applied_direction > 0.0


def test_permuted_control_is_far_better_matched_than_gaussian():
    # Adam normalises elementwise with per-element moment/variance left over from
    # training, so NO control can match the applied magnitude exactly. The
    # permuted control preserves the value multiset and lands within ~25%, while a
    # Gaussian of equal PRE-Adam norm is dense and overshoots by a large factor.
    # Both applied norms are always reported so the residual mismatch stays visible.
    m = model()
    applier = Applier(m)
    direction = case_direction(m, 'hello', m.target_tokens('Hello.'))
    _, applied_direction = applier.measure('hello', 'Hello.', direction)
    rng = np.random.default_rng(5)
    _, applied_permuted = applier.measure('hello', 'Hello.', permuted(direction, rng))
    _, applied_gaussian = applier.measure('hello', 'Hello.', norm_matched(direction, rng))
    permuted_error = abs(applied_permuted - applied_direction) / applied_direction
    gaussian_error = abs(applied_gaussian - applied_direction) / applied_direction
    assert permuted_error < .3
    assert gaussian_error > 2 * permuted_error
    # The permuted control must be a true permutation: same multiset, different order.
    shuffled = permuted(direction, np.random.default_rng(9))
    assert np.allclose(sorted(shuffled.ravel()), sorted(direction.ravel()))
    assert not np.array_equal(shuffled, direction)


def test_controls_are_norm_matched_to_the_direction():
    m = model()
    direction = case_direction(m, 'hello', m.target_tokens('Hello.'))
    rng = np.random.default_rng(7)
    control = norm_matched(direction, rng)
    assert control.shape == direction.shape
    assert np.isclose(np.linalg.norm(control.astype(np.float64)),
                      np.linalg.norm(direction.astype(np.float64)), rtol=1e-5)


def test_measure_case_reports_signed_deltas_and_alignment_flag():
    m = model()
    applier = Applier(m)
    direction = case_direction(m, 'hello', m.target_tokens('Hello.'))
    result = measure_case(applier, 'hello', 'Hello.', direction, np.random.default_rng(11), 3)
    assert result['delta_direction'] == pytest.approx(result['nll_direction'] - result['nll_before'])
    assert result['delta_signflip'] == pytest.approx(result['nll_signflip'] - result['nll_before'])
    assert result['aligned'] == (result['nll_direction'] < result['nll_before'])
    assert len(result['delta_random_values']) == 3
    # gain definitions must be internally consistent
    assert result['gain_vs_random'] == pytest.approx(
        np.mean(result['delta_random_values']) - result['delta_direction'])
    assert result['gain_vs_signflip'] == pytest.approx(
        result['delta_signflip'] - result['delta_direction'])


def test_heldout_direction_is_mean_and_rejects_misalignment():
    m = model()
    applier = Applier(m)
    tokens = m.target_tokens('Hello.')
    one = case_direction(m, 'hello', tokens)
    combined = heldout_direction(applier, [{'id': 'a'}], [one])
    assert combined.shape == applier.shape
    assert np.allclose(combined, one, atol=1e-6)
    with pytest.raises(ValueError):
        heldout_direction(applier, [], [])
    with pytest.raises(ValueError):
        heldout_direction(applier, [{'id': 'a'}, {'id': 'b'}], [one])


def test_alignment_refuses_non_correct_source_conditions():
    # Direction alignment is only meaningful on sources that actually applied credit.
    for condition in ('shuffled', 'zero', 'frozen'):
        m = model(condition)
        with pytest.raises(ValueError):
            Applier(m)
