import json
import math
from pathlib import Path

import pytest

from scarx.runtime.experiment_status import credit_alignment_statistics

RUN = '20261110T080426Z-local-credit-alignment-audit-v1-db01fc'


def _write(root, run_id, payload):
    run = Path(root) / 'runs' / run_id
    run.mkdir(parents=True)
    (run / 'result.json').write_text(json.dumps(payload), encoding='utf-8')
    return run


def _payload(**over):
    base = {
        'decision': 'LOCAL_CREDIT_ALIGNMENT_DIAGNOSTIC_ONLY',
        'neural_training': False,
        'deployed': False,
        'model_sources': 15,
        'applied_counterfactual_evaluations': 12288,
        'alignment_rows': [
            {'seed': 36313, 'split': 'development', 'cases': 64, 'nll_before': 0.9,
             'delta_direction': -0.13, 'delta_signflip': 0.10, 'delta_random_mean': 0.01,
             'delta_permuted_mean': 0.004, 'gain_vs_random': 0.14, 'gain_vs_permuted': 0.13,
             'gain_vs_signflip': 0.24, 'alignment_fraction': 0.67,
             'norm_match_ratio_random': 4.9, 'norm_match_ratio_permuted': 2.1},
            {'seed': 36313, 'split': 'train', 'cases': 64, 'nll_before': 0.1,
             'delta_direction': -0.5, 'delta_signflip': 0.2, 'delta_random_mean': 0.02,
             'delta_permuted_mean': 0.01, 'gain_vs_random': 0.5, 'gain_vs_permuted': 0.5,
             'gain_vs_signflip': 0.7, 'alignment_fraction': 0.99,
             'norm_match_ratio_random': 5.0, 'norm_match_ratio_permuted': 2.0},
        ],
        'dev_macro': {'cases': 192, 'alignment_fraction': 0.635, 'gain_vs_signflip': 0.216,
                      'norm_match_ratio_random': 4.9, 'norm_match_ratio_permuted': 2.1},
        'dev_gain_vs_signflip_interval': {'mean': 0.216, 'lower95': 0.197,
                                          'upper95': 0.237, 'n_seeds': 3},
        'heldout': {'seeds': 3, 'mean_dev_delta': -0.000084,
                    'interval': {'mean': -0.000084, 'lower95': -0.00019, 'upper95': 0.0000002}},
        'flags': {'LOCAL_DIRECTION_ALIGNED': True, 'DIRECTION_UNINFORMATIVE': False,
                  'DIRECTION_CONFLICT_CONFIRMED': False, 'GAUSSIAN_CONTROL_OVERSHOOTS': True},
    }
    base.update(over)
    return base


def test_returns_bounded_typed_rows_and_flags(tmp_path):
    _write(tmp_path, RUN, _payload())
    result = credit_alignment_statistics(tmp_path, RUN)
    assert result['rows'] and all(r['split'] in ('train', 'development') for r in result['rows'])
    assert result['dev_macro']['alignment_fraction'] == pytest.approx(.635)
    assert result['flags']['LOCAL_DIRECTION_ALIGNED'] is True
    assert result['intervals']['gain_vs_signflip']['lower95'] == pytest.approx(.197)
    assert result['heldout']['mean_dev_delta'] < 0
    assert 'rách' not in json.dumps(result)  # never leaks a path


def test_rejects_non_diagnostic_decisions_and_any_training_claim(tmp_path):
    for bad in ({'decision': 'STOP_DIRECT_WORD_BRIDGE_V1'},
                {'neural_training': True},
                {'deployed': True}):
        run = _write(tmp_path, 'x' + str(abs(hash(str(bad))))[:6], _payload(**bad))
        assert credit_alignment_statistics(tmp_path, run.name) == {}


def test_superseded_runs_are_never_rendered(tmp_path):
    # A run that measured the wrong thing must not appear as a result in the UI.
    for bad in ({'do_not_cite': True},
                {'superseded_by': 'runs/20261110T080426Z-local-credit-alignment-audit-v1-db01fc',
                 'superseded_reason': 'measured applied norm after the update'}):
        run = _write(tmp_path, 's' + str(abs(hash(str(bad))))[:6], _payload(**bad))
        assert credit_alignment_statistics(tmp_path, run.name) == {}


def test_rejects_missing_or_malformed_sections(tmp_path):
    for bad in ({'alignment_rows': 'nope'}, {'dev_macro': None},
                {'alignment_rows': [{'seed': 'x', 'split': 'development'}]}):
        run = _write(tmp_path, 'y' + str(abs(hash(str(bad))))[:6], _payload(**bad))
        result = credit_alignment_statistics(tmp_path, run.name)
        assert result == {} or result['rows'] == []


def test_rejects_invalid_run_id_and_oversized_file(tmp_path):
    assert credit_alignment_statistics(tmp_path, '../../etc') == {}
    run = _write(tmp_path, RUN + 'z', _payload())
    (run / 'result.json').write_text('x' * 600000, encoding='utf-8')
    assert credit_alignment_statistics(tmp_path, run.name) == {}
