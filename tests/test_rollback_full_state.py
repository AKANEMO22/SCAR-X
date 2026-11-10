"""Rollback must restore every piece of neural state exactly.

Verified in isolation on 2026-11-10 (scripts/prove_rollback_full_state.py): a
rejected attempt leaves weights, neurons, traces, eligibility, topology,
vocabulary, RNG and clocks byte-identical, and behaviour from an identical start
is unchanged. Only event bookkeeping (trace sequence, dialogue counters) moves,
because an attempt is still an event worth recording.

This test locks that property in so a future change cannot silently break it.
The guard verdict is injected so the rollback path runs deterministically.
"""
import itertools
import json

import numpy as np
import pytest

from scarx.app.generation import Generation, LearningGuardRejected
from scarx.evaluation.product_acceptance import weights_sha256
from scarx.neural.unified import UnifiedCore

ARRAYS = ('w_in', 'w_rec', 'w_out', 'bias', 'stable_in', 'stable_rec',
          'stable_out', 'stable_bias', 'voltage', 'adaptation', 'spikes',
          'pre', 'post', 'features', 'elig_in', 'elig_rec', 'rows', 'indices')
PROBE = ['hi', 'chào bạn', 'bạn là ai', 'tạm biệt']


def neural_state(app):
    core = app.model.core
    snap = {name: np.ascontiguousarray(getattr(core, name)).tobytes() for name in ARRAYS}
    snap['vocabulary'] = tuple(app.model.vocabulary)
    snap['ticks'] = app.model.ticks
    snap['updates'] = core.updates
    snap['capacity_rng'] = json.dumps(app.model.capacity_rng.bit_generator.state, sort_keys=True)
    snap['app_rng'] = json.dumps(app.rng.bit_generator.state, sort_keys=True)
    return snap


def read_fresh(app, prompt):
    clone = UnifiedCore.loads(app.model.dumps())
    clone.core.reset_state()
    clone.last_active[:] = clone.ticks
    return clone.generate(prompt)['text']


@pytest.fixture
def app(tmp_path):
    session = Generation(tmp_path, background=False)
    yield session
    session.close()


def test_rejected_attempt_restores_all_neural_state_and_behaviour(app, monkeypatch):
    before = neural_state(app)
    sha_before = weights_sha256(app.model)
    outs_before = {p: read_fresh(app, p) for p in PROBE}

    app.dialogue.enqueue('probe', 'chào bạn', 'Chào bạn.', 'rollback test')
    app.dialogue.lessons[-1]['remaining'] = 1

    verdicts = itertools.cycle([[True], [False]])
    monkeypatch.setattr(app, '_guard_pairs', lambda _p: [('probe skill', 'target')])
    monkeypatch.setattr(app, '_guard_scores', lambda *_a: next(verdicts))

    record = app._online_microstep('probe')
    assert record['accepted'] is False
    assert record['rolled_back'] is True

    after = neural_state(app)
    for key in before:
        assert before[key] == after[key], f'{key} changed across a rejected attempt'

    assert weights_sha256(app.model) == sha_before
    for prompt in PROBE:
        assert read_fresh(app, prompt) == outs_before[prompt], f'behaviour changed for {prompt}'


def test_rollback_evidence_is_recorded_as_zero_weight_delta(app, monkeypatch):
    app.dialogue.enqueue('probe', 'cảm ơn', 'Không có gì.', 'rollback test')
    app.dialogue.lessons[-1]['remaining'] = 1
    verdicts = itertools.cycle([[True], [False]])
    monkeypatch.setattr(app, '_guard_pairs', lambda _p: [('probe skill', 'target')])
    monkeypatch.setattr(app, '_guard_scores', lambda *_a: next(verdicts))
    record = app._online_microstep('probe')
    assert record['weight_delta'] == {'w_in': 0.0, 'w_rec': 0.0, 'w_out': 0.0}
    assert record['attempted_weight_delta']['w_out'] >= 0.0
