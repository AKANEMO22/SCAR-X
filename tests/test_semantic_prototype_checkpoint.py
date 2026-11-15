"""Strict research schema must not become a silently compatible live model."""
import io
import zipfile
import numpy as np
import pytest
from scarx.evaluation.semantic_prototype_alif import SemanticALIF
from scarx.hybrid.learner import LocalALIF


def rewrite(payload, key, value):
    with np.load(io.BytesIO(payload), allow_pickle=False) as old:
        arrays = {k: old[k] for k in old.files}
    arrays[key] = value
    out = io.BytesIO()
    np.savez(out, **arrays)
    return out.getvalue()


def test_research_not_loadable_as_live_and_reverse():
    payload = SemanticALIF(260113, 'prototype', device='cpu').dumps()
    with pytest.raises(ValueError):
        LocalALIF.loads(payload)
    with pytest.raises(ValueError):
        SemanticALIF.loads(LocalALIF().dumps(), device='cpu')


@pytest.mark.parametrize('key,value', [
    ('counts', np.zeros(128, dtype=np.float64)),
    ('counts', np.full(128, -1, dtype=np.int64)),
    ('counts', np.ones(128, dtype=np.int64)),
    ('initial_input', np.zeros((128, 64), dtype=np.float64)),
    ('initial_recurrent', np.full((128, 8), np.nan, dtype=np.float32)),
])
def test_malformed_checkpoint_rejected(key, value):
    payload = SemanticALIF(260113, 'prototype', device='cpu').dumps()
    with pytest.raises(ValueError):
        SemanticALIF.loads(rewrite(payload, key, value), device='cpu')


def test_duplicate_member_rejected():
    payload = SemanticALIF(260113, 'prototype', device='cpu').dumps()
    out = io.BytesIO(payload)
    with zipfile.ZipFile(out, 'a') as archive:
        archive.writestr('counts.npy', b'invalid')
    with pytest.raises(ValueError, match='Duplicate'):
        SemanticALIF.loads(out.getvalue(), device='cpu')


def test_cold_state_metadata_and_weights_exact_without_new_forward():
    model = SemanticALIF(260137, 'shuffled', device='cpu')
    controller = {'replay': [4, 9], 'seen': 12, 'stage': 1}
    clone, state = SemanticALIF.loads(model.dumps(controller), device='cpu')
    assert state == controller
    assert clone.signal_rng.bit_generator.state == model.signal_rng.bit_generator.state
    assert clone.dumps(controller) == model.dumps(controller)
