"""CPU-free training boundary and strict malformed checkpoint regression."""
import io
import json
from unittest.mock import patch
import numpy as np
import pytest
from scarx.evaluation.semantic_credit_alif import CreditALIF, FORMAT, CODEC


def payload(meta=None, model=None):
    default = {'format': FORMAT, 'codec': CODEC, 'seed': 260113, 'arm': 'hidden',
               'eta': .05, 'updates': 0, 'hidden_updates': 0, 'controller': {}}
    default.update(meta or {})
    out = io.BytesIO()
    np.savez(out, metadata=np.frombuffer(json.dumps(default).encode(), dtype=np.uint8),
             model=np.zeros(2, dtype=np.uint8) if model is None else model)
    return out.getvalue()


def test_cpu_constructor_and_restore_reject_before_backend_allocation():
    with patch('scarx.evaluation.semantic_credit_alif.initial_model', side_effect=AssertionError('must not initialize')):
        with pytest.raises(ValueError, match='GPU-only'):
            CreditALIF(260113, 'hidden', device='cpu')
    with pytest.raises(ValueError, match='GPU-only'):
        CreditALIF.loads(payload(), device='cpu')


@pytest.mark.parametrize('meta', [
    {'format': 'scarx-hybrid-alif-v2'}, {'codec': 'signed-hash64-v1'},
    {'updates': -1}, {'updates': True}, {'hidden_updates': 1},
    {'arm': 'frozen', 'updates': 1}, {'eta': .3}, {'seed': True},
])
def test_bad_metadata_rejected_before_any_GPU_or_CPU_model(meta):
    with patch('scarx.evaluation.semantic_credit_alif.LocalALIF.loads', side_effect=AssertionError('must not restore')):
        with pytest.raises(ValueError):
            CreditALIF.loads(payload(meta))


def test_float_model_storage_rejected():
    with pytest.raises(ValueError, match='byte arrays'):
        CreditALIF.loads(payload(model=np.zeros(2, dtype=np.float32)))


def test_learn_rejects_cpu_fallback_before_counting_an_attempt():
    obj = CreditALIF.__new__(CreditALIF)
    obj.model = type('Fake', (), {'device': 'cpu'})()
    with pytest.raises(RuntimeError, match='no silent CPU'):
        obj.learn(np.ones(64, dtype=np.float32), 0)


def test_summary_uses_JSON_native_booleans_for_both_gate_outcomes():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts/run_semantic_contrastive_credit_dev.py'
    spec = importlib.util.spec_from_file_location('credit_dev_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for value in (.8, .2):
        report = module.summarize_curve(np.full((20, 6), value))
        assert type(report['pilot_acquisition_and_retention_gate']) is bool
        json.dumps(report, allow_nan=False)
