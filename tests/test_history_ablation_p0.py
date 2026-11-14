"""Numerical/envelope checks only; zero actual model.learn calls in this file."""

import ast
import copy
import hashlib
import inspect
import io
import json
from pathlib import Path
import textwrap
import zipfile

import numpy as np
import pytest
from scarx.hybrid.learner import LocalALIF
from scarx.evaluation.history_ablation_p0 import (
    HistoryALIF, P0Budget, NormMismatch, history_delta, copied_learn_source,
)


def fixture():
    rng = np.random.default_rng(260313)
    return (rng.uniform(-2, 2, (128, 72)).astype(np.float32),
            rng.normal(0, .1, (128, 72)).astype(np.float32),
            np.full((128, 72), .01, np.float32),
            rng.normal(0, .2, (128, 72)).astype(np.float32))


def test_zero_history_first_step_is_exactly_the_current_rule_for_all_arms():
    w,h,e,d=fixture();h.fill(0)
    results=[history_delta(np,w,h,e,d,arm) for arm in ('current_hidden','current_only','matched')]
    for result in results[1:]:
        for a,b in zip(results[0][:3],result[:3]):np.testing.assert_array_equal(a,b)
    assert results[-1][3]['alpha']==1 and results[-1][3]['bisection_iterations']==0


def test_history_changes_only_trace_and_direction_not_energy_or_input_state():
    w,h,e,d=fixture();old=[a.copy() for a in (w,h,e,d)]
    current=history_delta(np,w,h,e,d,'current_hidden')
    only=history_delta(np,w,h,e,d,'current_only')
    np.testing.assert_array_equal(current[0],.9*h+.1*d)
    np.testing.assert_array_equal(only[0],.1*d)
    np.testing.assert_array_equal(current[1],only[1])
    assert not np.array_equal(current[2],only[2])
    for original,after in zip(old,(w,h,e,d)):np.testing.assert_array_equal(original,after)


def test_matched_norm_is_measured_after_both_clips_at_its_own_state():
    w,h,e,d=fixture();w.flat[:100]=3;w.flat[100:200]=-3
    trace,energy,delta,info=history_delta(np,w,h,e,d,'matched')
    target=history_delta(np,w,h,e,d,'current_hidden')[2]
    effect=lambda x:(np.clip(w+x,-3,3)-w).astype(np.float64)
    assert abs(np.linalg.norm(effect(delta))/np.linalg.norm(effect(target))-1)<=.01
    assert info['bisection_iterations']<=20 and 0<=info['alpha']<=256
    np.testing.assert_array_equal(trace,.1*d)
    np.testing.assert_array_equal(energy,.99*e+.01*d**2)


def test_zero_target_and_unreachable_support_are_not_silently_relaxed():
    w,h,e,d=fixture();h.fill(0);d.fill(0)
    _,_,delta,info=history_delta(np,w,h,e,d,'matched')
    assert not delta.any() and info['alpha']==0 and info['relative_norm_error']==0
    h.fill(1)
    with pytest.raises(NormMismatch,match='no attribution'):
        history_delta(np,w,h,e,d,'matched')


def test_copied_method_replaces_only_the_registered_three_statement_block():
    original=ast.parse(textwrap.dedent(inspect.getsource(LocalALIF.learn)))
    changed=ast.parse(copied_learn_source())
    expected=copy.deepcopy(original)
    for node in ast.walk(expected):
        if isinstance(node,ast.If) and ast.unparse(node.test)=="self.mode in ('learned', 'shuffled')":
            node.body[2:5]=ast.parse('delta = self._history_update(direction)').body
    assert ast.dump(expected,include_attributes=False)==ast.dump(changed,include_attributes=False)
    assert 'backward' not in copied_learn_source()


def test_research_checkpoint_is_quarantined_and_identity_checksum_are_strict(tmp_path):
    budget=P0Budget(tmp_path/'no-model-updates.jsonl')
    model=HistoryALIF(260313,'matched',budget);payload=model.dumps({'fixture':True})
    restored,controller=HistoryALIF.loads(payload,budget=budget,expected_arm='matched',device='cpu')
    assert restored.dumps(controller)==payload and controller=={'fixture':True}
    with pytest.raises((ValueError,KeyError)):LocalALIF.loads(payload)
    with pytest.raises(ValueError,match='identity'):HistoryALIF.loads(payload,budget=budget,expected_arm='current_only',device='cpu')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:members={n:archive.read(n) for n in archive.namelist()}
    def write(values):
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as archive:
            for name,raw in values:archive.writestr(name,raw)
        return out.getvalue()
    for name in ('arm','format','base_sha256','encoder','state_sha256'):
        metadata=json.loads(members['research.json']);metadata[name]='UNREGISTERED'
        bad=write([('research.json',json.dumps(metadata).encode()),('state.npz',members['state.npz'])])
        with pytest.raises(ValueError):HistoryALIF.loads(bad,budget=budget,expected_arm='matched',device='cpu')
    duplicate=write([*members.items(),('research.json',members['research.json'])])
    with pytest.raises(ValueError,match='duplicate'):HistoryALIF.loads(duplicate,budget=budget,expected_arm='matched',device='cpu')
    assert not budget.path.exists(), 'Serialization tests must not perform a learning attempt'


def test_persistent_budget_hash_chain_rejects_partial_or_modified_history(tmp_path):
    path=tmp_path/'counter-fixture-only.jsonl';budget=P0Budget(path)
    assert budget.charge('matched',260313,'gpu')==1
    assert P0Budget(path).charge('current_hidden',260337,'production_reference')==2
    assert len(budget.records())==2
    raw=path.read_bytes();path.write_bytes(raw[:-1])
    with pytest.raises(ValueError,match='Incomplete'):budget.charge('matched',260313,'gpu')
    path.write_bytes(raw.replace(b'260313',b'260337',1))
    with pytest.raises(ValueError,match='integrity'):budget.records()
    assert not path.with_suffix('.lock').exists()


def test_p0_kernel_rejects_precision_or_invalid_state():
    w,h,e,d=fixture()
    with pytest.raises(ValueError):history_delta(np,w.astype(np.float64),h,e,d,'matched')
    e.fill(0)
    with pytest.raises(ValueError):history_delta(np,w,h,e,d,'current_only')
