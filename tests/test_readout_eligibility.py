import io,json,zipfile
import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.visual_words import VisualWords
from scarx.evaluation.readout_eligibility import ReadoutEligibility
from scarx.evaluation.eligibility_validation import finite_difference_check
from scarx.evaluation.tasks21 import representation


def make(condition='correct'):
    m=ReadoutEligibility(Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=35101),condition)
    calibration,_=representation('xor',15101,128);m.calibrate(calibration);return m


def test_local_eligibility_matches_forward_finite_differences_on_fixed_presynaptic_tape():
    m=make();x,_=representation('xor',15103,1);r=finite_difference_check(m,x[0]);assert r['finite_difference_synapses']==64 and r['forward_state_exact']


def test_zero_frozen_match_while_correct_shuffled_hidden_change_and_rng_consumption_matches():
    models=[make(c) for c in ('correct','shuffled','zero','frozen')];before=[m.core.w_in.copy() for m in models];x,y=representation('xor',15107,8)
    for seq,label in zip(x,y):
        for m in models:m.learn(seq,int(label))
    for index in (0,1):assert np.any(models[index].core.w_in!=before[index]) and models[index].hidden_updates==8
    for index in (2,3):np.testing.assert_array_equal(models[index].core.w_in,before[index]);assert models[index].hidden_updates==0
    np.testing.assert_array_equal(models[2].core.w_out,models[3].core.w_out);np.testing.assert_array_equal(models[2].core.bias,models[3].core.bias)
    assert models[2].core.dumps()==models[3].core.dumps()
    assert all(m.signal_rng.bit_generator.state==models[0].signal_rng.bit_generator.state for m in models)
    assert not np.array_equal(models[0].core.w_in,models[1].core.w_in)


def test_partial_segment_optimizer_rng_checkpoint_exact_and_main_loaders_reject():
    m=make();x,y=representation('xor',15109,2);m.begin(True)
    for row in x[0,:5]:m.step(row)
    payload=m.dumps();copy=ReadoutEligibility.loads(payload);assert copy.dumps()==payload
    for row in x[0,5:]:np.testing.assert_array_equal(m.step(row),copy.step(row))
    assert m.accumulate(int(y[0]))==copy.accumulate(int(y[0]));m.finish();copy.finish();assert m.dumps()==copy.dumps()
    assert m.learn(x[1],int(y[1]))==copy.learn(x[1],int(y[1])) and m.dumps()==copy.dumps()
    for loader in (UnifiedCore.loads,VisualWords.loads):
        with pytest.raises(ValueError):loader(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:parts={name:z.read(name) for name in z.namelist()}
    meta=json.loads(parts['metadata.json']);meta['condition']='frozen';parts['metadata.json']=json.dumps(meta).encode();bad=io.BytesIO()
    with zipfile.ZipFile(bad,'w') as z:
        for name,value in parts.items():z.writestr(name,value)
    with pytest.raises(ValueError,match='checksum'):ReadoutEligibility.loads(bad.getvalue())


def test_no_target_forward_trace_memory_constant_and_learning_error_rolls_back_full_state():
    m=make();x,y=representation('xor',15111,2);before=m.dumps()
    with pytest.raises(TypeError):m.features(x[0],target=0)
    assert m.dumps()==before
    n=m.memory_bytes();m.begin(True)
    for row in np.tile(x[0],(10,1)):m.step(row)
    assert m.memory_bytes()==n and m.steps_in_segment==120
    m.features(x[0]);before=m.dumps()
    def fail(direction):raise MemoryError('Injected after readoutupdate')
    m._apply_direction=fail
    with pytest.raises(MemoryError):m.learn(x[1],int(y[1]))
    assert m.dumps()==before and m.core.w_out is m.head.weights and m.core.bias is m.head.bias


def test_error_signal_stays_at_its_postsynaptic_row_and_clip_derivative_can_cut_feedback():
    m=make();m.head.weights.fill(0);m.head.weights[:,0]=[.5,-.5];x,_=representation('xor',15113,1);m.begin(True)
    for row in x[0]:m.step(row)
    m.accumulate(0);assert np.any(m.bank[0]) and not np.any(m.bank[1:])
    n=make();n.begin(True)
    for row in x[0]:n.step(row)
    n.head.mean[:]=-100.;n.accumulate(0);assert not np.any(n.bank)


def test_isolated_bptt_control_has_matched_frozen_head_and_exact_optimizer_continuation():
    pytest.importorskip('torch')
    from scarx.evaluation.controls import EligibilityLearnabilityControl,torch
    torch.set_num_threads(1);a=make('frozen');b=ReadoutEligibility.loads(a.dumps());frozen=EligibilityLearnabilityControl(b);x,y=representation('xor',15119,4)
    for seq,label in zip(x,y):assert a.learn(seq,int(label))==frozen.learn(seq,int(label))
    assert a.dumps()==b.dumps() and frozen.backward_passes==0
    m=make();control=EligibilityLearnabilityControl(m);initial=m.core.w_in.copy()
    for seq,label in zip(x,y):control.learn(seq,int(label))
    assert np.any(initial!=m.core.w_in) and control.backward_passes==4 and control.max_graph_feature_error<=1e-6
    copy=EligibilityLearnabilityControl.loads(control.dumps());assert copy.dumps()==control.dumps()
    assert control.learn(x[0],int(y[0]))==copy.learn(x[0],int(y[0])) and control.dumps()==copy.dumps()
    with pytest.raises(ValueError):ReadoutEligibility.loads(control.dumps())
