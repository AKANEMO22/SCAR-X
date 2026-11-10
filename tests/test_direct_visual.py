import io,json,zipfile
import numpy as np
import pytest
from scarx.evaluation.direct_visual import DirectVisualWords
from scarx.evaluation.visual_words import VisualWords
from scarx.evaluation.vision_fixtures import scenes,render
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.readout_eligibility import ReadoutEligibility

VOCAB=['<eos>','<bos>','red','green','circle','square','hình','tròn']


def make(method='correct'):
    m=DirectVisualWords(VOCAB,37313,method);image=render(scenes()[0]);m.calibrate([{'prompt':'What color?','image':image}]);return m,image


def test_actual_forward_localhead_and_ablated_inputs_match32_without_hiddenupdates():
    for method in ('zero','frozen','no_image','no_question'):
        m,image=make(method);baseline=VisualWords(VOCAB,37313,method if method.startswith('no_') else 'full');baseline.calibrate([{'prompt':'What color?','image':image}])
        np.testing.assert_array_equal(m.head.mean,baseline.head.mean);np.testing.assert_array_equal(m.head.m2,baseline.head.m2)
        # Set no-modalitymethods' optimizerinactive only for numericalparity,
        # preserving their actualinputpath; no performance measurement.
        if method.startswith('no_'):m.learner.condition='zero'
        for _ in range(3):assert m.learn('What color?',image,'red')==baseline.learn('What color?',image,'red')
        assert m.core.dumps()==baseline.core.dumps() and m.head.dumps()==baseline.head.dumps()
        m.generate('What color?',image);baseline.generate('What color?',image)
        assert m.core.dumps()==baseline.core.dumps()


def test_cache_eligibility_partialword_resume_exact_and_endpoint_doesnot_replace_dynamics():
    m,image=make();m.training_trace=True;m.begin_prepare('What color?',image)
    for _ in range(23):m.prepare_step()
    imagecopy=DirectVisualWords.loads(m.dumps());assert imagecopy.dumps()==m.dumps()
    while m.prepare_cursor<33:m.prepare_step();imagecopy.prepare_step()
    assert m.dumps()==imagecopy.dumps()
    m.training_trace=False;initial=m.core.w_in.copy();m.learn('What color?',image,'red');assert np.any(initial!=m.core.w_in)
    m.training_trace=True;m.prepare('What color?',image);m.word_feature(1,0);dynamics=m.core.features.copy();trace=m.learner.e_f.copy();m.accumulate_word(m.ids['red'])
    np.testing.assert_array_equal(m.core.features,dynamics);np.testing.assert_array_equal(m.learner.e_f,trace)
    payload=m.dumps();copy=DirectVisualWords.loads(payload);assert copy.dumps()==payload and copy.core is copy.learner.core and copy.head is copy.learner.head
    for a in (m,copy):a.word_feature(a.ids['red'],1);a.accumulate_word(0);a.learner.finish();a.experiences+=1;a.training_trace=False
    assert m.dumps()==copy.dumps() and m.generate('What color?',image)==copy.generate('What color?',image) and m.dumps()==copy.dumps()
    assert m.learn('What color?',image,'green')==copy.learn('What color?',image,'green') and m.dumps()==copy.dumps()
    for loader in (VisualWords.loads,ReadoutEligibility.loads,UnifiedCore.loads):
        with pytest.raises(ValueError):loader(payload)


def test_targetfree_forward_and_wholecache_rollback_after_injected_failure():
    m,image=make();before=m.dumps()
    with pytest.raises(TypeError):m.generate('What color?',image,target='red')
    assert m.dumps()==before
    def fail(direction):raise MemoryError('Injected')
    m.learner._apply_direction=fail
    with pytest.raises(MemoryError):m.learn('What color?',image,'red')
    assert m.dumps()==before and m.core is m.learner.core and m.head is m.learner.head
    with zipfile.ZipFile(io.BytesIO(before)) as z:parts={n:z.read(n) for n in z.namelist()}
    meta=json.loads(parts['metadata.json']);meta['method']='frozen';parts['metadata.json']=json.dumps(meta).encode();bad=io.BytesIO()
    with zipfile.ZipFile(bad,'w') as z:
        for name,value in parts.items():z.writestr(name,value)
    with pytest.raises(ValueError,match='checksum'):DirectVisualWords.loads(bad.getvalue())


def test_pooled_cache_sensitivity_matches_independent_fixed_presynaptic_finite_difference():
    from scarx.evaluation.visual_eligibility_validation import check_pooled_sensitivities
    m,image=make();before=m.core.fingerprint();result=check_pooled_sensitivities(m,'What color?',image)
    assert result['synapses']==64 and m.core.fingerprint()==before and m.core.updates==0


def test_directvisual_control_frozenparity_and_optimizer_coldcontinuation():
    from scarx.evaluation.controls import DirectVisualLearnabilityControl,torch
    torch.set_num_threads(1);a,image=make('frozen');b=DirectVisualWords.loads(a.dumps());frozen=DirectVisualLearnabilityControl(b)
    assert a.learn('What color?',image,'red')==frozen.learn('What color?',image,'red') and a.dumps()==b.dumps()
    m,image=make();control=DirectVisualLearnabilityControl(m);before=m.core.w_in.copy();control.learn('What color?',image,'red')
    assert np.any(before!=m.core.w_in) and control.backward_passes==1
    copy=DirectVisualLearnabilityControl.loads(control.dumps());assert copy.dumps()==control.dumps()
    assert control.learn('What color?',image,'green')==copy.learn('What color?',image,'green') and control.dumps()==copy.dumps()
    assert control.max_graph_feature_error<=1e-6
