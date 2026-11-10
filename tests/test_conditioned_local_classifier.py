import io,json
import numpy as np
import pytest
from scarx.neural.core import Config,SpikingCore
from scarx.neural.local_classifier import softmax
from scarx.neural.local_classifier_conditioned import ConditionedLocalClassifierCore


def test_frozen_conditioning_local_derivative_and_unchanged_forward():
    c=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=12101)
    m=ConditionedLocalClassifierCore(c);base=SpikingCore(c)
    with pytest.raises(ValueError):m.learn_auxiliary(1)
    m.condition(np.full(48,.15),np.full(48,.2))
    for x in np.random.default_rng(91).normal(0,.3,(12,8)).astype(np.float32):
        m.step(x,True);base.step(x,False)
        for name in ('features','voltage','adaptation','spikes'):np.testing.assert_array_equal(getattr(m,name),getattr(base,name))
    signal,_=m.auxiliary_signal(1);row=0;j=int(m.rows[row]);i=int(np.argmax(np.abs(m.elig_in[row])));elig=float(m.elig_in[row,i])
    f=m.features.astype(np.float64);g=m.fixed_classifier.astype(np.float64)
    def proxy(delta):
        f2=f.copy();f2[j]+=delta*elig
        p=softmax(g@np.clip((f2-m.classifier_mean)/m.classifier_scale,-5,5));return -np.log(p[1])
    eps=1e-4;finite=(proxy(eps)-proxy(-eps))/(2*eps)
    assert -finite==pytest.approx(float(signal[j])*elig,abs=1e-6,rel=1e-4)
    clone=ConditionedLocalClassifierCore.loads(m.dumps());clone.w_out*=13;clone.bias+=4
    before=m.classifier_mean.copy();m.learn_auxiliary(1);clone.learn_auxiliary(1)
    np.testing.assert_array_equal(m.w_in,clone.w_in);np.testing.assert_array_equal(m.w_rec,clone.w_rec);np.testing.assert_array_equal(before,m.classifier_mean)
    with pytest.raises(ValueError):m.condition(before,np.ones(48))


def test_conditioned_partial_synapse_state_rng_continuation_and_checksum():
    m=ConditionedLocalClassifierCore(Config(hidden=48,plastic_rows=12,outputs=2));m.condition(np.zeros(48),np.ones(48))
    x=np.random.default_rng(92).normal(0,.3,(19,8)).astype(np.float32)
    for tick in x[:5]:m.step(tick,True)
    m.learn_auxiliary(1,True);clone=ConditionedLocalClassifierCore.loads(m.dumps())
    for tick in x[5:]:
        m.step(tick,True);clone.step(tick,True);m.learn_auxiliary(0,True);clone.learn_auxiliary(0,True)
    assert m.dumps()==clone.dumps()
    with np.load(io.BytesIO(m.dumps()),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    arrays['classifier_scale'][0]+=.01;stream=io.BytesIO();np.savez(stream,**arrays)
    with pytest.raises(ValueError,match='Corrupt'):ConditionedLocalClassifierCore.loads(stream.getvalue())
