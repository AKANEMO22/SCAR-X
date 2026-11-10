import numpy as np
from scarx.neural.core import SpikingCore,Config
from scarx.neural.local_classifier import LocalClassifierCore,softmax


def test_local_objective_tangent_matches_aux_signal_and_is_independent_of_task_head():
    c=Config(inputs=8,outputs=2,hidden=48,plastic_rows=12,seed=11113,anchor_strength=0)
    m=LocalClassifierCore(c);base=SpikingCore(c)
    for x in np.random.default_rng(1).normal(0,.4,(12,8)).astype(np.float32):
        m.step(x,True);base.step(x,False)
        np.testing.assert_array_equal(m.features,base.features)
    signal,_=m.auxiliary_signal(1);j=int(m.rows[0]);i=2;elig=float(m.elig_in[0,i]);f=m.features.astype(np.float64);g=m.fixed_classifier.astype(np.float64)
    def local_proxy(delta):
        f2=f.copy();f2[j]+=delta*elig
        return -np.log(softmax(g@f2)[1])
    eps=1e-4;finite=(local_proxy(eps)-local_proxy(-eps))/(2*eps)
    np.testing.assert_allclose(-finite,float(signal[j])*elig,atol=1e-6,rtol=1e-4)
    clone=LocalClassifierCore.loads(m.dumps());clone.w_out*=100;clone.bias+=10
    fixed=m.fixed_classifier.copy();m.learn_auxiliary(1);clone.learn_auxiliary(1)
    np.testing.assert_array_equal(m.w_in,clone.w_in);np.testing.assert_array_equal(m.w_rec,clone.w_rec);np.testing.assert_array_equal(m.fixed_classifier,fixed)


def test_auxiliary_rng_state_and_partial_eligibility_resume_exactly():
    m=LocalClassifierCore(Config(inputs=8,outputs=2,hidden=48,plastic_rows=12,seed=11117,anchor_strength=0))
    sequence=np.random.default_rng(2).normal(0,.4,(17,8)).astype(np.float32)
    for x in sequence[:5]:m.step(x,True)
    m.learn_auxiliary(1,True);clone=LocalClassifierCore.loads(m.dumps())
    for x in sequence[5:]:
        m.step(x,True);clone.step(x,True);m.learn_auxiliary(0,True);clone.learn_auxiliary(0,True)
    assert m.dumps()==clone.dumps()
