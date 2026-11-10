import numpy as np
from scarx import Config
from scarx.learning.lifelong import LifelongLearner
from scarx.evaluation.tasks21 import representation
from scarx.evaluation.lifelong21 import continuous_stream


def test_si_is_readout_local_and_resumes(tmp_path):
    m=LifelongLearner(Config(inputs=8,outputs=2),'si_readout')
    wi=m.model.w_in.copy(); x,y=representation('binding',61,68)
    for seq,label in zip(x[:65],y[:65]): m.update(seq,int(label))
    assert np.any(m.si['omega_out']>0)
    m.save(tmp_path); restored=LifelongLearner.load(tmp_path)
    for seq,label in zip(x[65:],y[65:]): m.update(seq,int(label)); restored.update(seq,int(label))
    np.testing.assert_array_equal(wi,m.model.w_in)
    for k,v in m.model.arrays().items(): np.testing.assert_array_equal(v,restored.model.arrays()[k])
    for k,v in m.si.items(): np.testing.assert_array_equal(v,restored.si[k])


def test_stream_reliability_is_not_perfect_oracle():
    rows=list(continuous_stream(56))
    assert all(x.shape==(12,8) for x,*_ in rows)
    assert any(label!=clean and reliability>.7 for _,label,reliability,clean,_ in rows)
    assert any(label==clean and reliability<.7 for _,label,reliability,clean,_ in rows)
