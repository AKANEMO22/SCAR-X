from dataclasses import replace
import numpy as np
import pytest
from scarx.neural.core import SpikingCore,Config
from scarx.neural.local_target import LocalTargetCore


def test_forward_matches_original_and_hidden_credit_does_not_depend_on_readout():
    c=Config(inputs=8,outputs=2,hidden=48,plastic_rows=12,seed=1201,anchor_strength=0)
    a=SpikingCore(c);b=LocalTargetCore(c);d=LocalTargetCore(c)
    d.w_out[:]=np.random.default_rng(2).normal(0,2,d.w_out.shape)
    for x in np.random.default_rng(1).normal(0,.4,(20,8)).astype(np.float32):
        a.step(x);b.step(x);d.step(x)
        np.testing.assert_array_equal(a.spikes,b.spikes);np.testing.assert_array_equal(a.features,b.features)
    b.learn(1);d.learn(1)
    np.testing.assert_array_equal(b.w_in,d.w_in);np.testing.assert_array_equal(b.w_rec,d.w_rec)
    batch=np.random.default_rng(3).normal(0,.3,(4,12,8)).astype(np.float32)
    features=b.features_many(batch)
    for seq,f in zip(batch,features):b.forward(seq);np.testing.assert_allclose(b.features,f,atol=1e-6)


def test_complete_checkpoint_preserves_forward_eligibility_and_shuffled_continuation():
    m=LocalTargetCore(Config(inputs=8,outputs=2,hidden=48,plastic_rows=12,seed=1213,anchor_strength=0))
    rows=np.random.default_rng(4).normal(0,.4,(6,12,8)).astype(np.float32)
    m.forward(rows[0],track=True);m.learn(1,'shuffled');m.step(rows[1,0],track=True)
    clone=LocalTargetCore.loads(m.dumps())
    for x in rows[1,1:]:m.step(x,True);clone.step(x,True)
    for seq in rows[2:]:
        m.learn(0,'shuffled');clone.learn(0,'shuffled');m.forward(seq,True);clone.forward(seq,True)
    for name,a in m.arrays().items():np.testing.assert_array_equal(a,clone.arrays()[name])
    assert m.dumps()==clone.dumps()
    with pytest.raises(ValueError):SpikingCore.loads(m.dumps())


def test_feedback_modulation_uses_fixed_random_code_not_output_weight_transpose():
    m=LocalTargetCore(Config(inputs=8,outputs=2,hidden=48,plastic_rows=12,seed=2203,anchor_strength=0))
    m.forward(np.random.default_rng(5).normal(0,.4,(12,8)).astype(np.float32),True)
    code=m.target_rates.copy();before=m.w_in.copy();rows=m.rows.copy()
    mod=code[rows,1]-code[rows]@m.probabilities()
    expected=before.copy();expected[rows]+=np.clip(m.config.hidden_lr*mod[:,None]*m.elig_in,-m.config.update_clip,m.config.update_clip)
    np.clip(expected,-m.config.weight_clip,m.config.weight_clip,out=expected)
    m.learn(1,'feedback');np.testing.assert_array_equal(m.target_rates,code);np.testing.assert_array_equal(m.w_in,expected)
    clone=LocalTargetCore.loads(m.dumps())
    for _ in range(3):m.learn(0,'feedback_shuffled');clone.learn(0,'feedback_shuffled')
    assert m.dumps()==clone.dumps()
