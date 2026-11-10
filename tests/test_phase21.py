import numpy as np
from scarx import Config
from scarx.neural.plasticity import PlasticCore
from scarx.learning.replay import ReplayBuffer
from scarx.evaluation.tasks21 import representation


def test_batch_forward_agrees_with_single():
    c=Config(inputs=8,outputs=2,hidden=48,plastic_rows=12)
    m=PlasticCore(c); x,_=representation('binding',5,1)
    m.forward(x[0],track=False)
    np.testing.assert_allclose(m.features_many(x,seed=c.seed+300000)[0],m.features,atol=1e-6)


def test_plasticity_mid_segment_checkpoint(tmp_path):
    c=Config(inputs=8,outputs=2,hidden=48,plastic_rows=12)
    a=PlasticCore(c,'score_meta',shuffle=True); x,y=representation('xor',55,1)
    for item in x[0,:5]: a.step(item)
    a.save(tmp_path/'m.npz'); b=PlasticCore.load(tmp_path/'m.npz')
    for item in x[0,5:]:
        a.step(item); b.step(item)
    a.learn(int(y[0])); b.learn(int(y[0]))
    for k,v in a.arrays().items(): np.testing.assert_array_equal(v,b.arrays()[k])
    assert a.rng.bit_generator.state==b.rng.bit_generator.state


def test_readout_does_not_update_hidden_and_eval_is_pure():
    m=PlasticCore(Config(inputs=8,outputs=2),'readout'); x,y=representation('order',23,4)
    wi=m.w_in.copy(); before=m.rng.bit_generator.state
    m.features_many(x); assert m.rng.bit_generator.state==before
    for seq,label in zip(x,y): m.forward(seq,track=True); m.learn(int(label))
    np.testing.assert_array_equal(wi,m.w_in)


def test_replay_capacity_and_resume(tmp_path):
    b=ReplayBuffer(8,12,8,'balanced',55)
    x,y=representation('xor',11,20)
    for i,(seq,label) in enumerate(zip(x,y)): b.add(seq,int(label),i%3)
    assert b.size==8
    b.save(tmp_path/'replay.npz'); c=ReplayBuffer.load(tmp_path/'replay.npz')
    for a,z in zip(b.sample(10),c.sample(10)):
        np.testing.assert_array_equal(a[0],z[0]); assert a[1]==z[1]


def test_order_has_balanced_labels_and_identical_signal_bag():
    x,y=representation('order',44,256)
    assert y.sum()==128
    # Noise aside, each example has exactly one large A and one large B event.
    assert np.all((x[:,:,0]>1).sum(1)==1)
    assert np.all((x[:,:,1]>1).sum(1)==1)
    assert np.all(x[:,-1,7]>.5)
