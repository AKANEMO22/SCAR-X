import json
import numpy as np
import pytest
from scarx import Config, SpikingCore
from scarx.evaluation.tasks import episodes
from scarx.learning.controller import LearningController


def test_corrupt_checkpoint_is_rejected(tmp_path):
    path=tmp_path/"m.npz"
    SpikingCore().save(path)
    with np.load(path,allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    arrays["w_in"][0,0] += 1
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match="Checksum"): SpikingCore.load(path)


def test_bad_reward_does_not_partially_update():
    m=SpikingCore(Config(inputs=5))
    x,y=next(episodes(1,1))
    m.forward(x,track=True)
    fingerprint=m.fingerprint()
    with pytest.raises(ValueError): m.learn(y,reward=float("nan"))
    assert m.fingerprint()==fingerprint and m.updates==0


def test_unbounded_batch_rejected_before_snapshot(tmp_path):
    m=SpikingCore(Config(inputs=5))
    c=LearningController(m,tmp_path)
    x,y=next(episodes(1,1))
    def infinite():
        while True: yield x,y
    with pytest.raises(ValueError,match="capacity"):
        c.learn(infinite(),guard=[(x,y,"dev")])
    assert not list(tmp_path.iterdir())
