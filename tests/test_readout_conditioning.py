import numpy as np
import pytest
from scarx.evaluation.readout_conditioning import ConditionedReadout


def test_calibration_is_train_only_constant_features_finite_and_inference_readonly():
    m=ConditionedReadout(np.zeros((2,4),np.float32),np.zeros(2,np.float32),'normalized')
    for _ in range(8):m.observe(np.ones(4,np.float32))
    m.freeze();saved=m.dumps()
    p=m.probabilities(np.ones((3,4),np.float32));assert np.isfinite(p).all();assert m.dumps()==saved
    with pytest.raises(ValueError):m.observe(np.ones(4))
    clone=ConditionedReadout.loads(saved);assert clone.dumps()==saved


def test_output_update_uses_only_presynaptic_feature_and_output_error():
    m=ConditionedReadout(np.zeros((2,2),np.float32),np.zeros(2,np.float32),'centered')
    m.observe(np.array([0.,0.]));m.observe(np.array([2.,4.]));m.freeze()
    x=np.array([2.,2.],np.float32);p=m.probabilities(x);error=-p;error[1]+=1
    expected=np.clip(.08*error[:,None]*m.transform(x)[None,:],-.025,.025)
    m.learn(x,1);np.testing.assert_array_equal(m.weights,expected)
    clone=ConditionedReadout.loads(m.dumps());m.learn(x,0);clone.learn(x,0);assert m.dumps()==clone.dumps()
