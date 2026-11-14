import numpy as np
import pytest


def test_isolated_hybrid_direction_matches_forward_without_mutating_weights():
    pytest.importorskip("torch")
    from dataclasses import asdict
    from scarx.hybrid.learner import LocalALIF, features
    from scarx.evaluation.controls import hybrid_tape_credit_direction
    model = LocalALIF(seed=260113)
    x = features("Owned local credit diagnostic", (.25,-.5,1))
    model.forward(x, head="relevance", track=True)
    expected_p = model.forward(x, head="relevance")[0]
    model.core.reset_state()
    spikes=[]
    for row in x:
        model.step(row,track=False)
        spikes.append(model.core.spikes.copy())
    state={name:getattr(model.core,name).copy() for name in ("w_in","w_rec","w_out","bias","indices")}
    state.update(config=asdict(model.core.config),f_mean=model.f_mean.copy(),f_var=model.f_var.copy())
    original={key:value.copy() for key,value in state.items() if isinstance(value,np.ndarray)}
    final=[getattr(model.core,name).copy() for name in ("voltage","adaptation","spikes","features")]
    result=hybrid_tape_credit_direction(state,x,np.stack(spikes),final,1,device="cpu")
    np.testing.assert_allclose(result["probabilities"],expected_p,atol=1e-6)
    assert result["hidden_direction"].shape==(128,72)
    assert result["input_direction"].shape==x.shape
    assert result["optimizer_steps"]==0 and np.linalg.norm(result["hidden_direction"])>0
    for key,value in original.items():np.testing.assert_array_equal(state[key],value)
