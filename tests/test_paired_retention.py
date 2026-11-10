import numpy as np
from scarx.evaluation.paired_retention import mixed_stream,component_intervention,retention_metrics,ridge_multiclass


def test_observable_context_stream_is_reproducible_unambiguous_without_future_reuse():
    x,y,g=mixed_stream(15217,per_window=16)
    xx,yy,gg=mixed_stream(15217,per_window=16)
    np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(y,yy);np.testing.assert_array_equal(g,gg)
    assert x.shape==(320,12,24)
    assert np.array_equal(x[:,:,4:].mean(1).argmax(1),g)
    for i in range(len(x)):
        cue=int(x[i,:2,:4].mean(0).argmax());expected=np.random.default_rng(700+int(g[i])).permutation(4)[cue]
        assert expected==y[i]


def test_component_interventions_and_joint_gate_cannot_hide_failed_acquisition():
    rng=np.random.default_rng(71);features=rng.normal(size=(64,4));labels=np.arange(64)%4;head=(np.eye(4),np.zeros(4))
    changed=(head[0][[1,0,2,3]],head[1]);r=component_intervention(features,features,labels,head,changed,np.zeros(4),np.ones(4))
    assert r['hidden_only_drop']==0 and abs(r['interaction_drop'])<1e-12
    m=retention_metrics(np.full((20,20),.25));assert m['mean_forgetting']==0 and not m['joint_gate']
    matrix=np.array([[.9,.2],[.8,.9]]);m=retention_metrics(matrix);assert m['worst_forgetting']==np.float64(.9)-np.float64(.8) and not m['joint_gate']
    f=np.tile(np.eye(4),(8,1));y=np.tile(np.arange(4),8)
    score,pred=ridge_multiclass(f,y,f,y);assert score==1 and np.array_equal(pred,y)
