import importlib.util,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from scarx.evaluation.credit_noise import split_statistics


def test_split_reproducibility_keeps_negative_energy_and_missing_zero_noise():
    zero=split_statistics(np.zeros((8,3)));assert zero['cross_signal_to_pair_noise'] is None and zero['replicate_cosine'] is None
    constant=split_statistics(np.ones((8,3)));assert constant['replicate_cosine']==1 and constant['cross_signal_to_pair_noise'] is None
    opposing=split_statistics(np.r_[np.ones((4,3)),-np.ones((4,3))]);assert opposing['cross_mean_energy']==-3 and opposing['replicate_cosine']==-1
    noisy=split_statistics(np.arange(24).reshape(8,3));assert noisy['pair_noise_energy']>0
    with pytest.raises(ValueError):split_statistics(np.ones((3,3)))


def _exercise():
    from scarx.evaluation.credit_noise import sample
    from scarx.evaluation.controls import WordLearnabilityControl,torch
    from scarx.evaluation.score_word import ScoreWord
    torch.set_num_threads(1)
    model=ScoreWord(['<eos>','<bos>','hi','bye','.'],28001,'frozen');model.calibrate(['hi','bye'])
    source=model.dumps();seed=9120000;rows=np.arange(12);expected=ScoreWord.loads(source);expected.mask(0);seq=expected.sequence('hi',[])
    # Direct existingformula on the exact firsttwo externaluniformrollouts.
    rng=np.random.default_rng(seed);u=rng.random((len(seq),48));v=rng.random(u.shape)
    a=expected.core.rollout(seq,u,rows);b=expected.core.rollout(seq,v,rows);r=[np.log(max(float(expected.core.probabilities(q[0])[2]),1e-12)) for q in (a,b)]
    ea=np.r_[a[1].ravel(),a[2].ravel()];eb=np.r_[b[1].ravel(),b[2].ravel()]
    direct=.5*((r[0]-expected.core.reward_mean)*ea+(r[1]-expected.core.reward_mean)*eb)
    result,raw=sample(model,'hi',[],2,rows,seed,pairs=8)
    np.testing.assert_allclose(raw['existing_score'][0],direct,atol=1e-5,rtol=1e-6)
    assert result['weights_learningstate_rng_unchanged'] and result['backward_passes']==8 and result['numpy_forward_ticks']==8*2*15
    assert result['max_graph_feature_error']==0
    for key in ('w_in','w_rec','w_out','row_updates','head_mean','moment_in'):np.testing.assert_array_equal(getattr(model.core,key),getattr(ScoreWord.loads(source).core,key))
    assert not model.core.hidden_updates and not model.core.updates


def test_isolated_actual_pair_formula_and_no_parameter_rng_update():
    if importlib.util.find_spec('torch') is None:pytest.skip('Optional control dependency')
    child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--isolated'],capture_output=True,text=True,timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert child.returncode==0,child.stdout+child.stderr


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'));_exercise()
