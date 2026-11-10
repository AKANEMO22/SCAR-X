"""Read-only conditional signal diagnostics, no new rule or weight updates.

The optional Torch control is imported only while sampling in an isolated audit
process. Backward implementation stays exclusively in evaluation/controls.py.
"""
import json
import numpy as np


def cosine(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip(np.dot(a,b)/denominator,-1,1)) if denominator>0 else None


def split_statistics(signals):
    x=np.asarray(signals,np.float64)
    if x.ndim!=2 or len(x)<4 or len(x)%2 or not np.isfinite(x).all():raise ValueError('Finite even signal sample matrix required')
    first,second=np.split(x,2);a,b=first.mean(0),second.mean(0)
    noise=.5*(first.var(0,ddof=1).sum()+second.var(0,ddof=1).sum());cross=float(a@b)
    return {'cross_mean_energy':cross,'pair_noise_energy':float(noise),
        'cross_signal_to_pair_noise':cross/float(noise) if noise>0 else None,
        'replicate_cosine':cosine(a,b),'sample_mean_norm':float(np.linalg.norm(x.mean(0))),
        'sample_variance_energy':float(x.var(0,ddof=1).sum()),'pairs':len(x),'coordinates':x.shape[1]}


def sample(reference,prompt,prefix,target,rows,seed,pairs=128):
    """Simulate discarded tapes; preserve source weight/learning/RNG information."""
    from .controls import WordLearnabilityControl
    if type(pairs) is not int or not 4<=pairs<=128 or pairs%2:raise ValueError('Even pairs4..128')
    rows=np.asarray(rows,np.int64);core=reference.core
    if rows.shape!=(core.config.plastic_rows,) or len(set(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=core.config.hidden):raise ValueError('Plastic row selection')
    reference.mask(len(prefix));sequence=reference.sequence(prompt,prefix)
    if type(target) is not int or not core.output_mask[target]:raise ValueError('Masked diagnostic target')
    protected_names=[n for n in core.arrays() if n not in set(core.state_names)|{'voltage_pre'}]
    protected={n:core.arrays()[n].copy() for n in protected_names}
    counters=(core.updates,core.hidden_updates,core.reward_mean,reference.responses_seen,reference.evaluation_ticks)
    rngs=json.dumps([core.rng.bit_generator.state,core.mod_rng.bit_generator.state,reference.decoder_rng.bit_generator.state],sort_keys=True)
    control=WordLearnabilityControl(reference);rng=np.random.default_rng(seed)
    signals=[];eligibilities=[];rewards=[];surrogates=[];input_coordinates=rows.size*core.config.inputs
    for _ in range(pairs):
        u=rng.random((len(sequence),core.config.hidden));v=rng.random(u.shape)
        first=control._record_rollout(sequence,u,rows);e1=np.concatenate((core.elig_in.ravel(),core.elig_rec.ravel())).astype(np.float64)
        second=control._record_rollout(sequence,v,rows);e2=np.concatenate((core.elig_in.ravel(),core.elig_rec.ravel())).astype(np.float64)
        r=np.array([np.log(max(float(core.probabilities(t[0])[target]),1e-12)) for t in (first,second)],np.float64)
        signals.append(.5*((r[0]-core.reward_mean)*e1+(r[1]-core.reward_mean)*e2))
        eligibilities.append((e1,e2));rewards.append(r)
        # No optimizer/head update. The biased control is not gradientgroundtruth.
        surrogates.append(np.concatenate([g.ravel() for g in control._gradient(sequence,first,second,target)]).astype(np.float64))
    score=np.stack(signals);eligibilities=np.asarray(eligibilities);rewards=np.asarray(rewards);surrogate=np.stack(surrogates)
    half=pairs//2;baseline=np.r_[np.repeat(rewards[half:].mean(),half),np.repeat(rewards[:half].mean(),half)]
    crossfit=.5*np.sum((rewards-baseline[:,None])[:,:,None]*eligibilities,axis=1)
    methods={'existing_score':score,'crossfit_baseline_score':crossfit,'biased_surrogate':surrogate}
    summary={}
    for name,g in methods.items():
        summary[name]={'all':split_statistics(g),'input':split_statistics(g[:,:input_coordinates]),'recurrent':split_statistics(g[:,input_coordinates:])}
    compare={}
    for block,selection in [('all',slice(None)),('input',slice(None,input_coordinates)),('recurrent',slice(input_coordinates,None))]:
        existing=summary['existing_score'][block];cross=summary['crossfit_baseline_score'][block];sur=summary['biased_surrogate'][block]
        compare[block]={'crossfit_vs_existing_noise_ratio':cross['pair_noise_energy']/existing['pair_noise_energy'] if existing['pair_noise_energy']>0 else None,
            'surrogate_minus_existing_replicate_cosine':sur['replicate_cosine']-existing['replicate_cosine'] if sur['replicate_cosine'] is not None and existing['replicate_cosine'] is not None else None,
            'existing_mean_cosine_to_biased_surrogate':cosine(score[:,selection].mean(0),surrogate[:,selection].mean(0)),
            'crossfit_mean_cosine_to_biased_surrogate':cosine(crossfit[:,selection].mean(0),surrogate[:,selection].mean(0))}
    for name,value in protected.items():np.testing.assert_array_equal(value,core.arrays()[name])
    assert counters==(core.updates,core.hidden_updates,core.reward_mean,reference.responses_seen,reference.evaluation_ticks)
    assert rngs==json.dumps([core.rng.bit_generator.state,core.mod_rng.bit_generator.state,reference.decoder_rng.bit_generator.state],sort_keys=True)
    result={'summary':summary,'comparisons':compare,'reward_mean':float(rewards.mean()),'reward_std':float(rewards.std(ddof=1)),
        'stored_ema_baseline':core.reward_mean,'baseline_gap_to_sample_reward_mean':float(core.reward_mean-rewards.mean()),
        'weights_learningstate_rng_unchanged':True,'sampling_neuronstate_discarded':True,'pairs':pairs,'sequence_ticks':len(sequence),
        'numpy_forward_ticks':pairs*2*len(sequence),'extra_surrogate_forward_ticks':control.surrogate_forward_ticks,
        'backward_passes':control.backward_passes,'max_graph_state_error':control.max_graph_state_error,
        'max_graph_feature_error':control.max_graph_feature_error,'rows':rows.tolist(),
        'last_neural_sample':{'computed':48,'allocated':48,'tick':core.ticks,'spikes':core.spikes.astype(int).tolist(),
            'voltage':np.round(core.voltage,4).tolist(),'sample_kind':'read-only actualtrajectory; NO learning; surrogate control isolated'}}
    raw={'existing_score':score,'crossfit_score':crossfit,'biased_surrogate':surrogate,'rewards':rewards,
        'opposite_half_baseline':baseline}
    return result,raw
