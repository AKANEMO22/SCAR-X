"""Arithmetic/provenance review helpers, never a learner or model evaluator."""
import itertools
import numpy as np


def paired_distractor(diagnostics):
    if not isinstance(diagnostics,dict) or len(diagnostics)!=3:raise ValueError('Three historical DEV seeds required')
    rows=[]
    for seed,row in sorted(diagnostics.items()):
        base=row['std_ew_he'];candidate=row['delay_eligibility_suppressed']
        values=[base[m] for m in ('direct','probe')]+[candidate[m] for m in ('direct','probe')]
        if not all(type(v) in (float,int) and np.isfinite(v) and 0<=v<=1 for v in values):raise ValueError('Accuracy bounds')
        rows.append({'seed':int(seed),'standard_direct':base['direct'],'standard_probe':base['probe'],
                     'suppressed_direct':candidate['direct'],'suppressed_probe':candidate['probe'],
                     'probe_effect_pp':100*(candidate['probe']-base['probe']),
                     'direct_effect_pp':100*(candidate['direct']-base['direct'])})
    effects=np.array([r['probe_effect_pp'] for r in rows])
    draws=np.array([np.mean(draw) for draw in itertools.product(effects,repeat=len(effects))])
    return {'per_seed':rows,'means':{key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='seed'},
            'paired_probe_interval_pp':{'lower95':float(np.quantile(draws,.025,method='inverted_cdf')),
                'upper95':float(np.quantile(draws,.975,method='inverted_cdf')),'independent_seed_clusters':len(rows),
                'method':'All27 equally likely 3-seed bootstrap resamples, inverse empirical CDF; exploratory DEV only'},
            'seeds_with_negative_probe_effect':int((effects<0).sum()),'training_runs_opened':0}


def formula_counterexamples():
    gamma=.25;prior=1.;gate=0.;new_eligibility=gamma*prior+(1-gamma)*gate*10.
    k1,k2=.005,.001;u1,u2=1.,0.;gap=u1-u2
    simultaneous=(u1-k1*gap,u2+k2*gap)
    first=u1-k1*gap;sequential=(first,u2+k2*(first-u2))
    return {'zero_gate':{'prior':prior,'gamma':gamma,'result':new_eligibility,'literal_freeze':prior,
                        'freezes_old_trace':new_eligibility==prior},
            'amplitude_gate_fixture':{'epsilon':1e-6,'noise_l1':8*.02,'distractor_l1':1.2,
                        'noise_opens_gate':8*.02>1e-6,'distractor_opens_gate':1.2>1e-6,
                        'scope':'Illustrative counterexample at a tiny threshold, not a calibrated event detector or task result'},
            'time_gate_overlap':{'id_second_valid_event_possible_ticks':[4,5],
                'suppression_range_inclusive':[3,8],
                'suppressed_valid_event_ticks':[t for t in (4,5) if 3<=t<=8],
                'scope':'Read from pinned generator/diagnostic source; not an oracle available to a new learner'},
            'diffusion_assignment':{'k1':k1,'k2':k2,'old_gap_simultaneous':simultaneous,
                'sequential_new_gap':sequential,'same_update':simultaneous==sequential,
                'simultaneous_weighted_mass':simultaneous[0]+5*simultaneous[1],
                'sequential_weighted_mass':sequential[0]+5*sequential[1]},
            'array_bytes_only':{'gamma_48_float32':48*4,'one_hidden_slow_state_H48_D8_K8':48*(8+8)*4,
                'one_hidden_slow_state_public_G1_D222':48*(222+8)*4,
                'one_hidden_slow_state_codec_V55_D253':48*(253+8)*4,
                'scope':'One auxiliary array only; excludes traces, readout, optimizer, replay, snapshots and runtime'}}
