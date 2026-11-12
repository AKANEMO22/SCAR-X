"""R9 manifest/summary audit helpers; no candidate training or selection."""
import hashlib,json
from collections import Counter,defaultdict


def effective_params(params):
    p=dict(params);p.pop('n_keys',None)
    if p.get('reset_mode')!='damped_subtractive':p.pop('damped_lambda',None)
    if p.get('surrogate_fn')=='fast_sigmoid':p.pop('surrogate_param',None)
    if p.get('gating_mode')=='none':p.pop('sensory_threshold',None);p.pop('sensory_floor',None)
    if not p.get('learned_attenuation'):p.pop('attenuation_factor',None)
    if not p.get('use_bf_cascade'):p.pop('kappa1',None);p.pop('kappa2',None)
    return p


def manifest_summary(manifest):
    ids=[r['candidate_id'] for r in manifest]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate candidate ID')
    groups=Counter(r['group'] for r in manifest);algorithms=defaultdict(list);evals=defaultdict(list)
    for row in manifest:
        payload=json.dumps({'group':row['candidate_id'].split('-')[0],'params':row['params']},sort_keys=True)
        if hashlib.sha256(payload.encode()).hexdigest()[:16]!=row['config_hash']:raise ValueError('Canonical manifest hash mismatch')
        signature=json.dumps(effective_params(row['params']),sort_keys=True);algorithms[signature].append(row['candidate_id'])
        group=row['group'];family='binding_pair' if group.startswith(('H1','H2')) else group
        evaluation=json.dumps([family,row['params'].get('n_keys'),effective_params(row['params'])],sort_keys=True)
        evals[evaluation].append(row['candidate_id'])
    # Count actual calls toscreen_* by branches in the pinned stage-B code.
    calls=sum(2*(2 if row['group'].startswith(('H1','H2')) else 1) for row in manifest)
    h4=Counter((r['params']['readout_mode'],r['params']['n_keys']) for r in manifest if r['group'].startswith('H4'))
    return {'declared_rows':len(manifest),'groups':dict(groups),'effective_model_signatures':len(algorithms),
        'effective_model_plus_evaluation_signatures':len(evals),
        'duplicate_effective_evaluation_groups':[v for v in evals.values() if len(v)>1],
        'h4_coverage':[{'readout':k[0],'n_keys':k[1],'configs':v} for k,v in sorted(h4.items())],
        'stage_b_actual_screen_function_calls':calls,
        'run_count_scope':'screen_*calls from source branch flow, not an asserted complete physical run ledger'}


def screening_checks(rows):
    deviations=[]
    for row in rows:
        m=row['metrics'];expected=m['final_mean']-2*m['mean_forgetting'] if row['group'].startswith('H5') else m['direct_acc']+.5*m['probe_gain']+.5*m['causal_revert_drop']
        if abs(expected-row['composite_score'])>1e-12:deviations.append(row['candidate_id'])
    return {'rows':len(rows),'composite_mismatches':deviations,'all_status_complete':all(r['status']=='COMPLETE' for r in rows),
        'recorded_runtime_sum_seconds':sum(r['runtime_sec'] for r in rows),
        'max_sampled_record_rss_mb':max(r['ram_rss_mb'] for r in rows)}
