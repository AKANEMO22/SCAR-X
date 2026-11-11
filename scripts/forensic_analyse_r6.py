"""Recompute audit metrics from saved predictions and immutable historical rows."""
from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.forensic.reference_evaluator import pair_metrics,seed_bootstrap,integrate_gates

DOC=ROOT/'docs/research/forensic-r6-20261111'
PROJECT=ROOT.parents[2]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(name,value):(DOC/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def gitjson(sha,path):return json.loads(subprocess.check_output(['git','-C',str(PROJECT),'show',sha+':'+path]))
def receipt(phase):
    r=read(ROOT/'artifacts/forensic-r6'/f'{phase}-receipt.json')
    if r['error']:raise ValueError('Failed replay '+phase)
    run=ROOT/r['run'];return run,read(run/'replay/original-output/result.json'),read(run/'replay/audit-instrumentation.json')


def audit_saved_predictions(run,instrument):
    rows=[]
    for entry in instrument['observations']:
        p=run/'replay'/entry['arrays']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Saved prediction artifact changed')
        with np.load(p,allow_pickle=False) as z:
            if 'pair_id' in z:
                pred=z['pair_prediction'] if 'pair_prediction' in z else z['prediction']
                labels=z['pair_label'] if 'pair_label' in z else z['label']
                independent=pair_metrics(pred,labels,z['vA'],z['vB'],z['q'],z['pair_id'],z['flips'])
                old=entry.get('source_metrics',{})
                for metric in ('both_A','both_B','both_all'):
                    if metric in old and old[metric]!=independent[metric]:raise ValueError('Independent pair metric mismatch: '+entry['id']+' '+metric)
                rows.append({'id':entry['id'],'seed':entry.get('seed'),'metrics':independent,'source_metric_match':True})
            elif 'prediction' in z and 'label' in z:
                direct=float(np.mean(z['prediction']==z['label']))
                if 'direct_independent' in entry and direct!=entry['direct_independent']:raise ValueError('Saved direct mismatch')
                rows.append({'id':entry['id'],'seed':entry.get('seed'),'direct':direct})
    return rows


def main():
    mr,matrix,mi=receipt('r6d-matrix');cr,controls,ci=receipt('r6d-controls');gr,grok,gi=receipt('r6b')
    mh=gitjson('4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893','runs/20261111T080352Z-r6d-confirm-v1-ff539c/result.json')
    ch=gitjson('4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893','runs/20261111T080601Z-r6d-controls-confirm-v1-b53925/result.json')
    gh=gitjson('c39e9db7771582d9933af402a209917328b71119','runs/grok-r6-causal-v1/result.json')
    exact={'r6d_matrix_trials':matrix['trials']==mh['trials'],'r6d_matrix_summary':matrix['summary']==mh['summary'],
           'r6d_controls_trials':controls['trials']==ch['trials'],'r6d_controls_summary':controls['summary']==ch['summary'],
           'r6b_fields':{k:grok[k]==gh[k] for k in gh if k not in ('seconds','peak_rss_bytes_sampled','peak_wset_bytes')}}
    checks={'matrix':audit_saved_predictions(mr,mi),'controls':audit_saved_predictions(cr,ci),'grok':audit_saved_predictions(gr,gi)}
    seeds=sorted({r['seed'] for r in mh['trials']})
    by={(r['seed'],r['arm'],r['conv'],r['n_unique']):r for r in mh['trials']}
    def vector(arm,conv,n,metric='both_A'):return {s:by[(s,arm,conv,n)][metric] for s in seeds}
    def delta(left,right):return seed_bootstrap({s:left[s]-right[s] for s in seeds})
    one=vector('grid','r5b',1024);large=vector('grid','r5b',16384)
    repeat=vector('matched','r5b',256);unique=vector('matched','r5b',16384)
    small6=vector('grid','r6a',1024);large6=vector('grid','r6a',16384)
    stress={kind:vector('stress-'+kind,'r5b',16384) for kind in ('temporal','distractor')}
    effects={'data_under_r5b':delta(large,one),'data_under_r6a':delta(large6,small6),
             'preproc_at_1024':delta(small6,one),'preproc_at_16384':delta(large6,large),
             'matched_unique_minus_repeated_r5b':delta(unique,repeat),
             'interaction_full_precision':seed_bootstrap({s:(large6[s]-small6[s])-(large[s]-one[s]) for s in seeds})}
    per_seed=[{'seed':s,'grid_1024_r5b_both_A':one[s],'grid_16384_r5b_both_A':large[s],
        'matched_repeat_256x64_r5b_both_A':repeat[s],'matched_unique_16384x1_r5b_both_A':unique[s],
        'stress_temporal_both_A':stress['temporal'][s],'stress_distractor_both_A':stress['distractor'][s],
        'grid_gain':large[s]-one[s],'matched_gain':unique[s]-repeat[s]} for s in seeds]
    supplied=read(DOC/'inputs/R6D_RECALCULATED_PER_SEED.json')
    supplied_match=per_seed==supplied['per_seed']
    rand={};first={};query={}
    for obs in ci['observations']:
        if not obs['id'].endswith('held-prediction'):continue
        with np.load(cr/'replay'/obs['arrays'],allow_pickle=False) as z:
            s=obs['seed'];first[s]=float(np.mean(z['vA']==z['label']));query[s]=float(np.mean(z['q']==z['label']))
            if not obs['training_labels_are_original']:rand[s]=float(np.mean(z['prediction']==z['label']))
    if set(rand)!=set(seeds):raise ValueError('Missing independently scored random-label control predictions')
    source_gates=dict(mh['summary']['gates'])
    current_gates=dict(source_gates)
    # Supplemental negative controls are labeled as such. Do not make a same-head
    # claim when their update-order digest differs from the matrix/stress heads.
    current_gates['G4_random_labels_chance']=float(np.mean(list(rand.values())))<=.60
    current_gates['G5_shortcuts_ok']=abs(float(np.mean(list(first.values())))-.75)<=.05 and abs(float(np.mean(list(query.values())))-.5)<=.05
    joins=[]
    for s in seeds:
        m=[o for o in mi['observations'] if o['id'].endswith('r6d-evaluate') and o['seed']==s and o['n_unique']==16384 and o['source_metrics']['head_digest']==by[(s,'grid','r5b',16384)]['head_digest']][0]
        c=[o for o in ci['observations'] if o['id'].endswith('r6d-evaluate') and o['seed']==s][0]
        keys=('train_features','training_labels','head_initial','order','head_final')
        joins.append({'seed':s,'comparison':{k:m['head_identity'][k]==c['head_identity'][k] for k in keys},
            'matrix_head':m['head_identity'],'controls_head':c['head_identity'],
            'status':'NON_EQUIVALENT_TRAINED_HEAD_DIFFERENT_ORDER'})
    violations=[x for x in mi['stress_applications'] if x.get('violation')]
    integration={'original_matrix_gates':source_gates,'supplemented_controls_gates':current_gates,
        'integration':integrate_gates(list(source_gates),current_gates),'control_join_audit':joins,
        'G8_status':'NOT_ESTABLISHED_BY_ORIGINAL_RESULT_OR_ITS_TESTS; independent small fixture found nonzero voltage rounding',
        'historical_mean_gate':{'clean':float(np.mean(list(large.values())))>=.75,'temporal':float(np.mean(list(stress['temporal'].values())))>=.75,'distractor':float(np.mean(list(stress['distractor'].values())))>=.75},
        'sensitivity_only_all_seeds_gate':{'clean':all(v>=.75 for v in large.values()),'temporal':all(v>=.75 for v in stress['temporal'].values()),'distractor':all(v>=.75 for v in stress['distractor'].values())},
        'counterfactual_stress_violations':violations,'all_actual_stress_row_oracles_checked':all(x.get('all_actual_held_oracle_labels_equal',True) for x in mi['stress_applications']),
        'harness_verdict':'INVALID_COUNTERFACTUAL_STRESS_PAIRING; ORIGINAL_AGGREGATION_INCOMPLETE',
        'scientific_verdict':'POSITIVE_FROZEN_READOUT_DATA_EFFECT; FAIL_ON_RECORDED_G6; NO_LEARNED_HIDDEN_CLAIM',
        'new_confirmation':False,'production_ready':False}
    save('R6D_GATE_INTEGRATION.json',integration)
    save('R6D_CLAIM_RECOMPUTE.json',{'source_sha':'4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893','per_seed':per_seed,'effects':effects,
        'attached_json_per_seed_matches_raw':supplied_match,'random_label_direct':seed_bootstrap(rand),
        'first_key_direct':seed_bootstrap(first),'query_only_direct':seed_bootstrap(query),
        'exact_historical_reproduction':exact,'raw_prediction_checks':checks,'evidence_runs':{'matrix':mr.relative_to(ROOT).as_posix(),'controls':cr.relative_to(ROOT).as_posix(),'grok':gr.relative_to(ROOT).as_posix()}})
    repro=grok['reproduction'];gseed={r['seed']:r['both_A'] for r in repro};ridge={r['seed']:r['ridge']['both_A'] for r in repro}
    pairs=[o for o in gi['observations'] if o['id'].endswith('r6b-head')]
    arrays_ok=True
    for s,val in gseed.items():
        matches=[o for o in pairs if o['seed']==s and o['projection']==1234 and o['class_name']=='Head']
        if not matches or matches[0]['independent_pair_metrics']['both_A']!=val:arrays_ok=False
    save('R5B_R6B_RECOMPUTE.json',{'head_both_A':seed_bootstrap(gseed),'ridge_both_A':seed_bootstrap(ridge),
        'source_reproduction_fields_exact':exact['r6b_fields'],'raw_prediction_reproduction_matches':arrays_ok,
        'screen_gain':grok.get('gain'),'permuted_gain':grok.get('perm_gain'),'protected_order_drop':grok.get('order_drop'),
        'verdict':'FAIL_CANDIDATE_SCREEN','fresh_confirmation':False,'source_shadowing':read(gr/'replay/effective_bindings.json'),
        'registered_but_missing_runner_conditions':['reversed_writes','independent_shift_stress','distractor_stress','noise_stress'],
        'hidden_learning_tested':False})
    print(json.dumps({'reproduction':exact,'attached_json_matches':supplied_match,'matrix_prediction_checks':len(checks['matrix']),
        'grok_prediction_checks':len(checks['grok']),'integration_verdict':integration['integration']['verdict'],'stress_pair_violations':len(violations)}))


if __name__=='__main__':main()
