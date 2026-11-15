"""Read P1A arrays/journals/owned state. No model import, tool replay or forward."""
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
import sqlite3
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ACTIONS=('show_schema','inspect_status','retry','clarify','defer','finish')
DYNAMIC={'core.voltage','core.adaptation','core.spikes','core.pre','core.post','core.features'}


def serial(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(serial(value)).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def encode_text(text,numeric=()):
    words=re.findall(r'[^\W_]+',text.lower(),re.UNICODE)[:128];vector=np.zeros(64,np.float32)
    for term in words+[a+' '+b for a,b in zip(words,words[1:])]:
        value=hashlib.blake2b(term.encode(),digest_size=8).digest()
        vector[int.from_bytes(value[:4],'little')%48]+=1. if value[4]&1 else -1.
    norm=np.linalg.norm(vector[:48])
    if norm:vector[:48]*=2./norm
    for index,value in enumerate(numeric):vector[48+index]=np.clip(float(value),-1,1)
    return np.repeat(vector[None,:],8,axis=0)


def reencode(decision):
    task=decision['public_task'];blocks=[encode_text(serial(task['request']).decode()+' '+task['instruction'])]
    last=0.
    for row in decision['observations']:
        assert last<=row['returned_epoch']<=decision['cutoff'];last=row['returned_epoch']
        p=row['probe'] or {}
        numbers=(row['reported_status']=='ok',row['reported_status']=='error',row['empty_result'],row['spillover'],
            min(row['streak']/3,1),row['probe'] is not None,p.get('read_ok',False),
            p.get('reported_not_found',False),p.get('original_path_differs',False),p.get('kind')=='requested_path_binding')
        blocks.append(encode_text('Observed event [observed tool] '+row['tool']+' '+row['error_type'],numbers))
    return np.concatenate(blocks).astype(np.float32)


def model_object(run,sha):
    path=run/'model-objects'/(sha+'.npz');raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==sha
    with np.load(io.BytesIO(raw),allow_pickle=False) as data:
        meta=json.loads(data['metadata'].tobytes());arrays={n:data[n] for n in data.files if n!='metadata'}
    assert meta['mode']=='frozen' and meta['updates']==meta['hidden_updates']==meta['core_updates']==0 and meta['replay']==[]
    assert arrays['core.w_in'].shape==(128,64) and arrays['core.w_out'].shape==(8,128)
    assert all(np.isfinite(v).all() for v in arrays.values())
    return arrays,meta


def audit(run,diagnostic_stopped=False):
    run=run.resolve()
    if run.parent!=ROOT/'runs' or '-neural-outcome-bridge-p1a-v1-' not in run.name:raise ValueError('Not an owned P1A run')
    manifest=read(run/'manifest.json');result=read(run/'result.json')
    if diagnostic_stopped:
        assert result['gate_pass'] is False and result['verdict']=='P1A_STOPPED_NOT_PASS'
        assert result['failures']==['ValueError: Live/model-call/learning invariant failure']
    else:assert result['gate_pass']
    assert len(result['seed_results'])==2
    for name,sha in manifest['source_sha256'].items():
        assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==sha
    ledger=[json.loads(line) for line in (run/'attempt-ledger.jsonl').read_bytes().splitlines()]
    previous='0'*64;counts={'forward':0,'dispatch':0};by_attempt={}
    for index,row in enumerate(ledger,1):
        data={k:v for k,v in row.items() if k!='sha256'}
        assert data['previous']==previous and data['attempt']==index
        assert hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()==row['sha256']
        assert manifest['started_epoch']<=row['epoch']<=manifest['started_epoch']+600
        counts[row['kind']]+=1;by_attempt[index]=row;previous=row['sha256']
    assert counts=={'forward':18,'dispatch':10} and len(ledger)==28
    decisions={p.stem:read(p) for p in (run/'decisions').glob('*.json')}
    assert len(decisions)==18
    max_probability_error=0.;per_seed={};ordinary=[]
    for identifier,decision in decisions.items():
        assert identifier==decision['id'] and decision['selector']=='neural_policy' and decision['device']=='gpu:0'
        assert len(decision['allowed'])>=2 and decision['mode']=='frozen' and decision['learning_authorized'] is False
        attempt=by_attempt[decision['forward_attempt']]
        assert attempt['kind']=='forward' and attempt['seed']==decision['seed'] and attempt['reason']=='frozen_proposal:'+identifier
        sequence=reencode(decision)
        if decision['observations']:
            matches=[]
            for path in (run/'fixtures').glob('*/trace.json'):
                trace=read(path)
                if trace['task_id']==decision['task_id'] and trace['scope']==decision['scope']:matches.append(trace)
            assert len(matches)==1 and len(matches[0]['events'])==1
            event=matches[0]['events'][0];data=event['result']
            expected_observation={'tool':event['tool'],'reported_status':'ok' if data.get('ok') is True else 'error',
                'error_type':data.get('error',''),'empty_result':False,'spillover':False,'streak':1,
                'probe':None,'returned_epoch':event['returned_epoch']}
            assert decision['observations']==[expected_observation]
        with np.load(run/decision['measurement'],allow_pickle=False) as data:
            measured={k:data[k] for k in data.files}
        np.testing.assert_array_equal(sequence,measured['sequence'])
        assert hashlib.sha256(sequence.tobytes()).hexdigest()==decision['input_sha256']
        before,ma=model_object(run,decision['before_model_sha256']);after,mb=model_object(run,decision['after_model_sha256'])
        assert ma['seed']==mb['seed']==decision['seed'] and mb['ticks']==ma['ticks']+len(sequence)
        assert all(np.array_equal(v,after[k]) for k,v in before.items() if k not in DYNAMIC)
        if decision['seed'] not in per_seed:per_seed[decision['seed']]=before
        assert all(np.array_equal(v,per_seed[decision['seed']][k]) for k,v in before.items() if k not in DYNAMIC)
        expected_scale=np.maximum(np.sqrt(after['f_var']),.05)
        f=np.clip((after['core.features']-after['f_mean'])/expected_scale,-5,5)
        np.testing.assert_allclose(f,measured['features'],rtol=3e-6,atol=3e-6)
        logits=after['core.w_out'][2:8].astype(np.float64)@f.astype(np.float64)+after['core.bias'][2:8]
        probability=np.exp(logits-logits.max());probability/=probability.sum()
        error=float(np.max(np.abs(probability-measured['probabilities'])));max_probability_error=max(max_probability_error,error)
        assert error<=3e-6
        assert all(decision['probabilities'][a]==float(measured['probabilities'][i]) for i,a in enumerate(ACTIONS))
        selected=max(decision['allowed'],key=lambda a:decision['probabilities'][a])
        assert selected==decision['proposed_action']
        with sqlite3.connect((run/('journal-'+str(decision['seed'])+'.sqlite')).as_uri()+'?mode=ro',uri=True) as db:
            stored=json.loads(db.execute('SELECT payload FROM decisions WHERE id=?',(identifier,)).fetchone()[0])
            assert stored==decision
    for seed_result in result['seed_results']:
        seed=seed_result['seed']
        assert len([d for d in decisions.values() if d['seed']==seed])==9
        assert len(seed_result['case5_rejections'])==4 and len(seed_result['case6_rejections'])==3
        for entry in seed_result['ordinary']+[seed_result['case7']]:
            receipt=entry['receipt'];decision=decisions[entry['decision_id']]
            assert receipt==read(run/'receipts'/(decision['id']+'.json'))
            assert receipt['decision_id']==decision['id'] and receipt['task_id']==decision['task_id'] and receipt['scope']==decision['scope']
            attempt=by_attempt[receipt['dispatch_attempt']]
            assert attempt['kind']=='dispatch' and attempt['reason']=='action_macro:'+decision['id']
            assert receipt['proposed_action']==decision['proposed_action'] and receipt['learning_authorized'] is False
            with sqlite3.connect((run/('journal-'+str(seed)+'.sqlite')).as_uri()+'?mode=ro',uri=True) as db:
                stored=json.loads(db.execute('SELECT payload FROM receipts WHERE id=?',(decision['id'],)).fetchone()[0])
                assert stored==receipt
                active=db.execute('SELECT active FROM contexts WHERE task_id=?',(decision['task_id'],)).fetchone()[0]
            directory=run/entry['fixture'];trace=read(directory/'trace.json');contract=read(directory/'contract.json')
            assert trace['task_id']==decision['task_id'] and trace['scope']==decision['scope']
            assert digest(trace)==receipt['trace_sha256'] and contract['task_id']==trace['task_id']
            snapshots={p.stem:read(p) for p in (directory/'snapshots').glob('*.json')}
            assert all(digest(v)==k for k,v in snapshots.items())
            final=snapshots[trace['final_snapshot']];project=directory/'project'
            assert {p.name:{'text':p.read_bytes().decode(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in project.glob('*.txt')}==final['files']
            with sqlite3.connect((project/'state.sqlite').as_uri()+'?mode=ro',uri=True) as db:
                assert dict(zip(('progress','status'),db.execute('SELECT progress,status FROM job').fetchone()))==final['job']
                assert [r[0] for r in db.execute('SELECT field FROM clarifications ORDER BY rowid')]==final['clarifications']
            if receipt['guard_override']:
                assert active==0 and receipt['executed_action']=='deny' and receipt['selector_origin']=='runtime_scope_guard'
                assert not receipt['raw_tool_events'] and not trace['events'] and not receipt['outcome']['whole_task_completion']
            else:
                assert active==1 and receipt['executed_action']==decision['proposed_action'] and receipt['selector_origin']=='neural_policy'
                assert len(trace['events'])==1 and len(receipt['raw_tool_events'])==1
                event=trace['events'][0];ref=receipt['raw_tool_events'][0]
                assert event['decision_id']==ref['tool_event_id'] and digest(event)==ref['sha256']
                assert event['selector']=='fixed_fixture' # executor producer, not neural selection origin
                assert decision['created']<=event['started_epoch']<=event['returned_epoch']
                request=decision['public_task']['request'];action=decision['proposed_action']
                expected_tool=('show_schema' if action=='show_schema' else 'ask_field' if action=='clarify'
                    else 'read_file' if request['tool']=='read_file' else 'inspect_status' if action=='inspect_status' else request['tool'])
                assert event['tool']==expected_tool
                effect=False;clarification=False
                if event['tool']=='read_file' and event['result'].get('ok'):
                    assert event['arguments']==request['arguments'] and event['result']['source']==contract['source']
                    data=snapshots[event['before_snapshot']]['files'][contract['source']]
                    assert event['result']['text']==data['text']
                    effect=data['sha256']==contract['value']
                elif event['tool']=='read_file':
                    assert request['arguments']['path'] not in snapshots[event['before_snapshot']]['files']
                    assert event['result']['error']=='FileNotFoundError'
                elif contract['kind']=='job':effect=final['job']=={'progress':3,'status':'completed'}
                elif event['tool']=='ask_field':clarification=final['clarifications']==['alarm_time']
                if event['tool']=='show_schema':
                    assert event['result']['schema']==read(project/'registry.json')[request['tool']]
                assert receipt['outcome']['task_effect_satisfied']==effect
                assert receipt['outcome']['clarification_supported']==clarification
                assert receipt['outcome']['whole_task_completion']==effect
                ordinary.append({'seed':seed,'case':entry['case'],'action':action,'raw_tool':event['tool'],
                    'task_effect_satisfied':effect,'clarification_supported':clarification})
        restored=seed_result['case8'];ids=restored['continuation_ids'];group=[decisions[i] for i in ids]
        assert all({k:d[k] for k in ('proposed_action','probabilities','input_sha256','allowed')}==
                   {k:group[0][k] for k in ('proposed_action','probabilities','input_sha256','allowed')} for d in group)
        assert len({d['after_model_sha256'] for d in group})==1
        raw=(run/restored['snapshot_file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==restored['snapshot_sha256']
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            meta=json.loads(z.read('metadata.json'));inner=z.read('local-alif.npz')
        assert restored['pending_decision_id'] in meta['pending']
        assert hashlib.sha256(inner).hexdigest()==meta['model_sha256']
        assert all(restored[k] for k in ('same_device_proposals_exact','inner_model_bytes_exact','snapshot_roundtrip_exact','journal_never_rewound','revoked_scope_stays_revoked'))
        assert set(restored['old_loaders_rejected'])=={'production','old_shadow'}
        with sqlite3.connect((run/('journal-'+str(seed)+'.sqlite')).as_uri()+'?mode=ro',uri=True) as db:
            assert db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]==9
            assert db.execute('SELECT COUNT(*) FROM receipts').fetchone()[0]==5
    resources=read(run/'resources.json')
    assert all(max(r['peak_RSS_bytes'],r['peak_private_commit_bytes'])<=1_000_000_000 for r in resources)
    assert min(r['physical_cuda_free_bytes'] for r in resources if r['physical_cuda_free_bytes'] is not None)>=512*1024**2
    assert result['elapsed_seconds']<=600 and result['learn_calls']==result['native_VLM_ASR_calls']==0
    if not diagnostic_stopped:assert not result['blocked_attempts']
    live=read(run/'live-state-audit.json');assert live['equal'] and live['before']==live['after']==manifest['live_before']
    assert sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<=20_000_000
    return {'verdict':('P1A_STOPPED_RUN_DATA_DIAGNOSTIC_MATCHES_NOT_PROTOCOL_PASS' if diagnostic_stopped
                       else 'P1A_INDEPENDENT_DECISION_MACRO_STATE_READOUT_LEDGER_AUDIT_MATCHES'),
        'diagnostic_only':diagnostic_stopped,'original_result_gate_pass':result['gate_pass'],
        'original_guard_gate_satisfied':not result['blocked_attempts'],
        'original_blocked_attempts':result['blocked_attempts'],
        'current_postrun_auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'registered_auditor_sha256':manifest['source_sha256']['scripts/audit_hybrid_neural_outcome_bridge_p1a.py'],
        'technical_cases':8,'seeds':[260113,260137],'actual_GPU_forwards':18,'dispatch_attempts':10,
        'actual_tool_macro_executions':8,'scope_guard_denials':2,'ordinary_selections':ordinary,
        'maximum_independent_readout_probability_error':max_probability_error,
        'same_device_continuation_exact':True,'new_forward_or_tool_calls_in_audit':0,
        'independent_readout_math_is_not_CPU_GPU_trajectory_certification':True,
        'same_orchestrator_independent_code_not_external_peer_approval':True,
        'learned_skill_retention_hidden_advantage_or_production_PASS':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--diagnostic-stopped',action='store_true',help='Read completed data from a failed run without certifying its guard gate')
    args=parser.parse_args();report=audit(args.run,args.diagnostic_stopped)
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))
