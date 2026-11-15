"""Single approved P1A: frozen CUDA ALIF -> owned macro -> bound outcome.

No VLM/ASR calls, learning, new credit rule, serving import or live promotion.
"""
from datetime import datetime,timezone
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback
import uuid

import numpy as np
import psutil
import platform

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.cuda_alif import cupy_backend
from scarx.evaluation.neural_outcome_bridge import FrozenBridgePolicy,BindingJournal,encode_task_observations
from scarx.evaluation.neural_bridge_sandbox import OwnedMacroSandbox,P0
from scarx.evaluation.bridge_attempt_budget import AttemptBudget
from scarx.evaluation.recovery_policy import RecoveryPolicy
from scarx.evaluation.bridge_effect_guard import StickyEffectGuard
from scarx.evaluation.bridge_environment_metrics import sample_physical_gpu

SEEDS=(260113,260137)
TASKS=[
    ('source',{'instruction':'Read requested.txt in this owned project.','request':{'tool':'read_file','arguments':{'path':'requested.txt'}}},('inspect_status','retry')),
    ('missing',{'instruction':'Read missing.txt in this owned project.','request':{'tool':'read_file','arguments':{'path':'missing.txt'}}},('inspect_status','retry')),
    ('job',{'instruction':'Check the job progress and retry its poll if useful.','request':{'tool':'poll','arguments':{}}},('inspect_status','retry')),
    ('field',{'instruction':'Create an alarm; ask for a missing parameter if needed.','request':{'tool':'create_alarm','arguments':{}}},('show_schema','clarify')),
]


def json_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(P0.encode(value))


def signature(decision):
    return {k:decision[k] for k in ('proposed_action','probabilities','input_sha256','allowed')}


def journal_state(journal):
    return {table:list(journal.db.execute('SELECT * FROM '+table+' ORDER BY 1'))
            for table in ('contexts','decisions','receipts','metadata')}


def main():
    started=time.time();run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-neural-outcome-bridge-p1a-v2-'+uuid.uuid4().hex[:6])
    run.mkdir();os.environ['CUPY_CACHE_DIR']=str(run/'cupy-cache');os.environ['CUDA_CACHE_PATH']=str(run/'cuda-cache')
    marker=ROOT/'local/neural-outcome-bridge-p1a-v2-execution.json'
    json_write(marker,{'run':str(run),'started_epoch':started,'single_approved_run':True})
    (ROOT/'local/current-neural-outcome-bridge-p1a-v2.txt').write_text(str(run),encoding='utf-8')
    budget=AttemptBudget(ROOT/'local/neural-outcome-bridge-p1a-v2-budget.jsonl',started)
    names=['scripts/run_hybrid_neural_outcome_bridge_p1a_v2.py','scripts/audit_hybrid_neural_outcome_bridge_p1a.py',
        'src/scarx/evaluation/neural_outcome_bridge.py','src/scarx/evaluation/neural_bridge_sandbox.py',
        'src/scarx/evaluation/bridge_attempt_budget.py','tests/test_neural_outcome_bridge.py',
        'experiments/hybrid-neural-bridge-guard-confirmation-v2.md',
        'src/scarx/evaluation/hybrid_incident_encoder.py','src/scarx/evaluation/recovery_policy.py',
        'src/scarx/evaluation/outcome_contract.py','scripts/run_hybrid_outcome_contract_p0.py',
        'src/scarx/hybrid/learner.py','src/scarx/hybrid/cuda_alif.py','src/scarx/neural/core.py',
        'src/scarx/evaluation/bridge_effect_guard.py','src/scarx/evaluation/bridge_environment_metrics.py',
        'scripts/prepare_neural_bridge_confirmation_v2.py','tests/test_bridge_effect_guard.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    for name in names:
        dest=run/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    live_before=P0.live_learned_state()
    system=Path(os.environ['WINDIR'])/'System32'
    external={'platform':Path(platform.__file__).resolve(),'cmd':(system/'cmd.exe').resolve(),
              'gpu':(system/'nvidia-smi.exe').resolve(),
              'monitor':(ROOT/'src/scarx/evaluation/bridge_environment_metrics.py').resolve()}
    pins={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in external.items()}
    json_write(run/'environment-pins.json',pins)
    prior=ROOT/'runs/20261115T054242Z-neural-outcome-bridge-p1a-v1-f39bc6'
    previous={'run':prior.name,'forwards':18,'dispatches':10,
        'result_sha256':hashlib.sha256((prior/'result.json').read_bytes()).hexdigest(),
        'ledger_sha256':hashlib.sha256((prior/'attempt-ledger.jsonl').read_bytes()).hexdigest()}
    json_write(run/'manifest.json',{'source_git_sha':os.environ.get('SCARX_P1A_SOURCE_SHA'),
        'pid':os.getpid(),'process_created_epoch':psutil.Process().create_time(),
        'confirmation_of':previous,'environment_pins':pins,
        'source_sha256':hashes,'started_epoch':started,'seeds':list(SEEDS),'max_forwards':24,
        'max_dispatch_attempts':10,'max_tool_executions':8,'max_seconds':600,'max_process_bytes':1_000_000_000,
        'max_artifact_bytes':20_000_000,'min_free_cuda_bytes':512*1024**2,
        'locked_public_tasks':TASKS,'expected_forwards':18,'expected_dispatches':10,
        'live_before':live_before,'approval':'User approved confirmation v2 on15/11/2026; additional quota, no reset of v1',
        'P0_parent':'20261115T044756Z-outcome-contract-p0-v1-5842c6','ponytail_available':False})
    preflight=ROOT/'local/neural-bridge-v2-preflight-20261115.xml'
    (run/'preflight.xml').write_bytes(preflight.read_bytes())
    report={'run_id':run.name,'gate_pass':False,'failures':[],'seed_results':[],
        'learn_calls':0,'native_VLM_ASR_calls':0,'live_model_changed':False,'research_or_production_PASS':False}
    memory=[];blocked=[];cp=None;journals=[];sandboxes=[]
    original_learn=LocalALIF.learn
    def no_learning(*args,**kwargs):
        report['learn_calls']+=1;raise RuntimeError('Learning is forbidden in P1A')
    LocalALIF.learn=no_learning
    effect_guard=StickyEffectGuard(
        pins['platform']['path'],pins['platform']['sha256'],pins['cmd']['path'],pins['cmd']['sha256'],
        pins['monitor']['path'],pins['monitor']['sha256'],pins['gpu']['path'],pins['gpu']['sha256'])
    blocked=effect_guard.violations
    def guard(event,args):
        if event=='import' and str(args[0]).split('.')[0]=='torch':
            blocked.append({'event':'torch_import'});raise PermissionError('No autograd control in P1A')
        effect_guard.hook(event,args)
    sys.addaudithook(guard)
    def resources():
        effect_guard.check() # Sticky even if an SDK swallowed its PermissionError.
        info=psutil.Process().memory_info()
        rss=max(info.rss,getattr(info,'peak_wset',0));private=max(getattr(info,'private',0),getattr(info,'peak_pagefile',0))
        available,available_total=(map(int,cp.cuda.runtime.memGetInfo()) if cp is not None else (None,None))
        physical=sample_physical_gpu(pins['gpu']['path'],pins['gpu']['sha256'])
        effect_guard.check()
        free,total=physical['free_bytes'],physical['total_bytes']
        memory.append({'epoch':time.time(),'peak_RSS_bytes':rss,'peak_private_commit_bytes':private,
                       'physical_cuda_free_bytes':free,'physical_cuda_total_bytes':total,
                       'physical_gpu_source':physical['source'],
                       'cuda_runtime_available_bytes':available,'cuda_runtime_total_bytes':available_total})
        if max(rss,private)>1_000_000_000 or time.time()-started>600:raise RuntimeError('P1A time/process-memory cap')
        if free is not None and free<512*1024**2:raise RuntimeError('Physical CUDA reserve breached')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>20_000_000:raise RuntimeError('Artifact cap breached')
    def charge(kind,seed,reason):resources();return budget.charge(kind,seed,reason)
    def save_blob(raw):
        resources();sha=hashlib.sha256(raw).hexdigest();path=run/'model-objects'/(sha+'.npz')
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            if path.read_bytes()!=raw:raise ValueError('Immutable model object changed')
        else:
            with path.open('xb') as f:f.write(raw)
        return sha
    def save_arrays(identifier,arrays):
        path=run/'measurements'/(identifier+'.npz');path.parent.mkdir(exist_ok=True)
        with path.open('xb') as f:np.savez(f,**arrays)
        return path.relative_to(run).as_posix()
    def propose(policy,sandbox,observations,allowed,tag,cutoff=None):
        decision=policy.propose(sandbox.task_id,sandbox.scope,sandbox.task,observations,
                                time.time() if cutoff is None else cutoff,allowed)
        if not all(getattr(v,'device',None) is not None and v.device.id==0 for v in policy.model.arrays().values()):
            raise ValueError('A neural array is not on CUDA0')
        json_write(run/'decisions'/(decision['id']+'.json'),decision)
        P0.publish(run,'P1A '+tag,json.dumps({'decision_id':decision['id'],'seed':decision['seed'],
            'origin':decision['selector'],'allowed':decision['allowed'],'proposed_action':decision['proposed_action'],
            'probabilities':decision['probabilities'],'device':'gpu:0','updates':0},ensure_ascii=False))
        resources();return decision
    try:
        P0.publish(run,'P1A started','Frozen ALIF H128 on CUDA0; max24 forwards/10 dispatch attempts, no learning.','started')
        cp=cupy_backend();effect_guard.check()
        cp.cuda.Device(0).use();effect_guard.check();resources()
        if budget.records():raise ValueError('Fresh v2 ledger was already consumed before E0')
        json_write(run/'environment-preflight.json',{'passed':True,'forward_attempts':0,'dispatch_attempts':0,
            'before_ALIF_initialization':True,'dedicated_gpu_sample':memory[-1],
            'allowed_queries':effect_guard.allowed_queries,'declined_sdk_probes':effect_guard.declined_sdk_probes})
        report['environment_preflight_passed']=True
        P0.publish(run,'P1A-v2 environment gate passed','Pinned SDK metadata handling and dedicated GPU query passed before ALIF initialization.')
        for seed in SEEDS:
            journal=BindingJournal(run/('journal-'+str(seed)+'.sqlite'));journals.append(journal)
            policy=FrozenBridgePolicy(seed,journal,charge,save_blob,save_arrays);resources()
            result={'seed':seed,'ordinary':[],'case5_rejections':[],'case6_rejections':[]}
            owned=[]
            for index,(name,task,allowed) in enumerate(TASKS,1):
                sandbox=OwnedMacroSandbox(run,f'{seed}-case{index}-{name}',task,journal,charge)
                owned.append(sandbox);sandboxes.append(sandbox)
                decision=propose(policy,sandbox,[],allowed,'case'+str(index))
                receipt=sandbox.dispatch(decision);policy.settle(receipt,decision['id'])
                if receipt['guard_override'] or receipt['executed_action']!=decision['proposed_action']:raise ValueError('Unexpected override')
                json_write(run/'receipts'/(decision['id']+'.json'),receipt);sandbox.save()
                P0.publish(run,'P1A actual outcome case'+str(index),json.dumps({
                    'decision_id':decision['id'],'public_task':sandbox.task,'proposed_action':decision['proposed_action'],
                    'executed_action':receipt['executed_action'],'origin':receipt['selector_origin'],
                    'raw_tool_results':[e['result'] for e in sandbox.fixture.trace['events']],
                    'outcome':receipt['outcome'],'updates':0},ensure_ascii=False))
                result['ordinary'].append({'case':index,'decision_id':decision['id'],'receipt':receipt,
                    'fixture':sandbox.fixture.directory.relative_to(run).as_posix()})
            count=len(budget.records());receipt=result['ordinary'][0]['receipt'];expected=receipt['decision_id']
            for key in ('task_id','scope','decision_id'):
                changed=copy.deepcopy(receipt);changed[key]='foreign-'+changed[key]
                try:journal.validate_receipt(changed,expected)
                except ValueError:result['case5_rejections'].append(key)
                else:raise ValueError('Foreign receipt accepted')
            try:journal.validate_receipt(result['ordinary'][1]['receipt'],expected)
            except ValueError:result['case5_rejections'].append('other_valid_receipt')
            else:raise ValueError('Valid receipt used for another expected decision')
            source=owned[1];observations=source.observations();assert observations and observations[-1]['reported_status']=='error'
            for bad in ('future','private_task','private_observation'):
                task=copy.deepcopy(source.task);rows=copy.deepcopy(observations);cutoff=time.time()
                if bad=='future':rows[-1]['returned_epoch']=cutoff+10
                elif bad=='private_task':task['contract']={'expected':'private verifier data'}
                else:rows[-1]['case_id']='fixture identity'
                try:policy.propose(source.task_id,source.scope,task,rows,cutoff,('inspect_status','retry'))
                except ValueError:result['case6_rejections'].append(bad)
                else:raise ValueError('Malformed policy input accepted')
            if len(budget.records())!=count:raise ValueError('Malformed cases reached a forward/dispatch')
            guarded=OwnedMacroSandbox(run,f'{seed}-case7-guard',TASKS[0][1],journal,charge);sandboxes.append(guarded)
            decision=propose(policy,guarded,[],TASKS[0][2],'case7-guard')
            journal.revoke(guarded.task_id);denial=guarded.dispatch(decision);policy.settle(denial,decision['id'])
            assert denial['guard_override'] and denial['executed_action']=='deny' and not denial['raw_tool_events']
            json_write(run/'receipts'/(decision['id']+'.json'),denial);guarded.save()
            P0.publish(run,'P1A actual guard denial',json.dumps(denial,ensure_ascii=False))
            result['case7']={'decision_id':decision['id'],'receipt':denial,'fixture':guarded.fixture.directory.relative_to(run).as_posix()}
            pending=propose(policy,source,observations,TASKS[1][2],'case8-pending')
            snapshot=policy.dumps();path=run/('seed-'+str(seed)+'-pending.bridge');path.write_bytes(snapshot)
            cutoff=time.time();future=propose(policy,source,observations,TASKS[1][2],'case8-original-continuation',cutoff)
            expected_model=policy.model.dumps();journal_before=journal_state(journal)
            restored=FrozenBridgePolicy.loads(snapshot,journal,charge,save_blob,save_arrays)
            assert restored.dumps()==snapshot and pending['id'] in restored.pending
            assert journal_state(journal)==journal_before
            cold=propose(restored,source,observations,TASKS[1][2],'case8-cold-continuation',cutoff)
            assert signature(cold)==signature(future) and restored.model.dumps()==expected_model
            journal_before_rollback=journal_state(journal)
            rolled=FrozenBridgePolicy.loads(snapshot,journal,charge,save_blob,save_arrays)
            assert rolled.dumps()==snapshot and journal_state(journal)==journal_before_rollback
            rollback=propose(rolled,source,observations,TASKS[1][2],'case8-rollback-continuation',cutoff)
            assert signature(rollback)==signature(future) and rolled.model.dumps()==expected_model
            assert not journal.context(guarded.task_id,guarded.scope,guarded.task,allow_revoked=True)
            rejected=[]
            for label,loader in [('production',LocalALIF),('old_shadow',RecoveryPolicy)]:
                try:loader.loads(snapshot)
                except (ValueError,KeyError):rejected.append(label)
                else:raise ValueError('Old loader accepted bridge envelope')
            result['case8']={'pending_decision_id':pending['id'],'continuation_ids':[future['id'],cold['id'],rollback['id']],
                'same_device_proposals_exact':True,'inner_model_bytes_exact':True,'snapshot_roundtrip_exact':True,
                'journal_never_rewound':True,'revoked_scope_stays_revoked':True,'old_loaders_rejected':rejected,
                'snapshot_file':path.relative_to(run).as_posix(),'snapshot_sha256':hashlib.sha256(snapshot).hexdigest()}
            assert len(result['case5_rejections'])==4 and len(result['case6_rejections'])==3 and len(rejected)==2
            report['seed_results'].append(result)
            policy.assert_frozen();resources()
        rows=budget.records();counts={kind:sum(r['kind']==kind for r in rows) for kind in ('forward','dispatch')}
        if counts!={'forward':18,'dispatch':10}:raise ValueError('Execution plan/counter mismatch')
        if hashes!={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}:raise ValueError('Source changed during run')
        live_after=P0.live_learned_state();json_write(run/'live-state-audit.json',{'before':live_before,'after':live_after,'equal':live_before==live_after})
        if live_before!=live_after or report['learn_calls'] or blocked or any(n=='torch' or n.startswith('torch.') for n in sys.modules):
            raise ValueError('Live/model-call/learning invariant failure')
        resources();report.update(gate_pass=True,verdict='P1A_FROZEN_NEURAL_ACTION_OUTCOME_BINDING_PASS_NOT_LEARNED_SKILL',
            technical_cases=8,seeds=list(SEEDS),forwards=18,dispatch_attempts=10,tool_macro_executions=8,
            scope_guard_denials=2,live_saved_state_equal=True)
    except Exception as exc:
        report['failures'].append(type(exc).__name__+': '+str(exc));report['verdict']='P1A_STOPPED_NOT_PASS'
        (run/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        LocalALIF.learn=original_learn
        for sandbox in sandboxes:sandbox.close()
        for journal in journals:journal.close()
        json_write(run/'resources.json',memory)
        json_write(run/'guard-events.json',{'violations':effect_guard.violations,
            'allowed_queries':effect_guard.allowed_queries,'declined_sdk_probes':effect_guard.declined_sdk_probes,
            'gpu_queries':effect_guard.gpu_queries,'ver_queries_declined':effect_guard.ver_queries})
        if budget.path.exists():(run/'attempt-ledger.jsonl').write_bytes(budget.path.read_bytes())
        budget.close()
        report.update(elapsed_seconds=time.time()-started,blocked_attempts=blocked,
            environment_preflight_passed=report.get('environment_preflight_passed',False),
            peak_RSS_bytes=max([r['peak_RSS_bytes'] for r in memory]+[0]),
            peak_private_commit_bytes=max([r['peak_private_commit_bytes'] for r in memory]+[0]),
            minimum_sampled_cuda_free_bytes=min([r['physical_cuda_free_bytes'] for r in memory if r['physical_cuda_free_bytes'] is not None],default=None))
        json_write(run/'result.json',report)
        P0.publish(run,'P1A '+('completed' if report['gate_pass'] else 'stopped'),json.dumps({k:v for k,v in report.items() if k!='seed_results'}),
                   'completed' if report['gate_pass'] else 'failed')
    print(json.dumps({'run':str(run),**{k:v for k,v in report.items() if k!='seed_results'}},ensure_ascii=False),flush=True)
    return 0 if report['gate_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
