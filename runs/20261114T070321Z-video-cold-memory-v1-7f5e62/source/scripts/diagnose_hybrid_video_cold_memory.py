"""Actual private process restart, existing video evidence and shared native VLM."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from scarx.hybrid.store import Store,encode
from scarx.hybrid.device_budget import gpu_headroom
from scarx.evaluation.shared_native import SharedNative


def sha(raw):return hashlib.sha256(raw).hexdigest()


def read_only_db(directory):
    path=(directory/'hybrid.sqlite').resolve()
    return sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)


def memory_state(db):
    evidence=[tuple(row) for row in db.execute('SELECT * FROM evidence ORDER BY id')]
    streams=[tuple(row) for row in db.execute("SELECT id,payload FROM objects WHERE kind='stream' ORDER BY id")]
    return {'evidence_sha256':sha(encode(evidence).encode()),'evidence_count':len(evidence),
            'stream_sha256':sha(encode(streams).encode()),'stream_count':len(streams)}


def learned_state(model):
    arrays={name:model.host(value) for name,value in model.arrays().items()
            if name.startswith('core.') and name[5:] in model.core.weight_names
            or name in ('importance','f_mean','f_var','syn_trace','syn_energy','gamma')}
    return {'array_sha256':{name:sha(value.tobytes()) for name,value in arrays.items()},
        'indices_sha256':sha(model.host(model.core.indices).tobytes()),
        'rng':encode(model.rng.bit_generator.state),'signal_rng':encode(model.signal_rng.bit_generator.state),
        'replay_sha256':sha(encode(model.replay).encode()),'replay_seen':model.replay_seen,
        'updates':model.updates,'hidden_updates':model.hidden_updates,'core_updates':model.core.updates,
        'reward_mean':model.reward_mean,'seed':model.seed,'mode':model.mode}


def child_phase(run,cell,phase):
    from scarx.hybrid.engine import HybridEngine
    from scarx.hybrid.config import HybridConfig
    from scarx.hybrid.resources import ResourceMonitor
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    info=manifest['cells'][cell]
    directory=ROOT/'local'/run.name/cell
    output=run/(cell+'-'+phase+'.json')
    assert not output.exists()
    native=SharedNative(ROOT)
    native.check()
    assert gpu_headroom()['free_MiB']>=512
    monitor=ResourceMonitor(limit=None);monitor.start()
    store=Store(directory)
    calls=[]
    class Backend:
        cancel_idle=threading.Event()
        def generate(self,messages,max_tokens=128,**kwargs):
            assert gpu_headroom()['free_MiB']>=512
            body={'messages':messages,'max_tokens':min(max_tokens,128)}
            began=time.monotonic();raw=native.generate(body)
            calls.append({'request':body,'request_sha256':sha(encode(body).encode()),
                          'raw':raw,'seconds':time.monotonic()-began})
            return {'text':raw['choices'][0]['message'].get('content') or '',
                    'usage':raw.get('usage',{}),'raw':raw}
    engine=None
    started=time.monotonic()
    report={'cell':cell,'phase':phase,'pid':os.getpid(),'process_created_epoch':psutil.Process().create_time(),
            'selector_override':info['selector'],'answers':[],'failures':[],'native_calls':calls,
            'no_background_thread_or_learning':True,'VLM_worker_pid':native.worker.pid,
            'VLM_worker_created':native.worker_created,'expected_SNN_seed':260113}
    trace=Store(ROOT/'local/hybrid-v1')
    try:
        active=store.get('runtime','active_checkpoint')
        payload=store.snapshot(active['id'])
        report['entry_checkpoint_payload_sha256']=sha(payload)
        engine=HybridEngine(store,Backend(),monitor,HybridConfig(execution_mode='gpu',asr_revision=info['ASR_revision']))
        engine.model.to_device('gpu')
        report['entry_memory']=memory_state(store.db)
        report['entry_learned_state']=learned_state(engine.model)
        assert not engine.state['learning_enabled']
        assert engine.model.updates==engine.model.hidden_updates==0
        assert engine.model.seed==260113
        if info['selector']=='RAG_candidateorder':
            engine.model.relevance=lambda query,evidence:0.0
        for question in manifest['questions']:
            answer=engine.ask(question['question'],stream=info['stream_id'])
            assert len(calls)==len(report['answers'])+1
            row={**question,'actual':answer,'request_sha256':calls[-1]['request_sha256'],
                 'strict_original_contract_passed':answer['answer'].strip().casefold()==question['expected'],
                 'selector_override':info['selector'],'model_learned_updates':engine.model.updates,
                 'raw_engine_retrieval_string_is_generic_not_override_identity':True}
            report['answers'].append(row)
            trace.event('video_cold_memory.actual_answer',{
                'run_id':run.name,'cell':cell,'phase':phase,**row,
                'known_clip_not_newTEST':True,'not_live_neural_update':True},'orchestrator')
        assert memory_state(store.db)==report['entry_memory']
        assert learned_state(engine.model)==report['entry_learned_state']
        engine.close();engine=None
        active=store.get('runtime','active_checkpoint')
        payload=store.snapshot(active['id'])
        report['exit_checkpoint_payload_sha256']=sha(payload)
        report['exit_memory']=memory_state(store.db)
        (run/(cell+'-'+phase+'-saved-checkpoint.bin')).write_bytes(payload)
        report['native_calls_completed']=len(calls)
        report['strict_score']=sum(a['strict_original_contract_passed'] for a in report['answers'])
        report['verdict']='PRIVATE_COLD_PHASE_COMPLETED_NOT_QUALITY_PASS'
    except Exception:
        report['failures'].append(traceback.format_exc());report['verdict']='PRIVATE_COLD_PHASE_FAILED'
    finally:
        if engine:engine.close()
        report['seconds']=time.monotonic()-started;report['resources']=monitor.close()
        store.close();trace.close()
        with output.open('x',encoding='utf-8') as f:f.write(encode(report))
    return not report['failures']


def main():
    parent=ROOT/'runs/20261114T020701Z-asr-video-integration-p0-a98122'
    result_raw=(parent/'result.json').read_bytes();prior=json.loads(result_raw)
    assert not prior['failures']
    questions=[{key:row[key] for key in ('id','question','expected')}
               for row in prior['arms']['base']['answers']]
    assert len(questions)==3 and [q['expected'] for q in questions]==['15','9','yellow']
    name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-video-cold-memory-v1-'+uuid.uuid4().hex[:6]
    run=ROOT/'runs'/name;run.mkdir()
    data=ROOT/'local'/name;data.mkdir()
    cells={};source_memory={};source_hashes={}
    for asr in ('base','small'):
        source=ROOT/'local'/parent.name/asr
        assert source.is_relative_to(ROOT/'local') and source.is_dir()
        with read_only_db(source) as db:
            source_memory[asr]=memory_state(db)
            active=json.loads(db.execute("SELECT payload FROM objects WHERE kind='runtime' AND id='active_checkpoint'").fetchone()[0])
            checkpoint=db.execute('SELECT payload FROM checkpoints WHERE id=?',(active['id'],)).fetchone()[0]
            source_hashes[asr]={'database_sha256':sha((source/'hybrid.sqlite').read_bytes()),
                                'checkpoint_payload_sha256':sha(checkpoint),'memory':source_memory[asr],
                                'source_media':[]}
            for (media,) in db.execute('SELECT DISTINCT source FROM evidence'):
                path=Path(media).resolve()
                assert path.is_relative_to(source.resolve()),'Unexpected source path outside owned video fixture'
                source_hashes[asr]['source_media'].append({'path':str(path),'available':path.is_file(),
                    'sha256':sha(path.read_bytes()) if path.is_file() else None})
            for selector in ('SNN_frozen_existing','RAG_candidateorder'):
                cell=asr+'-'+selector;directory=data/cell;directory.mkdir()
                destination=sqlite3.connect(directory/'hybrid.sqlite')
                db.backup(destination);destination.close()
                cells[cell]={'ASR_source':asr,'selector':selector,
                    'ASR_revision':prior['arms'][asr]['ASR_identity']['revision'],
                    'stream_id':prior['arms'][asr]['stream']['id']}
    paths=list((ROOT/'src/scarx/hybrid').glob('*.py'))+[
        Path(__file__),ROOT/'src/scarx/neural/core.py',
        ROOT/'src/scarx/evaluation/readout_eligibility.py',ROOT/'src/scarx/evaluation/shared_native.py',
        ROOT/'experiments/hybrid-video-cold-memory-diagnostic-v1.md']
    hashes={}
    for path in paths:
        raw,rel=path.read_bytes(),path.relative_to(ROOT);dest=run/'source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);hashes[rel.as_posix()]=sha(raw)
    native=SharedNative(ROOT)
    manifest={'run_id':run.name,'source_sha256':hashes,'parent_result_sha256':sha(result_raw),
        'parent_run':parent.name,'original_owned_source_sha256':source_hashes,'cells':cells,'questions':questions,
        'max_native_calls':24,'max_seconds':600,'neural_learning':False,
        'original_native_pid':native.worker.pid,'original_native_created':native.worker_created,
        'source_scope':'Previously captured21second licensed engineering clip, not new qualitysamples'}
    (run/'manifest.json').write_text(encode(manifest),encoding='utf-8')
    trace=Store(ROOT/'local/hybrid-v1');started=time.monotonic()
    report={'run_id':run.name,'cells':[],'failures':[],'native_calls':0,'new_learning':False,
            'SNN_advantage_or_lifelong_quality_PASS':False}
    try:
        for cell,info in cells.items():
            phases=[]
            for phase in ('before','cold_restart'):
                assert time.monotonic()-started<600
                assert gpu_headroom()['free_MiB']>=512
                with (run/(cell+'-'+phase+'.log')).open('wb') as log:
                    command=[sys.executable,str(Path(__file__)),'--child',str(run),'--cell',cell,'--phase',phase]
                    child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONIOENCODING='utf-8'),
                        stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    code=child.wait(timeout=max(1,600-(time.monotonic()-started)))
                assert code==0,'Private child failed; keep actual phase log/result'
                measured=json.loads((run/(cell+'-'+phase+'.json')).read_text(encoding='utf-8'))
                assert not measured['failures'] and measured['native_calls_completed']==3
                assert measured['VLM_worker_pid']==native.worker.pid and measured['VLM_worker_created']==native.worker_created
                phases.append(measured);report['native_calls']+=measured['native_calls_completed']
            a,b=phases
            assert a['pid']!=b['pid'] and b['process_created_epoch']>a['process_created_epoch']
            assert a['exit_checkpoint_payload_sha256']==b['entry_checkpoint_payload_sha256']
            assert a['entry_memory']==a['exit_memory']==b['entry_memory']==b['exit_memory']==source_memory[info['ASR_source']]
            assert a['entry_learned_state']==b['entry_learned_state']
            pairs=[]
            for x,y in zip(a['answers'],b['answers']):
                assert x['id']==y['id'] and x['request_sha256']==y['request_sha256']
                assert x['actual']['candidate_ids']==y['actual']['candidate_ids']
                assert x['actual']['evidence']==y['actual']['evidence']
                pairs.append({'question_id':x['id'],'request_and_evidence_exact':True,
                    'answer_text_exact':x['actual']['answer']==y['actual']['answer'],
                    'before_pass':x['strict_original_contract_passed'],'after_pass':y['strict_original_contract_passed']})
            summary={'cell':cell,**info,'before_pid':a['pid'],'after_pid':b['pid'],
                'persistent_memory_and_learned_state_exact':True,'actual_saved_boundary_exact':True,
                'before_score':a['strict_score'],'after_score':b['strict_score'],'pairs':pairs}
            report['cells'].append(summary)
            trace.event('video_cold_memory.actual_restart_comparison',{'run_id':run.name,**summary},'orchestrator')
        assert report['native_calls']==24
        for asr in ('base','small'):
            source=ROOT/'local'/parent.name/asr
            assert sha((source/'hybrid.sqlite').read_bytes())==source_hashes[asr]['database_sha256']
            with read_only_db(source) as db:assert memory_state(db)==source_memory[asr]
            for entry in source_hashes[asr]['source_media']:
                path=Path(entry['path']);assert path.is_file()==entry['available']
                if entry['available']:assert sha(path.read_bytes())==entry['sha256']
        for rel,digest in hashes.items():assert sha((ROOT/rel).read_bytes())==digest
        native.check();report['verdict']='PRIVATE_COLD_MEMORY_PERSISTENCE_VERIFIED_SEMANTIC_SCORES_REPORTED_SEPARATELY'
    except Exception:
        report['failures'].append(traceback.format_exc());report['verdict']='COLD_VIDEO_DIAGNOSTIC_INCOMPLETE'
    finally:
        report['seconds']=time.monotonic()-started
        with (run/'result.json').open('x',encoding='utf-8') as f:f.write(encode(report))
        trace.event('video_cold_memory.completed',{k:v for k,v in report.items() if k!='cells'},'orchestrator');trace.close()
        print(encode(report),flush=True)
    return not report['failures']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',type=Path);p.add_argument('--cell');p.add_argument('--phase');a=p.parse_args()
    raise SystemExit(0 if (child_phase(a.child.resolve(),a.cell,a.phase) if a.child else main()) else 1)
