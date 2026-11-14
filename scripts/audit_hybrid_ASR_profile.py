"""Data-only review of four private native selected-ASR calls before activation."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.runtime.asr_profiles import PROFILES,verify_execution_log
from scarx.hybrid.store import encode,Store


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(2<<20):digest.update(chunk)
    return digest.hexdigest()


def main(run):
    run=run.resolve();assert run.is_relative_to(ROOT/'runs')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    result=json.loads((run/'private-result.json').read_text(encoding='utf-8'))
    assert result['gate_pass'] and not result['failures'] and len(result['calls'])==4
    assert result['private_neural_updates']==0 and result['private_cold_GPU_clone_and_rollback_exact']
    assert result['seconds']<=300
    for rel,expected in manifest['source_sha256'].items():assert sha(run/'source'/rel)==expected
    for filename,expected in manifest['candidate_source_sha256'].items():assert sha(run/'candidate-src'/filename)==expected
    private=ROOT/'local'/run.name
    database=(private/'hybrid.sqlite').resolve()
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
        events=[{'id':i,'kind':kind,'payload':json.loads(raw)} for i,kind,raw in
            db.execute("SELECT id,kind,payload FROM events WHERE kind IN ('worker_started','asr_result','checkpoint_restored') ORDER BY id")]
        checkpoints=db.execute('SELECT digest,payload FROM checkpoints').fetchall()
    workers=[e['payload'] for e in events if e['kind']=='worker_started']
    answers=[e['payload'] for e in events if e['kind']=='asr_result']
    assert len(workers)==len(answers)==4
    log=(private/'asr.log').read_text(encoding='utf-8',errors='replace')
    # Each CLI initialization names its source model. Split before each new
    # model init; followingprecision/backend evidence is independently checked.
    marker='whisper_init_from_file_with_params_no_state: loading model from '
    parts=log.split(marker)
    assert len(parts)==5,'Expected four actualmodel initialization records'
    calls=[]
    for index,(case,record,worker,answer) in enumerate(zip(manifest['cases'],result['calls'],workers,answers)):
        profile=PROFILES['openai/whisper-'+case['model']]
        assert record['model']==case['model'] and record['sample_id']==case['sample']['id']
        assert record['ASR_result']==answer
        assert answer['ASR_identity']==profile.identity()
        assert answer['actual_neural_device']=='CUDA0' and answer['execution_device_log_verified'] is True
        command=worker['command']
        argument=lambda flag:command[command.index(flag)+1]
        assert Path(argument('-m')).resolve()==(private/profile.relative_path).resolve()
        assert sha(Path(argument('-m')))==record['selected_native_sha256']==profile.native_sha256
        assert Path(argument('-f')).resolve()==Path(case['sample']['audio']).resolve()
        assert argument('-dev')=='0' and '-ng' not in command
        assert Path(command[0]).name=='whisper-cli.exe' and 'cuda12.4' in str(Path(command[0]).parent.parent)
        evidence=verify_execution_log(marker+parts[index+1],'gpu',0)
        native=json.loads((run/f"actual-call-{record['attempt']}.json").read_text(encoding='utf-8'))
        transcript=' '.join(r.get('text','').strip() for r in native['transcription']).strip()
        assert transcript==record['actual_transcript']==answer['text']
        calls.append({'model':case['model'],'sample_id':record['sample_id'],
            'actual_transcript':transcript,'native_worker_pid':worker['pid'],
            'profile_sha256_matches':True,'execution_evidence':evidence})
    payload_digests=set()
    for digest,payload in checkpoints:
        assert hashlib.sha256(payload).hexdigest()==digest;payload_digests.add(digest)
        with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
            meta=json.loads(arrays['metadata'].tobytes())
        assert meta['updates']==meta['hidden_updates']==0 and meta['replay']==[]
    assert len(payload_digests)==1 and checkpoints
    assert any(e['kind']=='checkpoint_restored' for e in events)
    assert result['global_free_samples'] and min(r['free_MiB'] for r in result['global_free_samples'])>=512
    report={'run_id':run.name,'verdict':'ACTUAL_SELECTED_ASR_PRIVATE_P0_DATA_AUDIT_PASS_ONLY',
        'actual_ASR_calls':4,'calls':calls,'commands_native_JSON_current_logs_and_pinnedfiles_match':True,
        'private_saved_payload_and_actual_rollback_verified':True,'all_private_checkpoint_payloads_exact':True,
        'private_neural_updates':0,'native_VLM_restarted_or_live_ASR_changed':False,
        'actual_runtime_activation_still_pending':True,'new_independent_quality_samples':0,
        'scope':'Data-only same-orchestrator audit, not outsidepeerreview, modelquality or fullproductionqualification'}
    (run/'private-events-audit-v1.json').write_text(encode(events),encoding='utf-8')
    with (run/'private-audit-v1.json').open('x',encoding='utf-8') as f:f.write(encode(report))
    trace=Store(ROOT/'local/hybrid-v1');trace.event('ASR_profile.private_independent_audit',
        {k:v for k,v in report.items() if k!='calls'},'orchestrator');trace.close()
    print(encode({k:v for k,v in report.items() if k!='calls'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);main(parser.parse_args().run)
