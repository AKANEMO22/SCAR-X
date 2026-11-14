"""Read-only engineering evidence for the installed local RC; no model calls.

Only explicitly owned smoke questions/clip are exported. No private prompt,
checkpoint, token, unrelated conversation or process command line is published.
"""

import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
from urllib.request import urlopen
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.shared_native import SharedNative


def main():
    private=ROOT/'local/release-activation/20261114T130846509562Z'
    activation=json.loads((private/'result.json').read_text())
    cancellation=ROOT/'runs/20261114T095534Z-small-mixed-endurance-v1-32cd13/user-cancellation-audit-v1.json'
    cancelled=json.loads(cancellation.read_bytes())
    assert cancelled['verdict']=='USER_CANCELLED_NOT_EIGHT_HOUR_PASS'
    assert not psutil.pid_exists(cancelled['pid'])
    assert not cancelled['residual_private_worker_pids']
    with np.load(io.BytesIO((private/'private-before.npz').read_bytes()),allow_pickle=False) as old,np.load(io.BytesIO((private/'private-after.npz').read_bytes()),allow_pickle=False) as new:
        assert set(old.files)==set(new.files)
        assert all(np.array_equal(old[key],new[key]) and old[key].dtype==new[key].dtype for key in old.files if key!='metadata')
        a=json.loads(old['metadata'].tobytes());b=json.loads(new['metadata'].tobytes())
        assert {k:v for k,v in a.items() if k!='controller'}=={k:v for k,v in b.items() if k!='controller'}
    native=SharedNative(ROOT);status=native.status()
    assert status['ready'] and not status.get('failure')
    assert status['neural']['execution_device']=='gpu' and status['neural']['updates']==b['updates']==5
    assert all(status['execution_placement'][key]=='gpu' for key in ('VLM_vision','ASR','ALIF_forward_learning'))
    candidates=[p.pid for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower()=='llama-server.exe']
    assert candidates==[native.worker.pid]
    with sqlite3.connect((ROOT/'local/hybrid-v1/hybrid.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        events=[{'id':i,'kind':k,'payload':json.loads(p)} for i,k,p in db.execute('SELECT id,kind,payload FROM events WHERE id>? ORDER BY id',(activation['startup_event_cursor'],))]
        stream=json.loads(db.execute("SELECT payload FROM objects WHERE kind='stream' AND id=?",('66f152c835ef4ab6bb7ad35948499a87',)).fetchone()[0])
    parity=[e for e in events if e['kind']=='native_parity_case']
    assert len(parity)==3 and all(e['payload']['passed'] for e in parity)
    asr=[e for e in events if e['kind']=='asr_result']
    assert len(asr)==3 and all(e['payload']['precision']=='FP32' and e['payload']['actual_neural_device']=='CUDA0'
                             and e['payload']['ASR_identity']['name']=='openai/whisper-small'
                             and e['payload']['execution_device_log_verified'] for e in asr)
    assert stream['status']=='complete' and all(stream[key]==21 for key in ('capture_frontier','asr_frontier','vision_frontier'))
    q='Trong clip đính kèm, hình đầu tiên có màu gì? Dẫn mốc thời gian nếu có.'
    bound=[e for e in events if e['kind']=='video_context_bound' and e['payload']['stream_id']==stream['id']]
    assert bound
    videos=[e for e in events if e['kind']=='answer' and e['id']>bound[-1]['id'] and e['payload'].get('question')==q and 'đỏ' in e['payload'].get('answer','')]
    assert videos and videos[-1]['payload'].get('evidence')
    greeting='Chào bạn! Trả lời ngắn bằng tiếng Việt nhé.'
    greetings=[e for e in events if e['kind']=='answer' and e['payload'].get('question')==greeting and e['payload'].get('answer')=='Bạn có muốn tôi giúp gì không?']
    assert greetings
    prepared=next(e for e in reversed(events) if e['kind']=='hermes_request_prepared' and e['id']<greetings[-1]['id'])
    assert prepared['payload']['tools']==0 and prepared['payload']['workspace_marker_present']
    web_raw=(private/'actual-web-search.json').read_bytes();web=json.loads(web_raw)
    assert not web.get('error') and any(row.get('url')=='https://docs.searxng.org/' for row in web['data']['web'])
    h=Path.home()/'AppData/Local/hermes'
    helper=ROOT/'integrations/hermes/scarx_session_transport.py'
    assert helper.read_bytes()==(h/'hermes-agent/agent/scarx_session_transport.py').read_bytes()
    assert 'return bind_transcript_session(agent, kwargs)' in (h/'hermes-agent/agent/chat_completion_helpers.py').read_text(encoding='utf-8')
    # Public source installation must match the two already audited candidates.
    targets={'store.py':'20261114T101526Z-full-query-retrieval-p0-v1-7a9ee0/candidate-store.py',
             'engine.py':'20261114T105211Z-verifier-capability-p0-v1-8e205a/candidate-engine.py'}
    for name,reference in targets.items():assert (ROOT/'src/scarx/hybrid'/name).read_bytes()==(ROOT/'runs'/reference).read_bytes()
    report={'verdict':'INSTALLED_RC_ENGINEERING_SMOKE_VERIFIED_WITH_CAPABILITY_LIMITS',
            'gateway_pid':native.owner.pid,'one_native_GPU_worker_pid':native.worker.pid,
            'startup_native_parity_passed':3,'original_precision_selected_ASR_calls':3,
            'ASR_CUDA0_actual_log_guard_passed':True,'ALIF_device':'gpu',
            'learned_arrays_RNG_replay_preserved_at_activation':True,'live_updates_preserved':5,
            'cancelled_8h':{'elapsed_seconds':cancelled['elapsed_seconds'],'cycles':cancelled['completed_video_cycles'],'checkpoints':cancelled['checkpoints'],'qualified':False},
            'real_Hermes_smoke':{'greeting_event':greetings[-1]['id'],'greeting_answer':greetings[-1]['payload']['answer'],
                'video_bound_event':bound[-1]['id'],'video_answer_event':videos[-1]['id'],
                'video_answer':videos[-1]['payload']['answer'],'video_seconds':21,'capture_ASR_vision_frontiers':21,
                'same_known_owned_fixture_not_unseen_quality_evaluation':True,
                'web_search_real_dispatch_found_official_docs':True,'web_result_sha256':hashlib.sha256(web_raw).hexdigest()},
            'Hermes_session_helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),
            'installed_P0_candidates_exact':True,'new_architecture_or_learning_rule':False,
            'new_model_or_ASR_calls_in_this_audit':0,
            'not_established':['unseen_video_quality','SNN_advantage_over_same_LLM_RAG','long_term_skill_retention','learned_tool_error_causality','8h_on_current_profile','full_real_UI_compaction_and_screen_acceptance'],
            'research_or_production_PASS':False}
    destination=ROOT/'docs/release/evidence/RC3_OPERATIONAL_SMOKE_20261114.json'
    with destination.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
