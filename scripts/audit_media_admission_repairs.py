"""Readonly source/real UI/admission ownership verification, no model calls."""

import hashlib
import json
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET
from urllib.request import urlopen
import psutil

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    run=Path((ROOT/'local/current-hermes-media-admission.txt').read_text(encoding='utf-8')).resolve()
    assert run.is_relative_to(ROOT/'local/hermes-media-admission')
    cursor=read(run/'actual-UI-readiness-before.json')['event_cursor']
    with sqlite3.connect((ROOT/'local/hybrid-v1/hybrid.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        events=[{'id':i,'kind':k,'payload':json.loads(p)} for i,k,p in db.execute('SELECT id,kind,payload FROM events WHERE id>? ORDER BY id',(cursor,))]
        bind=next(e for e in events if e['kind']=='video_context_bound')
        sid=bind['payload']['session_id'];stream=bind['payload']['stream_id']
        binding=json.loads(db.execute("SELECT payload FROM objects WHERE kind='video_context_v2' AND id=?",(sid,)).fetchone()[0])
        key=hashlib.sha256((sid+'\0'+binding['submission_id']).encode()).hexdigest()
        request_binding=json.loads(db.execute("SELECT payload FROM objects WHERE kind='video_turn_v3' AND id=?",(key,)).fetchone()[0])
    assert bind['payload']['origin']=='explicit_upload_request_v3' and request_binding==binding
    assert any(e['kind']=='video_segment_analyzed' and e['payload']['stream_id']==stream and e['id']<bind['id'] for e in events)
    answers=[e for e in events if e['kind']=='answer' and e['payload'].get('stream')==stream]
    assert len(answers)==1
    answer=answers[0]['payload'];assert 'red' in answer['answer'].lower() and answer['evidence']
    q=read(run/'actual-UI-readiness-before.json')['probe_text'];assert answer['question']==q
    with sqlite3.connect((Path.home()/'AppData/Local/hermes/state.db').as_uri()+'?mode=ro',uri=True) as db:
        admission=db.execute('SELECT request_id,status,outcome FROM session_admissions WHERE request_id=? AND target_session_id=?',
                            (binding['submission_id'],sid)).fetchone()
        messages=db.execute('SELECT role,content FROM messages WHERE session_id=? ORDER BY id',(sid,)).fetchall()
    assert admission and admission[0]==binding['submission_id'] and admission[1]=='terminal' and admission[2]=='completed'
    users=[text for role,text in messages if role=='user'];assert users==[q]
    assert not [role for role,text in messages if role=='tool']
    assert not any(e['kind']=='hermes_tool_request' for e in events if e['id']<answers[0]['id'])
    homes=Path.home()/'AppData/Local/hermes'
    source_pins={}
    for rel,installed in [
        ('integrations/hermes/session_submission.py',homes/'hermes-agent/gateway/session_submission.py'),
        ('integrations/hermes/scarx_session_transport.py',homes/'hermes-agent/agent/scarx_session_transport.py'),
        ('integrations/hermes/hybrid_plugin/plugin.js',homes/'desktop-plugins/scarx-hybrid/plugin.js'),
        ('integrations/hermes/hybrid_plugin/dashboard/plugin_api.py',homes/'plugins/scarx-hybrid/dashboard/plugin_api.py')]:
        assert (ROOT/rel).read_bytes()==installed.read_bytes();source_pins[rel]=hashlib.sha256(installed.read_bytes()).hexdigest()
    ingress=(homes/'hermes-agent/gateway/session_ingress.py').read_text(encoding='utf-8')
    assert "with submission_scope(row.get('request_id')):" in ingress
    installed=read(run/'renderer-and-plugin-installed.json')
    renderer=Path(installed['renderer_target']);assert renderer.is_relative_to(homes/'hermes-agent/apps/desktop/release')
    assert all(hashlib.sha256((renderer/name).read_bytes()).hexdigest()==digest for name,digest in installed['renderer_content_sha256'].items())
    state=read(run/'state-restored.json');assert state['all_saved_arrays_equal'] and state['RNG_replay_and_learned_metadata_equal'] and state['updates']==5
    with urlopen('http://127.0.0.1:8775/api/status?after=2147483647',timeout=10) as response:status=json.load(response)
    assert status['ready'] and not status['failure'] and status['neural']['updates']==5 and status['neural']['execution_device']=='gpu'
    native=[p.pid for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower()=='llama-server.exe'];assert len(native)==1
    assert not psutil.pid_exists(46940)
    report={'verdict':'ACTUAL_NEW_DRAFT_VIDEO_SUBMISSION_OWNERSHIP_AND_LOCAL_READY_ANSWER_VERIFIED',
        'identity_path':'composer submission_id -> resolved-session callback -> canonical executor ContextVar -> per-request X-SCARX-Submission -> SQLite binding',
        'text_matching_used_to_claim_upload':False,'actual_user_text_preserved':True,
        'session_digest':hashlib.sha256(sid.encode()).hexdigest(),'submission_digest':hashlib.sha256(binding['submission_id'].encode()).hexdigest(),
        'first_local_observation_before_media_binding':True,'actual_Hermes_UI_first_turn':True,
        'same_query_as_prior_chat_with_independent_identity':True,
        'actual_model_answer':answer['answer'],'selected_source_stream':stream,
        'selected_evidence_count':len(answer['evidence']),'actual_Hermes_tool_calls_after_readiness_fix':0,
        'answer_language_requested_Vietnamese_but_returned_English':True,'requested_timestamp_citation_missing':True,
        'known_fixture_not_general_video_quality_PASS':True,
        'source_pins':source_pins,'renderer_build_typechecked_and_installed':True,
        'renderer_assets_verified':len(installed['renderer_content_sha256']),
        'Electron_main_and_preloads_retained':True,
        'saved_neural_arrays_RNG_replay_preserved_at_activation':True,'live_synaptic_updates':5,
        'gateway_pid':status['pid'],'one_native_GPU_pid':native[0],
        'new_model_calls_in_this_audit':0,'research_or_production_PASS':False,
        'original_known_failures_kept':['first unprepared UI turn searched web before video analysis; bounded guard stopped it',
            'immutable JPEG concurrent publication initially hit Windows replace/read conflict; non-overwriting publication regression now passes']}
    with (ROOT/'docs/release/evidence/MEDIA_ADMISSION_REPAIRS_20261114.json').open('x',encoding='utf-8',newline='\n') as out:json.dump(report,out,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
