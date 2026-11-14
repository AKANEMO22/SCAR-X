"""Read-only audit of actual selected-window capture and cold retrieval receipts.

No model/capture calls; reports source retrieval separately from synaptic skill.
"""

import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.request import urlopen
import psutil

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    before=ROOT/'local/screen-window-acceptance/20261114T144620443796Z'
    after=ROOT/'local/screen-window-acceptance/20261114T145845777979Z'
    a,b=read(before/'result.json'),read(after/'result.json')
    assert a['capture_completed'] and a['post_stop_selected_stream'] is None
    assert a['final_stream']['session_id'] is None and a['final_stream']['status']=='stopped'
    assert b['capture_completed'] and b['owned_player_terminal']
    stream=b['stream_id'];session=b['fixture_session']
    assert b['final_stream']['session_id']==session and b['post_stop_selected_stream']==stream
    assert b['ungranted_request_HTTP_status']==400
    assert 'recent explicit user request' in json.dumps(b['ungranted_request_denial'])
    assert b['final_stream']['status']=='stopped' and not b['final_stream']['error']
    assert b['final_stream']['system_audio_requested'] is False and b['final_stream']['microphone_used'] is False
    assert b['final_stream']['asr_frontier']==0 and not b['final_stream']['audio_available']
    assert b['post_stop_evidence_count']==1 and 'đỏ' in b['post_stop_answer'].casefold()
    assert len(b['sampled_frame_evidence'])==2
    first,last=b['sampled_frame_evidence']
    assert 'red' in first['text'].casefold() and 'green' in last['text'].casefold()
    archive=ROOT/'local/hybrid-v1/archive'/stream
    rows=[json.loads(line) for line in (archive/'capture.jsonl').read_text(encoding='utf-8').splitlines()]
    starts=[r for r in rows if r['kind']=='started']
    assert len(starts)==1 and starts[0]['hwnd']==b['selected_window']['id']
    assert not starts[0]['system_audio_requested'] and not starts[0]['microphone_used']
    assert any(r['kind']=='stopped' for r in rows) and not any(r['kind'].startswith('audio') for r in rows)
    frames=[r for r in rows if r['kind']=='frame'];assert len(frames)>=2
    for row in frames:
        path=(archive/row['path']).resolve();assert path.is_relative_to(archive.resolve()) and path.is_file()
    for entry in b['sampled_frame_evidence']:
        source=Path(entry['source']).resolve();assert source.is_relative_to(archive.resolve()) and source.is_file()
    unrelated=read(after/'unrelated-turn-proof.json');cold=read(after/'cold-restart-followup.json')
    assert unrelated['no_old_screen_selected'] and unrelated['no_screen_evidence_injected'] and not unrelated['residual_capture_workers']
    assert cold['selected_stream']==stream and cold['selected_evidence_count']==1 and 'đỏ' in cold['answer'].casefold()
    assert cold['historical_question_does_not_authorize_recapture'] and not cold['residual_capture_workers']
    assert cold['learned_weight_trace_importance_RNG_replay_equal'] and cold['live_updates']==5
    assert cold['presnapshot_to_final_save_tick_difference']==16 and cold['full_saved_payload_equality_claimed'] is False
    with sqlite3.connect((ROOT/'local/hybrid-v1/hybrid.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        binding=json.loads(db.execute("SELECT payload FROM objects WHERE kind='screen_context_v1' AND id=?",(session,)).fetchone()[0])
        actual=json.loads(db.execute("SELECT payload FROM objects WHERE kind='stream' AND id=?",(stream,)).fetchone()[0])
    assert binding['stream_id']==stream and binding['session_id']==session and binding['origin']=='explicit_window_selection_v1'
    assert actual['status']=='stopped' and actual['session_id']==session
    with urlopen('http://127.0.0.1:8775/api/status?after=2147483647',timeout=10) as response:status=json.load(response)
    owner=psutil.Process(status['pid'])
    assert status['ready'] and not status['failure'] and status['neural']['updates']==5
    assert not any('hybrid_screen_worker.py' in p.cmdline() for p in owner.children(recursive=True))
    assert not psutil.pid_exists(46940)
    native=[p.pid for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower()=='llama-server.exe'];assert len(native)==1
    player=b['owned_player']
    try:p=psutil.Process(player['pid']);assert abs(p.create_time()-player['created'])>=.01
    except psutil.NoSuchProcess:pass
    report={'verdict':'ACTUAL_SELECTED_WINDOW_AND_COLD_SCREEN_EVIDENCE_RETRIEVAL_VERIFIED',
        'capture_surface':'one uniquely named script-owned FFplay window, silent known fixture',
        'capture_start_path':'installed Hermes scarx_video dispatcher after a session-scoped chat request',
        'baseline_capture_worked_but_post_stop_binding_missing':True,
        'captured_frames':len(frames),'VLM_sampled_frame_descriptions':2,
        'capture_frontier_seconds':b['final_stream']['capture_frontier'],
        'vision_frontier_seconds':b['final_stream']['vision_frontier'],
        'timebase':'seconds since sharing began, not original video playback position',
        'system_audio_or_microphone_used':False,
        'ungranted_request_denied':True,'ungranted_HTTP_status':b['ungranted_request_HTTP_status'],
        'post_stop_answer':b['post_stop_answer'],'cold_restart_answer':cold['answer'],
        'source_selected_after_stop_and_restart':True,'source_frame_retained_and_used':True,
        'unrelated_turn_does_not_inject_old_screen':True,'historical_question_does_not_grant_recapture':True,
        'owned_player_and_capture_workers_terminal':True,
        'learned_weights_traces_importance_RNG_replay_preserved':True,
        'transient_neuron_state_may_advance_before_final_save':cold['presnapshot_to_final_save_transient_neuron_array_differences'],
        'transient_tick_delta_before_final_save':16,'full_snapshot_byte_equality_claimed':False,
        'gateway_pid':status['pid'],'one_native_GPU_pid':native[0],'live_updates':5,
        'additional_model_or_capture_calls_in_this_audit':0,
        'raw_receipt_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in {
            'before':before/'result.json','after':after/'result.json','cold':after/'cold-restart-followup.json',
            'capture_journal':archive/'capture.jsonl','unrelated':after/'unrelated-turn-proof.json'}.items()},
        'not_proven':['general screen/video QA quality','speaker-loopback live acceptance','fully neural memory or skill learning','SNN superiority over same LLM+RAG','all visual Hermes workflows'],
        'research_or_production_PASS':False,'eight_hour_run_restarted':False}
    with (ROOT/'docs/release/evidence/SCREEN_HISTORY_20261114.json').open('x',encoding='utf-8',newline='\n') as out:json.dump(report,out,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
