"""Read-only verification of real owned-project/context acceptance receipts."""

import hashlib
import json
from pathlib import Path
import psutil
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    failed=ROOT/'local/release-session-acceptance/20261114T141031038789Z/result.json'
    baseline=read(failed)
    assert baseline['turns'][1]['characters']==7028
    assert baseline['turns'][1]['completion']['status']=='error'
    assert 'Search query exceeds bounded 6000-character' in baseline['turns'][1]['completion']['text']
    project_run=ROOT/'local/release-session-acceptance/20261114T141928208486Z'
    project=read(project_run/'result.json');project_path=Path(project['project'])
    assert Path(project['create_info']['cwd']).resolve()==project_path.resolve()
    assert (project_path/'.git').is_dir()
    assert all(t['completion']['status']=='complete' and 'JADE-472' in t['completion']['text']
               and '9237' in t['completion']['text'] for t in project['turns'])
    project_events=read(project_run/'events.json')
    tools=[e['params']['payload'] for e in project_events if e.get('params',{}).get('type')=='tool.complete']
    assert len(tools)==1 and tools[0]['name']=='terminal' and not tools[0]['is_error']
    assert tools[0]['tool_call_id'].startswith('scarx_read_')
    context_run=ROOT/'local/release-context-acceptance/20261114T142255469500Z'
    context=read(context_run/'result.json');snapshot=read(context_run/'final-snapshot.json')
    assert context['transport_completed'] and not context['final_running'] and len(context['turns'])==5
    assert all(t['input_characters']==7115 and t['completion']['status']=='complete'
               and 'JADE-472' in t['completion']['text'] and '9237' in t['completion']['text'] for t in context['turns'])
    usage=[t['completion']['usage'] for t in context['turns']]
    assert all(u['context_max']==16384 for u in usage) and usage[-1]['compressions']==3
    manual=context['compression'];assert manual['operation']=='compress'
    assert manual['summary']['headline']=='Compressed: 8 → 6 messages'
    assert not any(manual['summary'][key] for key in ('noop','aborted','refused_would_grow'))
    markers=('[PRIOR CONTEXT','[CONTEXT COMPACTION','[END OF PRIOR CONTEXT')
    raw_handoffs=sum(any(m in json.dumps(row,ensure_ascii=False) for m in markers) for row in snapshot['messages'])
    shown_handoffs=sum(any(m in json.dumps(row,ensure_ascii=False) for m in markers) for row in snapshot['display_messages'])
    assert raw_handoffs==4 and shown_handoffs==0
    expected=[t['prompt'] for t in project['turns']]
    expected += [(('Unrelated archived note '+str(i)+': no project settings changed. ')*125
        +'\n\nAnswer briefly with the receipt code and current port already read from the project. Do not edit files or browse.') for i in range(5)]
    display_user=[row['text'] for row in snapshot['display_messages'] if row.get('role')=='user']
    assert display_user==expected
    restored=read(ROOT/'local/release-long-input-fix/activation-state.json')
    assert restored['learned_arrays_RNG_replay_preserved'] and restored['updates']==5
    with urlopen('http://127.0.0.1:8775/api/status?after=2147483647',timeout=10) as response:status=json.load(response)
    assert status['ready'] and not status['failure'] and status['neural']['updates']==5
    assert status['neural']['execution_device']=='gpu' and not psutil.pid_exists(46940)
    native=[p.pid for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower()=='llama-server.exe']
    assert len(native)==1
    report={'verdict':'OWNED_PROJECT_LONG_TURN_AND_CANONICAL_COMPACTION_VERIFIED_NOT_NEURAL_QUALITY_PASS',
        'baseline_long_turn_HTTP_format_error':True,'accepted_long_turn_characters':7028,
        'actual_project_read_calls':1,'project_read_selector':'runtime readonly prelude; VLM produced the answer',
        'correct_project_purpose_receipt_and_port':True,
        'additional_real_turns':5,'characters_per_additional_turn':7115,
        'automatic_compressions':3,'manual_compression':'8 -> 6 messages',
        'working_context_tokens':16384,'max_observed_context_used':max(u['context_used'] for u in usage),
        'raw_handoff_rows':raw_handoffs,'display_handoff_rows':shown_handoffs,
        'all_seven_original_fixture_user_texts_retained_in_display':True,
        'desktop_owner_projection_verified':True,'visual_Desktop_capture_in_this_check':False,
        'gateway_pid':status['pid'],'one_native_GPU_pid':native[0],
        'private_learned_state_preserved_at_restart':True,'live_updates':5,
        'new_learning_rule_training_dataset_or_8h_run':False,
        'native_inference_was_executed_by_acceptance':True,'additional_native_calls_in_this_readonly_audit':0,
        'receipt_hashes':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in {
            'baseline':failed,'workspace':project_run/'result.json','context':context_run/'result.json',
            'projection':context_run/'final-snapshot.json'}.items()},
        'harness_errors_preserved':['invalid surface=desktop; normal desktop omits it',
            'constant titled create resumed earlier owned session; use unique title/request ID',
            'legacy session.compress RPC refused; Desktop uses revision-fenced session.mutate'],
        'not_proven':['unseen knowledge retention','SNN advantage over same LLM+RAG','learned tool-error causality','general long-work quality','all visual Desktop and screen workflows']}
    destination=ROOT/'docs/release/evidence/LONG_INPUT_AND_COMPACTION_20261114.json'
    with destination.open('x',encoding='utf-8',newline='\n') as out:json.dump(report,out,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
