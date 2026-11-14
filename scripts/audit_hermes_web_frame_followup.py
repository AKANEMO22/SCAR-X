"""Read-only audit of completed owned Hermes/source/review receipts; no inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.web_evidence import supplied_web_sources


def audit(run):
    run=run.resolve()
    if not run.is_relative_to(ROOT/'local/web-frame-live-followup'):
        raise ValueError('Expected an owned web-frame acceptance directory')
    result=json.loads((run/'result.json').read_text(encoding='utf-8'))
    snapshot=json.loads((run/'raw-session-snapshot.json').read_text(encoding='utf-8'))
    assert result['completed'] and snapshot['session_id']==result['session_id']
    calls=result['tool_calls'];assert len(calls)==1 and calls[0]['name']=='web_search'
    assert calls[0]['args']['query']=='jojo ra tập mới nhất là tập bn'
    call_id=calls[0]['tool_call_id']
    raw_results=result['tool_results'];assert len(raw_results)==1 and not raw_results[0]['is_error']
    raw=json.loads(raw_results[0]['result']);assert raw['success'] is True
    urls={row['url'] for row in raw['data']['web']}
    messages=snapshot['messages'];before=copy.deepcopy(messages)
    entries,coverage=supplied_web_sources(messages)
    assert messages==before and not coverage['partial'] and coverage['paired_results']==1
    assert {e['source'] for e in entries}==urls and len(entries)==5
    assert {e['tool_call_id'] for e in entries}=={call_id}
    provenance=result['bound_sources'];ids={p['evidence_id'] for p in provenance}
    assert len(ids)==5 and all(p['factual_verification'] is False for p in provenance)
    answers=result['bound_answers'];assert len(answers)==1
    answer=answers[0]
    assert answer['question']=='tìm kiếm đi' and answer['task_question']==calls[0]['args']['query']
    assert answer['task_interpretation']['expanded_question'].endswith('tập bao nhiêu')
    assert {e['id'] for e in answer['evidence']}==ids
    assert all(e['status']=='observed' and e['modality']=='web_search_snippet' for e in answer['evidence'])
    assert {e['source'] for e in answer['evidence']}==urls
    starts=[e for e in result['natural_review_events'] if e['kind']=='idle_review_started']
    completions=[e for e in result['natural_review_events'] if e['kind']=='idle_review_completed']
    assert len(starts)==len(completions)==1
    assert set(starts[0]['payload']['evidence_ids'])==ids
    assert starts[0]['payload']['task_question']==answer['task_question']
    review=completions[0]['payload'];assert review['weights_updated'] is False
    assert review['status']=='needs_verification' and review.get('no_learning_reason')
    return {'verdict':'ACTUAL_HERMES_FIVE_SOURCES_TASK_BINDING_AND_NATURAL_IDLE_REVIEW_OBSERVED',
        'canonical_Hermes_chat':True,'query':result['query'],'resolved_query':answer['task_question'],
        'actual_search_calls':1,'local_file_or_paid_extractor_calls':0,
        'returned_and_persisted_web_sources':5,'source_urls':sorted(urls),
        'source_status':'observed_not_fact_verified','coverage':coverage,
        'actual_message_wrapper_retained':messages==before,
        'task_context_not_replaced_by_bare_followup':True,'review_received_same_five_sources':True,
        'review_started_event_id':starts[0]['id'],'review_completed_event_id':completions[0]['id'],
        'native_answer_unverified':result['completion']['text'],
        'answer_quality':'UNRESOLVED: source snippet states a planned total; it does not establish current released episode index',
        'review':review,'review_quality':'NOT_INDEPENDENTLY_VERIFIED; reviewer still treats total count as episode evidence',
        'self_labelled_weight_update':False,'new_neural_rule':False,'new_calls_or_training_in_audit':0,
        'raw_receipts_sha256':{n:hashlib.sha256((run/n).read_bytes()).hexdigest()
                               for n in ('result.json','events.json','raw-session-snapshot.json')},
        'research_or_production_PASS':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path);args=parser.parse_args();report=audit(args.run)
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))
