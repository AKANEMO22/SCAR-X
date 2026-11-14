"""Data-only verification of saved video memory, process phases and native calls."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/20261114T070321Z-video-cold-memory-v1-7f5e62'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def encode(value):return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def main(output=None):
    manifest=json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
    report=json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    assert not report['failures'] and len(report['cells'])==4
    assert not report['new_learning'] and report['native_calls']==24
    for rel,expected in manifest['source_sha256'].items():
        assert sha((RUN/'source'/rel).read_bytes())==expected
    parent=ROOT/'runs'/manifest['parent_run']
    assert sha((parent/'result.json').read_bytes())==manifest['parent_result_sha256']
    expected=[{key:r[key] for key in ('id','question','expected')}
              for r in json.loads((parent/'result.json').read_text(encoding='utf-8'))['arms']['base']['answers']]
    assert manifest['questions']==expected
    questions={row['id']:row for row in expected}
    states={};memory={};sources={}
    for asr,original in manifest['original_owned_source_sha256'].items():
        directory=ROOT/'local'/parent.name/asr
        assert sha((directory/'hybrid.sqlite').read_bytes())==original['database_sha256']
        with sqlite3.connect((directory/'hybrid.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            evidence=[dict(r) for r in db.execute('SELECT * FROM evidence ORDER BY id')]
            streams=[tuple(r) for r in db.execute("SELECT id,payload FROM objects WHERE kind='stream' ORDER BY id")]
            # Hash in exact declared schema-column order, not dict sort order.
            columns=[r[1] for r in db.execute('PRAGMA table_info(evidence)')]
            rows=[tuple(r[k] for k in columns) for r in evidence]
            memory[asr]={'evidence_sha256':sha(encode(rows).encode()),'evidence_count':len(rows),
                          'stream_sha256':sha(encode(streams).encode()),'stream_count':len(streams)}
            assert memory[asr]==original['memory']
            sources[asr]={r['id']:r for r in evidence}
        for entry in original['source_media']:
            path=Path(entry['path'])
            assert path.resolve().is_relative_to(directory.resolve())
            assert path.is_file()==entry['available']
            if entry['available']:assert sha(path.read_bytes())==entry['sha256']
    actual_ids=[];requests={};records=[];process_ids=[]
    for cell,info in manifest['cells'].items():
        a,b=[json.loads((RUN/(cell+'-'+phase+'.json')).read_text(encoding='utf-8'))
             for phase in ('before','cold_restart')]
        assert not a['failures'] and not b['failures']
        assert a['pid']!=b['pid'] and a['process_created_epoch']<b['process_created_epoch']
        assert a['exit_checkpoint_payload_sha256']==b['entry_checkpoint_payload_sha256']
        assert a['entry_learned_state']==b['entry_learned_state']
        assert a['entry_memory']==a['exit_memory']==b['entry_memory']==b['exit_memory']==memory[info['ASR_source']]
        process_ids += [a['pid'],b['pid']]
        for phase,result in (('before',a),('cold_restart',b)):
            assert result['native_calls_completed']==len(result['native_calls'])==3
            assert result['VLM_worker_pid']==manifest['original_native_pid']
            assert result['VLM_worker_created']==manifest['original_native_created']
            checkpoint=(RUN/(cell+'-'+phase+'-saved-checkpoint.bin')).read_bytes()
            assert sha(checkpoint)==result['exit_checkpoint_payload_sha256']
            with np.load(__import__('io').BytesIO(checkpoint),allow_pickle=False) as z:
                meta=json.loads(z['metadata'].tobytes())
                assert meta['updates']==meta['hidden_updates']==meta['core_updates']==0
                assert meta['seed']==260113
                assert meta['rng']==json.loads(result['entry_learned_state']['rng'])
                assert meta['signal_rng']==json.loads(result['entry_learned_state']['signal_rng'])
                assert sha(encode(meta['replay']).encode())==result['entry_learned_state']['replay_sha256']
                for name,digest in result['entry_learned_state']['array_sha256'].items():
                    assert sha(z[name].tobytes())==digest
            strict=0
            for answer,call in zip(result['answers'],result['native_calls']):
                q=questions[answer['id']]
                assert answer['question']==q['question'] and answer['expected']==q['expected']
                assert sha(encode(call['request']).encode())==call['request_sha256']==answer['request_sha256']
                assert call['request']['max_tokens']==128
                assert len(call['request']['messages'])==2
                assert answer['actual']['answer']==(call['raw']['choices'][0]['message'].get('content') or '')
                correct=answer['actual']['answer'].strip().casefold()==q['expected']
                assert correct==answer['strict_original_contract_passed']
                strict+=correct
                assert len(answer['actual']['evidence'])<=4
                for item in answer['actual']['evidence']:
                    source=sources[info['ASR_source']][item['id']]
                    for key in ('start','end','source','status'):
                        assert item[key]==source[key]
                    assert item['text']==source['text'][:1100]
                    assert source['stream']==info['stream_id']
                record={'cell':cell,'phase':phase,'question_id':q['id'],
                    'actual_answer':answer['actual']['answer'],'strict_original_contract_passed':correct,
                    'request_sha256':call['request_sha256'],'evidence_ids':[e['id'] for e in answer['actual']['evidence']]}
                records.append(record)
                actual_ids.append(call['raw']['id'])
                requests.setdefault(call['request_sha256'],[]).append(record)
            assert strict==result['strict_score']==1
        for x,y in zip(a['answers'],b['answers']):
            assert x['request_sha256']==y['request_sha256']
            assert x['actual']['evidence']==y['actual']['evidence']
            assert x['actual']['answer']==y['actual']['answer']
    assert len(actual_ids)==len(set(actual_ids))==24
    assert all(len({r['actual_answer'] for r in group})==1 for group in requests.values())
    same_set_different_order=[]
    for asr in ('base','small'):
        for q in questions:
            snn=next(r for r in records if r['cell']==asr+'-SNN_frozen_existing' and r['phase']=='before' and r['question_id']==q)
            rag=next(r for r in records if r['cell']==asr+'-RAG_candidateorder' and r['phase']=='before' and r['question_id']==q)
            if snn['evidence_ids']!=rag['evidence_ids']:
                same_set_different_order.append({'ASR_source':asr,'question_id':q,
                    'same_evidence_membership':set(snn['evidence_ids'])==set(rag['evidence_ids']),
                    'SNN_actual_answer':snn['actual_answer'],'RAG_actual_answer':rag['actual_answer'],
                    'strict_primary_scores_unchanged':True,'not_a_new_semantic_rubric':True})
    result={'run_id':RUN.name,'native_calls_with_unique_server_response_ids':24,
        'unique_exact_request_bodies':len(requests),'private_process_phases':8,
        'verified_distinct_processes':len(set(process_ids)),
        'four_actual_saved_checkpoint_boundaries_exact':True,
        'source_evidence_stream_media_and_learned_state_hashes_match':True,
        'all12paired_prompt_evidence_and_answer_texts_exact_after_restart':True,
        'original_strict_score_each_cell_before_after':'1/3 and1/3',
        'original_strict_thresholds_or_rubric_changed':False,
        'raw_answers':records,'selector_order_contrasts':same_set_different_order,
        'source_owner_database_or_live_neural_state_changed':False,
        'scope':'One known21sclip/3questions,2ASRvariants;24actualcalls not independent samples or a lifelong/neural-memory test',
        'verdict':'RECORDED_COLD_PERSISTENCE_AND_STRICT_SCORES_VERIFIED_NO_SNN_ADVANTAGE_OR_QUALITY_PASS'}
    destination=output or RUN/'independent-data-audit-v1.json'
    assert destination.resolve().is_relative_to(ROOT)
    with destination.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(encode({k:v for k,v in result.items() if k not in ('raw_answers','selector_order_contrasts')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    main(p.parse_args().output)
