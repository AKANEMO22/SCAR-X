import json
import numpy as np
import pytest
from scarx.app.generation import Generation,LearningGuardRejected
from scarx.app.observatory import feed,observe_answer,answers_case
from scarx.app.observatory import archived_single_prompt_references
from scarx.runtime.work_trace import append,read
from scarx.runtime.work_trace import activity
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_real_inference_reference_labels_do_not_enter_model_or_learn(tmp_path):
    app=Generation(tmp_path,background=False)
    try:
        case,_=answers_case(ROOT,'b1-09:0');updates=app.model.core.updates;lessons=app.dialogue.dumps();weights=app.model.core.w_out.copy()
        observe_answer(app,ROOT,{'case_id':case['id']})
        actual=feed(app,tmp_path)['answers'][0]
        assert actual['target']==case['target'] and actual['qwen']['run'].startswith('runs/')
        assert actual['prompt']==case['messages'][-1]['text'] and actual['neural_ticks']>0
        assert actual['scope']=='Lõi đang dùng' and actual['generated'] is not None
        assert app.model.core.updates==updates and app.dialogue.dumps()==lessons
        np.testing.assert_array_equal(weights,app.model.core.w_out)
        before=app._pack();feed(app,tmp_path);feed(app,tmp_path);assert app._pack()==before
        with pytest.raises(ValueError):observe_answer(app,ROOT,{'case_id':'../../secret'})
    finally:app.close()
    restored=Generation(tmp_path,background=False)
    try:assert feed(restored,tmp_path)['answers'][0]['generated']==actual['generated']
    finally:restored.close()


def test_measured_loss_weight_delta_and_full_guard_rollback(tmp_path,monkeypatch):
    app=Generation(tmp_path,background=False)
    try:
        before=app.model.core.w_out.copy();updates=app.model.core.updates
        result=app.teach({'prompt':'chào bạn','answer':'Chào bạn.','repeats':2,'confirmation':True})
        metrics=result['learning_metrics']
        assert len(metrics['losses'])==2 and np.isfinite(metrics['loss_mean']) and metrics['loss_mean']>0
        assert metrics['update_steps']==app.model.core.updates-updates
        assert metrics['changed_synapses']['w_out']==np.count_nonzero(app.model.core.w_out!=before)
        assert metrics['weight_delta']['w_out']==pytest.approx(np.linalg.norm(app.model.core.w_out-before))
        assert not metrics['guard_checked'] # explicit form follows existing policy
        app.dialogue.enqueue('test','7 + 8','15','explicit unit-test lesson')
        app.dialogue.lessons[-1]['remaining']=1
        state=app.model.dumps();values=iter(([True],[False]))
        # Exercise rejection plumbing with real numerical learning; inject only guard verdict.
        monkeypatch.setattr(app,'_guard_pairs',lambda _: [('old skill','target')])
        monkeypatch.setattr(app,'_guard_scores',lambda *_: next(values))
        record=app._online_microstep('test')
        assert record['rolled_back'] and not record['accepted'] and record['loss_mean']>0
        assert record['attempted_weight_delta']['w_out']>0 and not any(record['weight_delta'].values())
        assert app.model.dumps()==state
        logged=feed(app,tmp_path)['learning'][0]
        assert logged['action']=='online-reject' and logged['guard_lost']==1
    finally:app.close()


def test_registered_runner_trace_bounded_and_cross_process_readable(tmp_path):
    run=tmp_path/'runs'/'registered';run.mkdir(parents=True)
    for i in range(607):append(run,'answer',{'generated':str(i),'scope':'isolated actual runner'})
    records=read(tmp_path);assert len(records)==200 and records[0]['generated']=='606'
    with pytest.raises(ValueError):append(tmp_path/'arbitrary','answer',{})
    with pytest.raises(ValueError):append(run,'answer',{'generated':'x'*33000})


def test_technical_work_stays_separate_and_poll_does_not_execute_or_mutate(tmp_path):
    run=tmp_path/'runs'/'work-check';run.mkdir(parents=True)
    activity(run,'Actual check','Check has finished.',evidence={'exit_code':0})
    append(run,'answer',{'prompt':'x','generated':'actual output','time':-1,'kind':'activity','run_id':'spoof'})
    app=Generation(tmp_path/'session',background=False)
    try:
        before=app._pack();first=feed(app,tmp_path);second=feed(app,tmp_path)
        assert app._pack()==before
        assert first['activities']==second['activities'] and len(first['activities'])==1
        assert first['activities'][0]['evidence']['exit_code']==0
        assert first['answers'][0]['run_id']=='work-check' and first['answers'][0]['time']>0
        assert not first['learning']
        with pytest.raises(ValueError):activity(run,'not measured','x',status='pretend')
    finally:app.close()


def test_saved_single_prompt_reference_is_matched_by_exact_text_not_case_id(tmp_path):
    directory=tmp_path/'artifacts';directory.mkdir()
    path=directory/'qwen-g1-state-reference-latest.json'
    path.write_text(json.dumps({'run':'runs/reference-one','outputs':[{'id':'same-id','prompt':'exact public text','generated':'actual saved reference','finish_reason':'length','measured_at_utc':'2026-11-09T00:00:00Z'}]}),encoding='utf-8')
    references=archived_single_prompt_references(tmp_path)
    assert references['exact public text']['finish_reason']=='length' and 'same-id' not in references and 'exact public text?' not in references
    path.write_text(json.dumps({'run':'runs/../../outside','outputs':[]}),encoding='utf-8');assert not archived_single_prompt_references(tmp_path)
    extra=directory/'qwen-word-control-reference-latest.json'
    extra.write_text(json.dumps({'run':'runs/reference-two','outputs':[{'prompt':'new public text','generated':'measured secondreference','finish_reason':'stop'}]}),encoding='utf-8')
    references=archived_single_prompt_references(tmp_path);assert references['new public text']['run']=='runs/reference-two' and 'exact public text' not in references


def test_reviewed_new_public_prompt_runs_real_model_and_adds_matching_archived_qwen_without_learning(tmp_path):
    app=Generation(tmp_path,background=False)
    try:
        state=app.dialogue.dumps();updates=app.model.core.updates;fingerprint=app.model.core.fingerprint()
        result=observe_answer(app,ROOT,{'dataset':'g1_state','case_id':'greeting_vi_01'})
        row=next(r for r in feed(app,ROOT)['answers'] if r.get('case_id')=='g1-state/greeting_vi_01')
        assert row['generated']==result['text'] and row['target']=='Chào bạn.' and row['qwen']['run'].startswith('runs/')
        assert app.model.core.updates==updates and app.dialogue.dumps()==state and app.model.core.fingerprint()==fingerprint
        with pytest.raises(ValueError):observe_answer(app,ROOT,{'dataset':'arbitrary-file','case_id':'x'})
        with pytest.raises(ValueError):observe_answer(app,ROOT,{'dataset':'g1_state','case_id':'../../outside'})
    finally:app.close()
