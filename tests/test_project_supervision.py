import json
import pytest
from scarx.app.generation import Generation
from scarx.app.learning_channel import post,feed
from scarx.learning.dialogue import DialogueLearning


def test_project_supervision_has_real_provenance_dedup_and_respects_personal_override(tmp_path):
    app=Generation(tmp_path,background=False)
    try:
        request={'kind':'project_lesson','client_id':'project-test-0001','family':1,'example':0}
        result=post(app,request);assert result['checkpoint_before'] and result['lesson_id']
        messages=feed(app)['messages'];supervisor=next(r for r in messages if r['id']=='project:project-test-0001')
        assert supervisor['role']=='supervisor' and supervisor['kind']=='project_lesson' and 'not a human message' in supervisor['source']
        state=app.model.dumps();assert post(app,request)['deduplicated'];assert app.model.dumps()==state
        with pytest.raises(ValueError):post(app,{**request,'family':2})
        app.dialogue.enqueue('human','hi','My preferred greeting.','explicit personal override')
        state=app._pack();skipped=post(app,{**request,'client_id':'project-test-0002'})
        assert skipped['skipped'] and app._pack()==state
        with pytest.raises(ValueError):post(app,{**request,'client_id':'project-test-0003','family':True})
    finally:app.close()


def test_identical_guard_rejections_stop_without_changing_weights_or_guard(tmp_path,monkeypatch):
    app=Generation(tmp_path,background=False)
    try:
        item=app.dialogue.enqueue('test','hello','Hello.','explicit testing fixture')
        state=app.model.dumps();calls=iter(([True],[False],[True],[False]))
        monkeypatch.setattr(app,'_guard_pairs',lambda _: [('previous skill','target')])
        monkeypatch.setattr(app,'_guard_scores',lambda *_:next(calls))
        first=app._online_microstep('test');assert first['identical_guard_rejections']==1 and app.model.dumps()==state
        app.dialogue=DialogueLearning.loads(app.dialogue.dumps())
        second=app._online_microstep('test')
        assert second['stopped_identical_proposal'] and second['identical_guard_rejections']==2
        assert app.model.dumps()==state and not any(second['weight_delta'].values())
        current=next(l for l in app.dialogue.lessons if l['id']==item['id'])
        assert current['remaining']==0 and current['status']=='needs_help' and app._online_microstep('test') is None
        assert app.dialogue.rejected_updates==2
    finally:app.close()
