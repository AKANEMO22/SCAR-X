import json,sqlite3,sys
from pathlib import Path
import pytest
from scarx.evaluation.product_acceptance import _judge,_judge_v1,copy_session,canonical_reply
from scarx.neural.unified import UnifiedCore


@pytest.mark.parametrize('text,expect,prompt',[('nothello.','hello','hi'),('Hello world.','hello','hi'),('Tôi không phải SCAR-X.','scar-x','bạn là ai'),('Tôi là scar-x là.','scar-x','bạn là ai')])
def test_v2_cannot_pass_negated_extra_or_partialtoken_content_but_v1_stays_reproducible(text,expect,prompt):
    assert _judge_v1(text,expect,prompt)['correct'] is True
    row=_judge(text,expect,prompt);assert row['correct'] is False and row['metric_version']=='product-canonical-v2'


def test_sqlite_online_copy_includes_committed_wal_and_never_copies_session_lock(tmp_path):
    src=tmp_path/'source';src.mkdir();db=sqlite3.connect(src/'generation1.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE counter(n)');db.execute('INSERT INTO counter VALUES(1)');db.commit()
    (src/'session.lock').write_text('ownedbysource');copy=copy_session(src,destination_parent=tmp_path/'copies')
    assert copy.resolve().is_relative_to((tmp_path/'copies').resolve()) and not (copy/'session.lock').exists()
    db.execute('UPDATE counter SET n=2');db.commit()
    with sqlite3.connect(copy/'generation1.sqlite') as cloned:assert cloned.execute('SELECT n FROM counter').fetchone()[0]==1
    assert db.execute('SELECT n FROM counter').fetchone()[0]==2;db.close()


def test_canonical_probe_does_not_advance_source_neuronstate():
    m=UnifiedCore(['<eos>','<bos>','hello','.']);before=m.dumps();canonical_reply(m,'hi');assert m.dumps()==before


def test_guard_acceptance_is_not_newskill_acquisition():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from acceptance_suite import metres
    base={'summary':{'free_answer_correct':.5,'premature_stop':0.,'repetition':0.},'heldout_acc':0.,'correct_prompts':['hi'],'guard_passing':1,'guard_total':1,'rss_bytes':1,'vms_bytes':1,'metric_version':'product-canonical-v2'}
    out={'baseline':base,'taught':base,'newly_lost':[],'acquisition':[{'accepted':True,'baseline_target_complete':True,'newly_acquired':False},
        {'accepted':True,'baseline_target_complete':False,'newly_acquired':False,'target_complete_final':False}]}
    metrics=metres(out);assert metrics['5_guard_acceptance']==1. and metrics['3_learning_acquisition']==0. and metrics['3b_acquisition_denominator']==1
    out['acquisition'][1].update(baseline_target_content=True,newly_acquired=True,new_content_acquired=False,target_complete_final=True)
    metrics=metres(out);assert metrics['3_learning_acquisition']==1. and metrics['3d_content_acquisition'] is None and metrics['3e_content_acquisition_denominator']==0


def test_missing_ram_is_unknown_and_commit_breach_fails_even_with_low_rss():
    from types import SimpleNamespace
    from scarx.evaluation.product_acceptance import resources
    class Missing:
        def memory_info(self):raise RuntimeError('unavailable')
    assert resources(SimpleNamespace(process=Missing()))['under_limit'] is None
    class Commit:
        def memory_info(self):return SimpleNamespace(rss=100,private=2_100_000_000,vms=2_100_000_000)
    assert resources(SimpleNamespace(process=Commit()))['under_limit'] is False
