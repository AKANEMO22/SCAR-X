from pathlib import Path
from decimal import Decimal
import pytest
from scarx.evaluation.knowledge_factory import CORE,UNIFIED,SERVER,APP,extract_facts,numeric_value,verify_candidate,verify_pool,REQUIRED_FIELDS


@pytest.fixture
def facts():
    root=Path(__file__).resolve().parents[1]
    return extract_facts({p:(root/p).read_text(encoding='utf-8') for p in (CORE,UNIFIED,SERVER,APP)})


def example(fact,variant=1):
    return {'fact_id':fact['id'],'variant':variant,'language':'vi' if variant%2 else 'en','prompt':f'Giá trị mặc định của `{fact["field"]}` trong Config là bao nhiêu?','canonical_answer':str(fact['value'])+'.'}


def test_ast_oracles_distinguish_Config_defaults_from_initial_G1(facts):
    f={r['id']:r for r in facts}
    assert len(facts)==40 and sum(r['known_domain'] for r in facts)==32
    assert f['config_hidden']['value']==64 and f['g1_hidden']['value']==48
    assert f['config_feature_decay']['value']==.92 and f['g1_feature_decay']['value']==.25
    assert all(r['source_hash'] and r['source_span']['text'] for r in facts)


@pytest.mark.parametrize('raw,expected',[('0.025.',Decimal('.025')),('0.002.',Decimal('.002')),('0,95.',Decimal('.95')),('2000000000.',Decimal(2000000000))])
def test_decimal_literals_are_not_misread_as_thousands(raw,expected):assert numeric_value(raw)==expected


def test_candidate_metadata_cannot_claim_verification_or_swap_source_value(facts):
    fact=next(r for r in facts if r['id']=='config_hidden');row=example(fact)
    assert verify_candidate(row,fact) is None
    assert verify_candidate({**row,'canonical_answer':'48.'},fact)=='WRONG_SOURCE_VALUE'
    assert verify_candidate({**row,'verified':True},fact)=='SCHEMA'
    assert verify_candidate({**row,'prompt':'Giá trị khởi tạo của `hidden` trong UnifiedCore là bao nhiêu?'},fact)=='UNVERIFIED_CONFIG_SCOPE'


def test_question_alignment_leakage_and_invalid_Unicode_fail_closed(facts):
    fact=next(r for r in facts if r['id']=='config_hidden');row=example(fact)
    assert verify_candidate({**row,'prompt':row['prompt']+'64?'},fact)=='VALUE_LEAK_OR_NEGATION'
    assert verify_candidate({**row,'prompt':row['prompt']+'\ud800'},fact)=='INVALID_UNICODE'
    assert verify_candidate({**row,'prompt':row['prompt'].replace('`hidden`','`outputs`')},fact)=='UNVERIFIED_PARAMETER_ALIGNMENT'


@pytest.mark.parametrize('prompt',[
    'Indicate the default rate specified for `leak` in Config.',
    'Browsing Config settings, what default is indicated for `leak`?',
    'What is the default magnitude for `leak` in Config?',
    'Trong tài liệu Config, tham số mặc định của `leak` được ấn định là gì?',
])
def test_literal_value_queries_allow_syntactic_variation_without_changing_the_oracle(facts,prompt):
    fact=next(r for r in facts if r['id']=='config_leak')
    row=example(fact,2 if prompt.isascii() else 1)
    assert verify_candidate({**row,'prompt':prompt},fact) is None


@pytest.mark.parametrize('prompt,reason',[
    ('What is twice the default value of `leak` in Config?','DERIVED_VALUE_NOT_SOURCE_LITERAL'),
    ('Give the default value of `leak` in Config as a percentage.','DERIVED_VALUE_NOT_SOURCE_LITERAL'),
    ('What is the purpose of the default `leak` value in Config?','DEFINITION_NOT_SOURCE_VALUE_QUERY'),
    ('What unit does the default `leak` value in Config use?','DEFINITION_NOT_SOURCE_VALUE_QUERY'),
    ('Hãy nêu tác dụng của giá trị mặc định `leak` trong Config.','DEFINITION_NOT_SOURCE_VALUE_QUERY'),
    ('Hãy tăng giá trị mặc định của `leak` trong Config.','DERIVED_VALUE_NOT_SOURCE_LITERAL'),
])
def test_numeric_source_oracle_must_not_label_explanations_or_derived_questions(facts,prompt,reason):
    fact=next(r for r in facts if r['id']=='config_leak');row=example(fact,2 if prompt.isascii() else 1)
    assert verify_candidate({**row,'prompt':prompt},fact)==reason


def test_bound_query_requires_source_field_and_maximum_qualification(facts):
    fact=next(r for r in facts if r['id']=='g1_max_neurons')
    row={'fact_id':fact['id'],'variant':2,'language':'en','prompt':'What is the initial maximum limit of `max_neurons` in UnifiedCore?','canonical_answer':'128.'}
    assert verify_candidate(row,fact) is None
    assert verify_candidate({**row,'prompt':'What value does `max_neurons` have at initial UnifiedCore setup?'},fact)=='MISSING_BOUND_SCOPE'


def test_final_split_and_near_duplicate_quarantine_are_not_teacher_choices(facts):
    fact=next(r for r in facts if r['id']=='config_hidden');row=example(fact)
    final=example(fact,7);final['prompt']=row['prompt']+'!'
    good,bad,_=verify_pool([row,final],[fact],{'model':'actualAGY','revision':'test'})
    assert len(good)==1 and good[0]['split']=='train' and set(REQUIRED_FIELDS)<=set(good[0])
    assert bad[0]['reason']=='NEAR_DUPLICATE_LEAKAGE'


def test_outside_domain_answer_is_policy_not_invented_factual_truth(facts):
    fact=next(r for r in facts if r['id']=='api_port')
    row={'fact_id':fact['id'],'variant':2,'language':'en','prompt':'Which listening port does the SCAR-X API use?','canonical_answer':'I do not know.'}
    assert verify_candidate(row,fact) is None
    assert verify_candidate({**row,'canonical_answer':'8765.'},fact)=='OUTSIDE_DOMAIN_REQUIRES_ABSTENTION'


def test_old_wording_schema_still_available_without_teacher_call():
    import importlib.util
    root=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('legacy_wording',root/'scripts/prepare_adaptive_curriculum.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert callable(m.validate) and callable(m.normalized) and m.normalized('Chào bạn!')==m.normalized('CHÀO BẠN?')


@pytest.mark.parametrize('state',['completed_receipt','interrupted_factory_run','interrupted_repair_run'])
def test_registered_generation_budget_cannot_restart_external_calls(tmp_path,monkeypatch,state):
    import importlib.util
    scripts=Path(__file__).resolve().parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('budget_factory',scripts/'basic_knowledge_factory.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    if state=='completed_receipt':
        (tmp_path/'artifacts').mkdir();(tmp_path/'artifacts/basic-knowledge-factory-v1.json').write_text('{}')
        protocol='basic-knowledge-factory-v1'
    else:
        protocol='basic-knowledge-repair-v1' if state=='interrupted_repair_run' else 'basic-knowledge-factory-v1'
        (tmp_path/'runs'/('20261111T000000Z-'+protocol+'-partial')).mkdir(parents=True)
    def forbidden(*args,**kwargs):raise AssertionError('CLI must not be invoked when budget was already started')
    monkeypatch.setattr(m.subprocess,'Popen',forbidden);monkeypatch.setattr(m.subprocess,'check_output',forbidden)
    with pytest.raises(ValueError,match='budget already started'):m.assert_factory_not_started(tmp_path,protocol)
    if state!='interrupted_repair_run':
        monkeypatch.setattr(m,'root',tmp_path)
        with pytest.raises(ValueError,match='budget already started'):m.main()


def test_budget_reservation_is_atomic_and_protocol_typo_fails_closed(tmp_path,monkeypatch):
    import importlib.util
    from concurrent.futures import ThreadPoolExecutor
    scripts=Path(__file__).resolve().parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('atomic_budget_factory',scripts/'basic_knowledge_factory.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with pytest.raises(ValueError,match='Unknown registered'):m.reserve_factory_budget(tmp_path,'basic-knowledge-typo')
    def reserve(_):
        try:m.reserve_factory_budget(tmp_path);return 'reserved'
        except ValueError:return 'blocked'
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(reserve,range(2)))
    assert sorted(results)==['blocked','reserved']
    with pytest.raises(ValueError,match='budget already started'):m.assert_factory_not_started(tmp_path)


def test_legacy_entrypoint_keeps_current_high_preference_and_cannot_repeat_completed_calls(tmp_path,monkeypatch):
    import importlib.util
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    spec=importlib.util.spec_from_file_location('legacy_budget_preference',scripts/'prepare_adaptive_curriculum.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.TEACHER_MODEL=='gemini-3.8-flash-high' and m.TEACHER_EFFORT=='high'
    (tmp_path/'artifacts').mkdir();(tmp_path/'artifacts/adaptive-curriculum-data.json').write_text('{}')
    monkeypatch.setattr(m,'root',tmp_path)
    def forbidden(*a,**kw):raise AssertionError('Legacy default must not spend a second completed budget')
    monkeypatch.setattr(m.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError,match='wording generation budget already started'):m.main()
