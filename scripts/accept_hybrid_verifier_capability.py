"""Actual controller authorization masks on shadow source, no model instances."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.store import Store,encode
from scarx.learning.verifiers import parse_integer_expression


def recipe():
    spec=importlib.util.spec_from_file_location('verifier_guard_acceptance',ROOT/'integrations/hermes/verifier_capability_guard_patch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def locked_cases():
    parent=ROOT/'runs/20261114T102423Z-learning-target-audit-v1-70f370'
    ledger=json.loads((parent/'event-ledger.json').read_text(encoding='utf-8'))
    goals={r['payload']['id']:r['payload'] for r in ledger if r['kind']=='learning_goal_created'}
    proofs=[r['payload'] for r in ledger if r['kind']=='machine_verification']
    rows=[{'id':'old-verified-'+str(i),'question':goals[p['goal_id']]['question'],
           'origin':'historical_owned_arithmetic','expected_arithmetic':True} for i,p in enumerate(proofs)]
    failed=json.loads((parent/'unsupported-action-feedback-context-v1.json').read_text(encoding='utf-8'))['failures']
    for i,entry in enumerate(failed):
        goal=entry['observed_failure']['payload']['goal_id']
        rows.append({'id':'old-unsupported-'+str(i),'question':goals[goal]['question'],
                     'origin':'actual_old_verification_unavailable','expected_arithmetic':False})
    for i,text in enumerate(('12+3','9 minus 11','có 9 bánh, thêm 2','１８＋３')):
        rows.append({'id':'supported-'+str(i),'question':text,'expected_arithmetic':True})
    for i,text in enumerate(('12*3','123+2','12+3+4','not 12+3','không tính 12+3','màu chiếc xe là gì?')):
        rows.append({'id':'unsupported-'+str(i),'question':text,'expected_arithmetic':False})
    # Document the existing parser's intent limit rather than silently treating
    # arithmetic-looking text as proof the user requested a sum.
    rows.append({'id':'incidental-expression-limitation','question':'What color is 12+3 written in?',
                 'expected_arithmetic':True,'intent_understanding_not_verified':True})
    cases=[]
    for row in rows:
        for condition in ('reviewed','needs_local_review','already_attempted','pending_question','public_search','search_rate_exhausted'):
            cases.append({**row,'condition':condition})
    return cases


def context(store,condition):
    now=time.time()
    return SimpleNamespace(store=store,state={'search_times':[now]*12 if condition=='search_rate_exhausted' else [],
        'question':{'id':'owned-question'} if condition=='pending_question' else None},
        config=SimpleNamespace(search_per_hour=12))


def goal(case):
    condition=case['condition']
    return dict(question=case['question'],public_topic_id='owned-approved' if condition in ('public_search','search_rate_exhausted') else None,
                origin='user',reviewed=condition!='needs_local_review',attempts=0 if condition=='needs_local_review' else 1,
                verified_attempted=condition=='already_attempted',searched=False)


def main():
    begun=time.monotonic()
    run=ROOT/'runs'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-verifier-capability-p0-v1-'+uuid.uuid4().hex[:6])
    run.mkdir();cases=locked_cases()
    (run/'locked-cases.json').write_text(encode(cases),encoding='utf-8')
    original=(ROOT/'src/scarx/hybrid/engine.py').read_bytes();changed=recipe().patch_bytes(original)
    (run/'candidate-engine.py').write_bytes(changed)
    scope={'__name__':'scarx.hybrid._verifier_capability_acceptance','__package__':'scarx.hybrid'}
    exec(compile(changed,'candidate-engine.py','exec'),scope)
    Candidate=scope['HybridEngine']
    paths=[Path(__file__),ROOT/'integrations/hermes/verifier_capability_guard_patch.py',
           ROOT/'src/scarx/hybrid/engine.py',ROOT/'src/scarx/learning/verifiers.py',
           ROOT/'experiments/hybrid-verifier-capability-guard-p0-v1.md']
    hashes={}
    for path in paths:
        rel=path.relative_to(ROOT);dest=run/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());hashes[rel.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    (run/'manifest.json').write_text(encode({'run_id':run.name,'source_sha256':hashes,
        'candidate_sha256':hashlib.sha256(changed).hexdigest(),'cases_sha256':hashlib.sha256((run/'locked-cases.json').read_bytes()).hexdigest(),
        'no_model_creation_forward_training_or_live_activation':True}),encoding='utf-8')
    store=Store(ROOT/'local'/run.name)
    report={'run_id':run.name,'rows':[],'neural_or_VLM_calls':0,'live_source_changed':False,
            'arithmetic_parser_expanded':False,'semantic_user_intent_verifier_added':False}
    try:
        store.put('public_topic','owned-approved',{'id':'owned-approved','topic':'owned fixture'})
        for case in cases:
            assert (parse_integer_expression(case['question']) is not None)==case['expected_arithmetic']
            before=HybridEngine.allowed(context(store,case['condition']),goal(case))
            after=Candidate.allowed(context(store,case['condition']),goal(case))
            expected=before if case['expected_arithmetic'] else [v for v in before if v!='verify']
            assert after==expected
            if case['condition']=='needs_local_review':assert before==after==['review']
            if case['condition']=='pending_question':assert 'ask' not in after
            if case['condition']=='already_attempted':assert 'verify' not in after
            if case['condition']=='search_rate_exhausted':assert 'search' not in after
            report['rows'].append({'case':case['id'],'condition':case['condition'],
                'before':before,'after':after,'checker_accepts':case['expected_arithmetic'],
                'intent_understanding_not_verified':case.get('intent_understanding_not_verified',False)})
        assert (ROOT/'src/scarx/hybrid/engine.py').read_bytes()==original
        assert time.monotonic()-begun<=15
        report.update(gate_pass=True,verdict='SHADOW_VERIFIER_MASK_CONTRACT_PASS_NOT_LEARNED_REPAIR',
            cases=len(cases),removed_unsupported_verify_masks=sum('verify' in r['before'] and 'verify' not in r['after'] for r in report['rows']),
            masks_otherwise_unchanged=True,parser_incidental_expression_limitation_retained=True)
    finally:store.close()
    report['seconds']=time.monotonic()-begun
    with (run/'result.json').open('x',encoding='utf-8') as out:out.write(encode(report))
    live=Store(ROOT/'local/hybrid-v1')
    try:live.event('verifier_capability_guard.shadow_completed',{k:v for k,v in report.items() if k!='rows'},'orchestrator')
    finally:live.close()
    print(encode({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
