"""Data-only mask delta and exact AST scope check; no controller/model execution."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.store import Store,encode
from scarx.learning.verifiers import parse_integer_expression


def method_tree(raw,skip_allowed=False):
    tree=ast.parse(raw)
    if skip_allowed:
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name=='HybridEngine':
                node.body=[m for m in node.body if not isinstance(m,ast.FunctionDef) or m.name!='allowed']
    return ast.dump(tree,include_attributes=False)


def main(run):
    run=run.resolve();assert run.is_relative_to(ROOT/'runs')
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    manifest,cases,result=map(read,('manifest.json','locked-cases.json','result.json'))
    assert result['gate_pass'] and result['seconds']<=15 and result['neural_or_VLM_calls']==0
    assert hashlib.sha256((run/'locked-cases.json').read_bytes()).hexdigest()==manifest['cases_sha256']
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==digest
    original=(run/'source/src/scarx/hybrid/engine.py').read_bytes()
    candidate=(run/'candidate-engine.py').read_bytes()
    assert hashlib.sha256(candidate).hexdigest()==manifest['candidate_sha256']
    assert method_tree(original,True)==method_tree(candidate,True)
    assert method_tree(original)!=method_tree(candidate)
    assert len(cases)==len(result['rows'])==192
    changed=0;supported_same=0;limitations=0
    for case,row in zip(cases,result['rows']):
        assert case['id']==row['case'] and case['condition']==row['condition']
        accepts=parse_integer_expression(case['question']) is not None
        assert accepts==case['expected_arithmetic']==row['checker_accepts']
        before,after=row['before'],row['after']
        expected=before if accepts else [action for action in before if action!='verify']
        assert after==expected
        changed+=int(before!=after)
        supported_same+=int(accepts and before==after)
        if case['condition']=='needs_local_review':assert before==after==['review']
        if case['condition']=='already_attempted':assert 'verify' not in after
        if case['condition']=='pending_question':assert 'ask' not in after
        if case['condition']=='search_rate_exhausted':assert 'search' not in after
        if case.get('intent_understanding_not_verified'):
            assert accepts
            limitations+=1
    assert changed==result['removed_unsupported_verify_masks']==28
    report={'run_id':run.name,'verdict':'SHADOW_CONTROLLER_MASK_DELTA_VERIFIED_NOT_SEMANTIC_OR_NEURAL_PASS',
        'cases':len(cases),'only_allowed_method_AST_changed':True,'unsupported_verify_removals':changed,
        'supported_masks_identical':supported_same,'source_neural_or_learning_methods_changed':False,
        'existing_parser_incidental_expression_contexts':limitations,
        'all_other_authorization_constraints_retained':True,'live_source_activation':False,
        'new_neural_or_VLM_calls':0,'learned_tool_repair_or_user_intent_understanding_proven':False}
    with (run/'independent-data-audit-v1.json').open('x',encoding='utf-8') as out:out.write(encode(report))
    store=Store(ROOT/'local/hybrid-v1')
    try:store.event('verifier_capability_guard.data_verified',report,'orchestrator')
    finally:store.close()
    print(encode(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);main(parser.parse_args().run)
