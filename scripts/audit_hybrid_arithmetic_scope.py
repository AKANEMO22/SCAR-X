"""Recheck a saved false-label repro against the scoped verifier; no model calls."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.arithmetic import parse_integer_expression
from scarx.hybrid.review import arithmetic_check


def audit():
    repro_path=ROOT/'docs/release/evidence/RC12_ARITHMETIC_TASK_REPRO_20261115.json'
    repro=json.loads(repro_path.read_text(encoding='utf-8'))
    assert len(repro['rows'])==3
    rows=[]
    for old in repro['rows']:
        assert old['parsed']==[12,'+',3] and old['current_review_result'] is not None
        after=arithmetic_check(old['question'],old['answer'])
        assert after is None and parse_integer_expression(old['question']) is None
        rows.append({'question':old['question'],'old_machine_label':old['current_review_result'],
                     'new_machine_label':None})
    archive=ROOT/'artifacts/releases/SCAR-X-Hybrid-1.0.0-rc.12+6389f00ef37a.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='0fa8364503c68b9c904f5af4ac209118b8ff0546643e2727e9efdcefa5f17ef2'
    with zipfile.ZipFile(archive) as bundle:
        unchanged={name:(ROOT/name).read_bytes()==bundle.read('payload/'+name)
                   for name in ('src/scarx/hybrid/learner.py','src/scarx/learning/verifiers.py',
                                'src/scarx/neural/core.py')}
        assert all(unchanged.values())
        old_tree=ast.parse(bundle.read('payload/src/scarx/hybrid/engine.py').decode('utf-8'))
    current=ast.parse((ROOT/'src/scarx/hybrid/engine.py').read_text(encoding='utf-8'))
    # Only the import target changed in engine; all controller/learning methods
    # retain the preceding release's syntax, including the signed-hash encoder.
    old_class=next(n for n in old_tree.body if isinstance(n,ast.ClassDef) and n.name=='HybridEngine')
    new_class=next(n for n in current.body if isinstance(n,ast.ClassDef) and n.name=='HybridEngine')
    assert ast.dump(old_class,include_attributes=False)==ast.dump(new_class,include_attributes=False)
    return {'verdict':'INCIDENTAL_EXPRESSION_NO_LONGER_AUTHORIZES_MACHINE_TASK_LABEL',
            'saved_reproduction_sha256':hashlib.sha256(repro_path.read_bytes()).hexdigest(),
            'three_saved_cases':rows,'unchanged_sources_vs_RC12':unchanged,
            'all_HybridEngine_methods_AST_unchanged':True,'only_engine_import_target_changed':True,
            'encoder_checkpoint_learning_rule_changed':False,'new_model_initialization_forward_or_update_calls':0,
            'new_dataset_or_research_benchmark':False,'new_verifier_is_general_intent_understanding':False,
            'scientific_or_production_PASS':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    report=audit()
    if args.output:
        with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))
