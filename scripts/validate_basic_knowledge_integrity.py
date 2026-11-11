"""Frozen data revalidation and non-executing retention fixture preparation."""
from pathlib import Path
import os,sys,json,hashlib,re
from collections import Counter,defaultdict
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.evaluation.knowledge_factory import verify_candidate,trusted_split,inference_input,REQUIRED_FIELDS,QUERY_VERIFIER_REVISION,similarity
from scarx.neural.language import tokenize
from scarx.evaluation.runs import new_run,write_json
from public_research_mirror import publish


def retention_fixture(rows):
    train=defaultdict(list)
    for row in rows:
        if row['split']=='train':train[row['fact_id']].append(row['id'])
    if len(train)!=32:raise ValueError('Expected32trainingfacts')
    rng=np.random.default_rng(44901);facts=sorted(train);angles=np.arange(len(facts))*2*np.pi/len(facts);stream=[]
    for index in range(320):
        weights=np.exp(2*np.cos(2*np.pi*index/128-angles));weights/=weights.sum();fact=facts[int(rng.choice(len(facts),p=weights))]
        stream.append(train[fact][int(rng.integers(len(train[fact])))])
    return {'format':'basic-knowledge-retention-fixture-v1','seed':44901,'stream_example_ids':stream,'evaluation_after_events':list(range(16,321,16)),'development_probe_ids':[r['id'] for r in rows if r['split']=='development'],'heldout_confirmation_ids_used':[],'learner_input_fields':['prompt'],'task_boundary_given_to_learner':False,'execution_authorized':False,'learner_executed':False,'scope':'Boundary-free changingmixture datafixture, not a trained20stage continuallearning result'}


def main():
    receipt=json.loads((root/'artifacts/basic-knowledge-verified-v1.json').read_text(encoding='utf-8'));path=root/receipt['data'];assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256'];rows=json.loads(path.read_text(encoding='utf-8'))['items']
    ledger=root/receipt['source_ledger'];assert hashlib.sha256(ledger.read_bytes()).hexdigest()==receipt['source_ledger_sha256'];facts={f['id']:f for f in json.loads(ledger.read_text(encoding='utf-8'))['facts']}
    run=new_run(root,'basic-knowledge-integrity-v1',{'protocol':'experiments/basic-knowledge-factory-v1.md','display_title':'FrozenFlashHigh pool · source/query/language/input boundary + retentionfixture','observability_kind':'data_validation','stage':2,'model_scope':'Revalidate frozen candidates without newteacher calls, studenttraining or finalqualityscores. Retention schedule metadata only.'},data_description='Frozen320FlashHigh records and40publicASTfacts; no privateinputs.')
    issues=[];formats=defaultdict(set)
    for row in rows:
        variant=int(row['id'].rsplit('_p',1)[1]);fact=facts[row['fact_id']];raw={k:row[k] for k in ('fact_id','language','prompt','canonical_answer')};raw['variant']=variant
        reason=verify_candidate(raw,fact)
        if reason:issues.append({'id':row['id'],'reason':reason})
        if not set(REQUIRED_FIELDS)<=set(row) or row['source_hash']!=fact['source_hash'] or row['source_id']!=fact['source_id'] or row['split']!=trusted_split(fact,variant):issues.append({'id':row['id'],'reason':'PROVENANCE_OR_SPLIT'})
        if hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=row['raw_candidate_sha256']:issues.append({'id':row['id'],'reason':'RAW_CANDIDATE_HASH'})
        assert set(inference_input(row))=={'prompt'} and inference_input(row)['prompt']==row['prompt']
        formats[(row['fact_id'],row['language'])].add(' '.join(tokenize(row['canonical_answer'])))
    fixture=retention_fixture(rows);write_json(run/'retention_stream.json',fixture)
    duplicates=defaultdict(list)
    for row in rows:duplicates[' '.join(tokenize(row['prompt']))].append(row)
    cross=[{'ids':[r['id'] for r in group]} for group in duplicates.values() if len({r['split'] for r in group})>1]
    near=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a['fact_id']!=b['fact_id'] or a['split']==b['split']:continue
            w,c=similarity(a['prompt'],b['prompt'])
            if w>=.9 or c>=.8:near.append({'ids':[a['id'],b['id']],'word_jaccard':w,'character_jaccard':c})
    result={'run':run.relative_to(root).as_posix(),'decision':'FROZEN_POOL_INTEGRITY_PASS' if not issues and not cross and not near else 'FROZEN_POOL_REQUIRES_QUERY_REVIEW','verifier_revision':QUERY_VERIFIER_REVISION,'source_data':receipt['data'],'source_sha256':receipt['sha256'],'records':len(rows),'source_facts':len(facts),'issues':issues,'cross_split_exact_duplicates':cross,'same_fact_cross_split_near_duplicates':near,'answer_token_format_variants':{k[0]+':'+k[1]:len(v) for k,v in formats.items() if len(v)>1},'format_variant_grouping':'fact_id and language; translated abstention phrases are not formatting conflicts','retention_fixture':(run/'retention_stream.json').relative_to(root).as_posix(),'retention_events':320,'retention_stages':20,'retention_executed':False,'input_boundary_checked':True,'new_teacher_calls':0,'main_training':False,'pilot_run':False,'final_quality_scores_opened':False,'frozen_source_preserved':hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']}
    write_json(run/'result.json',result);write_json(root/'artifacts/basic-knowledge-integrity-v1.json',result);(run/'report.md').write_text('# Frozen basicdata integrity\n\nIndependent source/query/language checks, not neuralquality. No newgeneration or training; retentionfixture not executed.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');publish(run)
    print(json.dumps({'decision':result['decision'],'records':len(rows),'issue_reasons':dict(Counter(r['reason'] for r in issues)),'exact_leak_groups':len(cross),'format_variant_facts':len(result['answer_token_format_variants']),'retention_executed':False}))


if __name__=='__main__':main()
