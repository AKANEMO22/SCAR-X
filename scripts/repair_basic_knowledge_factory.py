"""Single preregistered AGY repair, same strict source/query rules. No training."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from basic_knowledge_factory import generate_batch,MODEL,assert_factory_not_started,reserve_factory_budget
from scarx.evaluation.knowledge_factory import verify_pool
from scarx.evaluation.runs import new_run,write_json
from public_research_mirror import publish


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    assert_factory_not_started(root,'basic-knowledge-repair-v1')
    receipt=root/'artifacts/basic-knowledge-verified-v1.json'
    if receipt.exists():raise ValueError('Repair already attempted; no automatic additional calls')
    parent=json.loads((root/'artifacts/basic-knowledge-factory-v1.json').read_text(encoding='utf-8'))
    if len(parent['batches'])!=4 or any(r['error'] for r in parent['batches']):raise ValueError('Repair is not a retry for failed/incomplete generation batches')
    ledger=root/parent['source_ledger'];assert hashlib.sha256(ledger.read_bytes()).hexdigest()==parent['source_ledger_sha256'];facts=json.loads(ledger.read_text(encoding='utf-8'))['facts']
    rejected=json.loads((root/parent['quarantine']).read_text(encoding='utf-8'));missing={(r['fact_id'],r['variant']) for r in rejected['missing']};ids=sorted({k[0] for k in missing})[:10]
    chosen=[f for f in facts if f['id'] in ids];assert chosen and len(chosen)<=10
    cli=Path.home()/'.gemini/bin/agy.exe';version=subprocess.check_output([str(cli),'--version'],text=True,timeout=10).strip()
    with cli.open('rb') as f:exe_sha=hashlib.file_digest(f,'sha256').hexdigest()
    assert version==parent['cli_version'] and exe_sha==parent['executable_sha256']
    reserve_factory_budget(root,'basic-knowledge-repair-v1')
    run=new_run(root,'basic-knowledge-repair-v1',{'protocol':'experiments/basic-knowledge-factory-v1.md','display_title':'FlashHigh · bounded repair of11source/parser rejects · no train','observability_kind':'data_generation','model_requested':MODEL,'effort':'high','repair_calls_max':1,'prior_calls':4,'stage':2,'model_scope':'Same source/query/schema/leakage rules. No quality scores or threshold changes, no privateinputs/studentlearning.'},data_description='Existing publicfacts and source-validation rejection metadata only.',data_files=[parent['source_ledger'],'data/contracts/ANTIGRAVITY_BASIC_KNOWLEDGE_V1.md'])
    private=root/'local/basic-knowledge-factory'/run.name;private.mkdir(parents=True);directory=root/'data/candidates'/run.name;directory.mkdir(parents=True)
    (run/'report.md').write_text('# Bounded sourcevalidationrepair\n\nOne preregistered FlashHigh call, no criterion change or studentlearning.\n',encoding='utf-8');publish(run)
    note={'missing_slots':[{'fact_id':f,'variant':v} for f,v in sorted(missing)],'important':'For API questions avoid Config, UnifiedCore AND the substring configuration; ask only about API subject. Avoid any spelled-out numeric filler such as one/một or negation. Existing verifier remains unchanged. Return all8variants per suppliedfact; only original missing slots can replace rejected candidates.'}
    started=time.perf_counter();values,batch=generate_batch(cli,chosen,run,5,private,repair_note=note,total_batches=5)
    selected=[r for r in values if (r.get('fact_id'),r.get('variant')) in missing]
    old=[]
    for i in range(1,5):old.extend(json.loads((root/parent['run']/f'candidates-batch-{i}.json').read_text(encoding='utf-8'))['items'])
    generator={'model':'AntigravityCLI/'+MODEL,'revision':version+'@'+exe_sha,'dataset_version':'basic-knowledge-v1.r2'}
    verified,quarantine,remaining=verify_pool(old+selected,facts,generator)
    for split in sorted({r['split'] for r in verified}):(directory/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in verified if r['split']==split),encoding='utf-8')
    pool=directory/'verified-pool.json';write_json(pool,{'items':verified,'automatic_learning_allowed':False,'final_scores_opened':False,'parent_candidate_run':parent['run']})
    write_json(directory/'quarantine.json',{'items':quarantine,'remaining_slots':remaining,'unrequested_repair_rows_ignored':len(values)-len(selected)})
    result={'run':run.relative_to(root).as_posix(),'decision':'SOURCE_POOL_REPAIRED_COMPATIBILITY_PENDING' if not remaining and not batch['error'] else 'SOURCE_POOL_INCOMPLETE_NO_MORE_AUTO_CALLS','data':pool.relative_to(root).as_posix(),'sha256':hashlib.sha256(pool.read_bytes()).hexdigest(),'parent_run':parent['run'],'source_ledger':parent['source_ledger'],'source_ledger_sha256':parent['source_ledger_sha256'],'facts':40,'known_facts':32,'verified':len(verified),'remaining_slots':len(remaining),'original_quarantines_preserved':len(rejected['items']),'repair_candidates':len(values),'repair_slots_used':len(selected),'repair_batch':batch,'total_generation_calls':5,'criteria_changed':False,'automatic_learning_allowed':False,'student_training':False,'pilot_run':False,'sealed_final_score_used':False,'seconds':time.perf_counter()-started}
    write_json(run/'result.json',result);write_json(receipt,result);(run/'report.md').write_text('# Bounded sourcevalidationrepair\n\nOld candidates/results preserved, no qualityscore selection.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');publish(run)
    print(json.dumps({k:result[k] for k in ('run','decision','verified','remaining_slots','repair_slots_used','seconds')}))
    return int(bool(remaining) or batch['error'] is not None)


if __name__=='__main__':raise SystemExit(main())
