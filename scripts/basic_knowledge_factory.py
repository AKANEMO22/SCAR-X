"""Extension of prepare_adaptive_curriculum: actual FlashHigh candidates/oracle."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,tempfile,io,zipfile
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from prepare_adaptive_curriculum import normalized
from scarx.evaluation.knowledge_factory import verify_pool,REQUIRED_FIELDS
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.experiment_status import progress
from public_research_mirror import publish

MODEL='gemini-3.8-flash-high';EFFORT='high'


def assert_factory_not_started(workspace,protocol='basic-knowledge-factory-v1'):
    """A completed or interrupted registered budget is never an automatic retry.

    A run directory is evidence an invocation may already have reached the CLI.
    An unknown partial state needs review, not another four external calls.
    """
    workspace=Path(workspace)
    if protocol not in ('basic-knowledge-factory-v1','basic-knowledge-repair-v1'):
        raise ValueError('Unknown registered generation protocol')
    receipts=('basic-knowledge-factory-v1.json','basic-knowledge-verified-v1.json') if protocol=='basic-knowledge-factory-v1' else ('basic-knowledge-verified-v1.json',)
    reservation=workspace/'artifacts/generation-budgets'/protocol
    if reservation.exists() or any((workspace/'artifacts'/name).exists() for name in receipts) or any((workspace/'runs').glob('*-'+protocol+'-*')):
        raise ValueError('Registered generation budget already started; no automatic additional AGY calls. Revalidate existing data or register a separately approved dataset version.')


def reserve_factory_budget(workspace,protocol='basic-knowledge-factory-v1'):
    """Atomically reserve a registered invocation, including interrupted attempts."""
    workspace=Path(workspace);assert_factory_not_started(workspace,protocol)
    parent=workspace/'artifacts/generation-budgets';parent.mkdir(parents=True,exist_ok=True)
    reservation=parent/protocol
    try:reservation.mkdir()
    except FileExistsError as exc:raise ValueError('Registered generation budget already started concurrently; no automatic additional AGY calls') from exc
    write_json(reservation/'reservation.json',{'protocol':protocol,'created_unix':time.time(),
        'status':'ATTEMPT_RESERVED','automatic_retry_allowed':False})
    return reservation


def generate_batch(cli,facts,run,index,private,repair_note=None,total_batches=4):
    sandbox=Path(tempfile.mkdtemp(prefix='scarx-agy-basic-')).resolve()
    schema={'type':'object','properties':{'items':{'type':'array','minItems':8*len(facts),'maxItems':8*len(facts),'items':{'type':'object','properties':{'fact_id':{'type':'string','enum':[f['id'] for f in facts]},'variant':{'type':'integer','minimum':1,'maximum':8},'language':{'type':'string','enum':['vi','en']},'prompt':{'type':'string'},'canonical_answer':{'type':'string'}},'required':['fact_id','variant','language','prompt','canonical_answer'],'additionalProperties':False}}},'required':['items'],'additionalProperties':False}
    schema_path=sandbox/'schema.json';write_json(schema_path,schema)
    publicfacts=[{k:f[k] for k in ('id','scope','field','value','value_type','maximum','known_domain')} for f in facts]
    prompt=(root/'data/contracts/ANTIGRAVITY_BASIC_KNOWLEDGE_V1.md').read_text(encoding='utf-8')+'\nPUBLIC VERIFIED FACTS:\n'+json.dumps(publicfacts,ensure_ascii=False)
    if repair_note is not None:prompt+='\nSOURCE-VALIDATION REPAIR NOTES (no modelquality scores):\n'+json.dumps(repair_note,ensure_ascii=False)
    invocation=[str(cli),'--print',prompt,'--model',MODEL,'--effort',EFFORT,'--output-format','stream-json','--json-schema',str(schema_path),'--disable-slash-commands','--print-timeout','4m']
    folder=private/f'batch-{index}';folder.mkdir();write_json(folder/'invocation.json',{'argv':invocation,'cwd':str(sandbox)})
    process=None;start=time.perf_counter();error=None;values=[];reported=None;usage=None
    with ResourceMonitor(limit=1_000_000_000,interval=.1) as monitor,(folder/'events.jsonl').open('wb') as out,(folder/'stderr.log').open('wb') as err:
        try:
            process=subprocess.Popen(invocation,cwd=sandbox,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            print(json.dumps({'batch':index,'pid':process.pid,'model_requested':MODEL,'effort':EFFORT,'facts':len(facts)}),flush=True)
            while process.poll() is None:
                monitor.check();progress(run,f'AGY FlashHigh batch{index} đang sinh candidate',index-1,total_batches,metrics={'cli_pid':process.pid,'batch_seconds':time.perf_counter()-start,'main_training':False});publish(run)
                if time.perf_counter()-start>270:raise TimeoutError('Registered270s batch cap')
                time.sleep(1)
            events=folder/'events.jsonl'
            if events.stat().st_size>12_000_000:raise ValueError('Teacherstream cap')
            parsed=[]
            for line in events.read_text(encoding='utf-8-sig').splitlines():
                try:parsed.append(json.loads(line))
                except ValueError:continue
            inits=[e['init'] for e in parsed if e.get('event')=='init'];reported=inits[-1].get('model') if inits else None
            if reported!=MODEL:raise ValueError('Requested/reportedmodel mismatch or missing attestation')
            results=[e['result'] for e in parsed if e.get('event')=='result']
            if process.returncode or not results or results[-1].get('status')!='SUCCESS':raise ValueError('AGY result not SUCCESS')
            result=results[-1];value=result.get('structured_output')
            if value is None:value=json.loads(result['response'])
            steps=[e['step_update'] for e in parsed if e.get('step_update',{}).get('step_type')=='tool']
            if any(s.get('tool_name')!='finish' or s.get('tool_info',{}).get('parameters')!=value for s in steps):raise ValueError('Unexpected toolstep; batch quarantined')
            values=value.get('items',[])
            if not isinstance(values,list) or len(values)!=8*len(facts):raise ValueError('Batch item count')
            usage={k:v for k,v in result.get('usage',{}).items() if k in ('input_tokens','output_tokens','thinking_tokens','total_tokens','cache_read_tokens')}
            write_json(run/f'candidates-batch-{index}.json',{'items':values,'automatic_learning_allowed':False,'generator':MODEL})
        except Exception as exc:error=type(exc).__name__+': '+str(exc)[:200];values=[]
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        receipt={'batch':index,'items':len(values),'model_requested':MODEL,'model_reported':reported,'effort_requested':EFFORT,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'schema_sha256':hashlib.sha256(schema_path.read_bytes()).hexdigest(),'stream_sha256':hashlib.sha256((folder/'events.jsonl').read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(json.dumps(values,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'usage':usage,'cost_usd_reported':None,'seconds':time.perf_counter()-start,'resources':monitor.report(),'error':error,'invocation_private':(folder/'invocation.json').relative_to(root).as_posix()}
    return values,receipt


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    assert_factory_not_started(root)
    cli=Path.home()/'.gemini/bin/agy.exe';assert cli.is_file()
    version=subprocess.check_output([str(cli),'--version'],text=True,timeout=10).strip()
    with cli.open('rb') as f:executable_sha=hashlib.file_digest(f,'sha256').hexdigest()
    ledger=json.loads((root/'artifacts/basic-fact-ledger-v1.json').read_text(encoding='utf-8'));path=(root/ledger['data']).resolve();assert path.is_relative_to(root/'data/candidates') and hashlib.sha256(path.read_bytes()).hexdigest()==ledger['sha256']
    facts=json.loads(path.read_text(encoding='utf-8'))['facts'];assert len(facts)==40
    reserve_factory_budget(root)
    run=new_run(root,'basic-knowledge-factory-v1',{'protocol':'experiments/basic-knowledge-factory-v1.md','display_title':'AGY FlashHigh ·40facts/320candidates · source verifier · no train','observability_kind':'data_generation','model_scope':'Userselected FlashHigh effortHigh. ActualAGY questions+candidateanswers, independentAST verification. No neuraltraining/pilot, no privateinputs.','model_requested':MODEL,'effort':EFFORT,'cli_version':version,'calls_max':5,'batches':4,'stage':2},data_description='Only40public sourceverifiedfact scalars and controlledquery contract; no privatehistory/SQLite/checkpoints/weights.',data_files=[ledger['data'],'data/contracts/ANTIGRAVITY_BASIC_KNOWLEDGE_V1.md'])
    private=root/'local/basic-knowledge-factory'/run.name;private.mkdir(parents=True);directory=root/'data/candidates'/run.name;directory.mkdir(parents=True)
    (run/'report.md').write_text('# Basic knowledgefactory\n\nActual FlashHigh generation/oracle validation in progress, no student training.\n',encoding='utf-8');publish(run)
    candidates=[];receipts=[];started=time.perf_counter()
    for index in range(4):
        values,record=generate_batch(cli,facts[index*10:(index+1)*10],run,index+1,private);receipts.append(record);candidates.extend(values)
        write_json(run/'batch-receipts.json',receipts);publish(run)
        if record['error']:break
    generator={'model':'AntigravityCLI/'+MODEL,'revision':version+'@'+executable_sha}
    verified,quarantine,missing=verify_pool(candidates,facts,generator)
    for split in sorted({r['split'] for r in verified}):
        rows=[r for r in verified if r['split']==split];(directory/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    write_json(directory/'verified-pool.json',{'items':verified,'automatic_learning_allowed':False,'final_scores_opened':False})
    write_json(directory/'quarantine.json',{'items':quarantine,'missing':missing})
    result={'run':run.relative_to(root).as_posix(),'decision':'CANDIDATE_POOL_SOURCE_VERIFIED_COMPATIBILITY_PENDING','data':(directory/'verified-pool.json').relative_to(root).as_posix(),'sha256':hashlib.sha256((directory/'verified-pool.json').read_bytes()).hexdigest(),'quarantine':(directory/'quarantine.json').relative_to(root).as_posix(),'facts':40,'expected_candidates':320,'actual_candidates':len(candidates),'verified':len(verified),'quarantined':len(quarantine),'missing_slots':len(missing),'batches':receipts,'cli_version':version,'executable_sha256':executable_sha,'model_requested':MODEL,'effort':'high','source_ledger':ledger['data'],'source_ledger_sha256':ledger['sha256'],'schema_required_fields':list(REQUIRED_FIELDS),'label_origin':'AGY candidate answers verified independently; no Codex/Qwen answerlabels','automatic_learning_allowed':False,'student_training':False,'pilot_run':False,'sealed_final_score_used':False,'seconds':time.perf_counter()-started}
    write_json(run/'result.json',result);write_json(root/'artifacts/basic-knowledge-factory-v1.json',result)
    (run/'report.md').write_text('# ActualAntigravity basic knowledge candidatefactory\n\nSourceverification does not establish codec compatibility, learning, quality or production readiness.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');publish(run)
    print(json.dumps({k:result[k] for k in ('run','decision','actual_candidates','verified','quarantined','missing_slots','seconds')}))
    return int(any(r['error'] for r in receipts))


if __name__=='__main__':raise SystemExit(main())
