"""Post-run bounded comparator correction; no optimizer/ALIF/live changes."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sqlite3,struct,subprocess,time,traceback,uuid
import psutil,torch
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
from peft import PeftModel
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1];MODEL=ROOT/'local/hybrid-v1/models/Qwen3.5-0.8B'


def enc(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def main():
    original=Path((ROOT/'local/current-shared-lora-retention20.txt').read_text(encoding='utf-8').strip())
    source_result=json.loads((original/'result.json').read_text(encoding='utf-8'))
    assert source_result['failures']==['ValueError: No truncation; token bound512'] and len(source_result['stages'])==20
    assert source_result['top_level_model_calls']==1344 and source_result['optimizer_attempts']==480
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-retention-comparator-view-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    with (ROOT/'local/retention20-comparator-view-repair-v1-execution.json').open('x',encoding='utf-8') as f:json.dump({'run':str(run)},f)
    (ROOT/'local/current-retention-comparator-view.txt').write_text(str(run),encoding='utf-8')
    rows=json.loads((original/'locked-data.json').read_text(encoding='utf-8'));train=[r for r in rows if r['split']=='TRAIN'];test=[r for r in rows if r['split']=='TEST']
    artifact=ROOT/source_result['last_saved_adapter'];begun=time.time();calls=0
    report={'run_id':run.name,'completed':False,'failures':[],'original_run':original.name,
        'original_result_sha256':hashlib.sha256((original/'result.json').read_bytes()).hexdigest(),
        'original1344calls_and480updates_remain_charged':True,'optimizer_or_ALIF_updates':0,
        'original_whole_protocol_is_still_STOPPED_NOT_PASS':True,'predictions':[],'production_PASS':False}
    def event(title,payload):
        with sqlite3.connect(ROOT/'local/hybrid-v1/hybrid.sqlite',timeout=5) as db:
            db.execute('INSERT INTO events(created,actor,kind,payload) VALUES(?,?,?,?)',(time.time(),'orchestrator','work_trace.activity',enc({
                'kind':'actual_corrected_comparator_only','title':title,'run_id':run.name,'measured':True,'SNN_updates':0,**payload})))
    def guard():
        if time.time()-begun>600:raise TimeoutError('Comparator10minute budget')
        info=psutil.Process().memory_info()
        if max(info.rss,getattr(info,'private',0))>8_000_000_000:raise MemoryError('Host8GB cap')
        raw=subprocess.check_output([str(Path(os.environ['WINDIR'])/'System32/nvidia-smi.exe'),'--id=0','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=3,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if int(raw.split(',')[0].strip())<512:raise MemoryError('Dedicated reserve512MiB')
    def charge(kind,ident):
        nonlocal calls
        guard()
        if calls>=42:raise RuntimeError('42call cap')
        calls+=1
        with (run/'attempts.jsonl').open('a',encoding='utf-8') as f:f.write(enc({'attempt':calls,'kind':kind,'id':ident,'epoch':time.time()})+'\n')
    try:
        torch.manual_seed(260117);torch.cuda.manual_seed_all(260117);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);prompts=[]
        for row in test:
            words=set(row['error'].lower().split());examples=sorted([r for r in train if r['api']==row['api']],key=lambda r:-len(words&set(r['error'].lower().split())))[:2]
            views=[]
            for r in examples:
                public=json.loads(r['messages'][-1]['content'])
                views.append({'error':r['error'],'request':public['request'],'repair':r['target']})
            messages=row['messages']+[{'role':'user','content':'Verified TRAIN examples, use the current request values: '+enc(views)}]
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer(text,add_special_tokens=False)['input_ids'];target=tokenizer(row['target'],add_special_tokens=False)['input_ids']
            if len(ids)+len(target)+1>512:raise ValueError('Corrected view still exceeds512; no truncation')
            prompts.append({'id':row['id'],'api':row['api'],'messages':messages,'input_ids':ids,'target':row['target']})
        (run/'locked-corrected-prompts.json').write_text(enc(prompts),encoding='utf-8')
        (run/'source.py').write_bytes(Path(__file__).read_bytes())
        (run/'protocol.md').write_bytes((ROOT/'experiments/shared-lora-retention-comparator-view-repair-v1.md').read_bytes())
        guard();base=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,dtype='auto',attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True).to('cuda:0')
        weights=MODEL/'model.safetensors-00001-of-00001.safetensors'
        with weights.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
        params=dict(base.named_parameters());aux=torch.nn.ParameterDict();base.add_module('scarx_preserved_source_auxiliary',aux)
        with safe_open(weights,framework='pt',device='cpu') as source:
            for name,spec in header.items():
                if name=='__metadata__':continue
                if name not in params:
                    if not name.startswith('mtp.'):raise ValueError('Original source tensor missing')
                    aux[name.replace('.','__')]=torch.nn.Parameter(source.get_tensor(name).to('cuda:0'),requires_grad=False);params[name]=aux[name.replace('.','__')]
                dtype={'F32':torch.float32,'BF16':torch.bfloat16}[spec['dtype']]
                if params[name].dtype!=dtype:params[name].data=source.get_tensor(name).to(device='cuda:0',dtype=dtype)
        model=PeftModel.from_pretrained(base,artifact,is_trainable=False,local_files_only=True,autocast_adapter_dtype=True);model.eval()
        with model.disable_adapter():
            for p in prompts:
                charge('frozen_base_RAG',p['id'])
                with torch.no_grad():out=model.generate(input_ids=torch.tensor([p['input_ids']],device='cuda:0'),max_new_tokens=96,
                    do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
                text=tokenizer.decode(out[0,len(p['input_ids']):],skip_special_tokens=True)
                try:correct=json.loads(text)==json.loads(p['target'])
                except ValueError:correct=False
                item={'id':p['id'],'api':p['api'],'actual_answer':text,'target':p['target'],'exact_correct':correct}
                report['predictions'].append(item);event('Actual corrected RAG comparator answer',item)
        fixed=torch.tensor([prompts[0]['input_ids']],device='cuda:0');charge('reload_probe','before')
        with torch.no_grad():a=model(input_ids=fixed,use_cache=False,logits_to_keep=1).logits.detach().clone()
        model.load_adapter(str(artifact),adapter_name='roundtrip',is_trainable=False);model.set_adapter('roundtrip');charge('reload_probe','after')
        with torch.no_grad():b=model(input_ids=fixed,use_cache=False,logits_to_keep=1).logits.detach()
        assert torch.equal(a,b)
        report.update(completed=True,verdict='CORRECTED_FROZEN_RAG_COMPARATOR_MEASURED_NO_ORIGINAL_VERDICT_REWRITE',
            frozen_base_RAG_accuracy=sum(p['exact_correct'] for p in report['predictions'])/40,
            by_API={api:sum(p['exact_correct'] for p in report['predictions'] if p['api']==api)/sum(p['api']==api for p in report['predictions']) for api in ('AddAlarm','AddReminder','CompleteReminder','DeleteAlarm')},
            adapter_artifact_reload_logits_exact=True,maximum_prompt_plus_target_tokens=max(len(p['input_ids'])+len(tokenizer(p['target'],add_special_tokens=False)['input_ids'])+1 for p in prompts))
    except Exception as exc:
        report['failures'].append(type(exc).__name__+': '+str(exc));report['verdict']='CORRECTED_COMPARATOR_STOPPED_NOT_PASS'
        (run/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        report.update(new_top_level_model_calls=calls,elapsed_seconds=time.time()-begun)
        (run/'result.json').write_text(enc(report),encoding='utf-8');event('Corrected comparator terminal',{k:v for k,v in report.items() if k!='predictions'})
    print(enc({'run':str(run),**{k:v for k,v in report.items() if k!='predictions'}}),flush=True)
    return 0 if report['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
