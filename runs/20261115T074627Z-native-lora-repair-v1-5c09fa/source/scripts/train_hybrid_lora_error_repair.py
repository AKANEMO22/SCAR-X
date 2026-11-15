"""Authorized isolated native-precision Qwen LoRA pilot; no ALIF imports.

Supervised targets are independently executed teacher repairs, not self labels.
No adapter merge/live deployment, quantization or precision reduction.
"""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='1'
from datetime import datetime,timezone
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import sys
import subprocess
import time
import traceback
import uuid

import psutil
import torch
from safetensors import safe_open
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
from peft import LoraConfig,get_peft_model,PeftModel

ROOT=Path(__file__).resolve().parents[1];MODEL=ROOT/'local/hybrid-v1/models/Qwen3.5-0.8B'
SEED=260115


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)


def event(run,title,data):
    with sqlite3.connect(ROOT/'local/hybrid-v1/hybrid.sqlite',timeout=5) as db:
        db.execute('INSERT INTO events(created,actor,kind,payload) VALUES(?,?,?,?)',
            (time.time(),'orchestrator','work_trace.activity',json.dumps({'kind':'actual_VLM_LoRA_training',
             'title':title,'run_id':run.name,'measured':True,'scope':'VLM LoRA research, not ALIF learning',**data},ensure_ascii=False)))


def tensor_hash(tensor):return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    begun=time.time();run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-native-lora-repair-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    previous_marker=ROOT/'local/native-lora-repair-v1-execution.json'
    if previous_marker.exists():
        previous_run=Path(json.loads(previous_marker.read_text(encoding='utf-8'))['run'])
        previous=json.loads((previous_run/'result.json').read_text(encoding='utf-8'))
        if (previous['optimizer_attempts']!=0 or previous['forward_calls']!=0
                or previous['failures']!=['ValueError: Original tensor missing in model: mtp.fc.weight']):
            raise RuntimeError('Previous pilot consumed calls or failed outside the exact zero-call loader compatibility issue')
    with (ROOT/'local/native-lora-native-mtp-compatibility-v1-execution.json').open('x',encoding='utf-8') as f:json.dump({'run':str(run),'started':begun,'previous_zero_call_loader_failure_preserved':previous_marker.exists()},f)
    (ROOT/'local/current-native-lora-repair-run.txt').write_text(str(run),encoding='utf-8')
    data_path=Path((ROOT/'local/current-qualified-repair-data.txt').read_text(encoding='utf-8').strip())
    rows=json.loads((data_path/'verified-data.json').read_text(encoding='utf-8'))
    train=[r for r in rows if r['split']=='TRAIN'];probe=[r for r in rows if r['split']=='DEV'][:4]+[r for r in rows if r['split']=='TEST']
    assert len(train)==32 and len(probe)==20
    weights=MODEL/'model.safetensors-00001-of-00001.safetensors'
    with weights.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
    sources=[Path(__file__),ROOT/'experiments/hybrid-lora-error-repair-pilot-v1.md',ROOT/'scripts/qualify_hybrid_repair_data.py']
    for p in sources:
        out=run/'source'/p.relative_to(ROOT);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(p.read_bytes())
    write(run/'manifest.json',{'seed':SEED,'source_git_sha':os.environ.get('SCARX_LORA_SOURCE_SHA'),
        'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'data_run':data_path.name,'data_sha256':hashlib.sha256((data_path/'verified-data.json').read_bytes()).hexdigest(),
        'base_header_sha256':hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest(),
        'max_optimizer_attempts':32,'max_generation_probes_per_phase':20,'max_total_forward_calls':128,
        'base_model_revision':'2fc06364715b967f1860aea9cf38778875588b17','base_precision':'BF16 + original FP32 tensors',
        'LoRA':{'rank':4,'alpha':8,'dropout':0,'targets':['q_proj','v_proj'],'lr':1e-4},
        'no_ALIF_learning_or_live_promotion':True})
    report={'run_id':run.name,'completed':False,'failures':[],'updates':[],'predictions':[],
            'backprop_scope':'VLM LoRA only, ALIF untouched','production_or_retention_qualified':False}
    forward_calls=0;attempts=0;model=None;adapter_initialized=False
    def guard():
        if time.time()-begun>1200:raise TimeoutError('20minute pilot cap')
        memory=psutil.Process().memory_info()
        if max(memory.rss,getattr(memory,'private',0))>8_000_000_000:raise MemoryError('Host memory cap')
        free,total=torch.cuda.mem_get_info()
        if free<512*1024**2:raise MemoryError('CUDA availability below reserve; no dtype reduction/fallback')
        result=subprocess.run([str(Path(os.environ['WINDIR'])/'System32/nvidia-smi.exe'),'--id=0',
            '--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,
            timeout=3,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        physical_free,physical_total=[int(x.strip())*1024**2 for x in result.stdout.strip().split(',')]
        if physical_free<512*1024**2:raise MemoryError('Dedicated GPU reserve below512MiB')
        return {'cuda_available_free_bytes':free,'cuda_total_bytes':total,'RSS_bytes':memory.rss,
                'dedicated_GPU_free_bytes':physical_free,'dedicated_GPU_total_bytes':physical_total,
                'CUDA_allocated_bytes':torch.cuda.memory_allocated(),'CUDA_reserved_bytes':torch.cuda.memory_reserved()}
    def charge(kind,identifier):
        nonlocal forward_calls,attempts
        guard()
        if kind=='update':
            if attempts>=32:raise RuntimeError('Update cap');
            attempts+=1
        else:
            if forward_calls>=128:raise RuntimeError('Forward cap')
            forward_calls+=1
        with (run/'attempts.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'epoch':time.time(),'kind':kind,'id':identifier,'update_attempts':attempts,'forward_calls':forward_calls})+'\n');f.flush();os.fsync(f.fileno())
    try:
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        guard();tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype='auto',attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda:0')
        # Restore any loader-cast original FP32 tensors before inference/training.
        parameters=dict(model.named_parameters());source_to_loaded={};auxiliary=torch.nn.ParameterDict()
        model.add_module('scarx_preserved_source_auxiliary',auxiliary)
        with safe_open(weights,framework='pt',device='cpu') as source:
            for name,description in header.items():
                if name=='__metadata__':continue
                if name not in parameters:
                    if name=='lm_head.weight':continue
                    if name.startswith('mtp.'):
                        # Transformers has no MTP inference path in this class.
                        # Preserve every unused original tensor, unchanged and
                        # frozen, rather than silently dropping/pruning it.
                        auxiliary[name.replace('.','__')]=torch.nn.Parameter(source.get_tensor(name).to('cuda:0'),requires_grad=False)
                        parameters[name]=auxiliary[name.replace('.','__')]
                    else:raise ValueError('Original tensor missing in model: '+name)
                p=parameters[name];expected={'F32':torch.float32,'BF16':torch.bfloat16}[description['dtype']]
                if p.dtype!=expected:p.data=source.get_tensor(name).to(device='cuda:0',dtype=expected)
                if p.dtype!=expected:raise ValueError('Base dtype not preserved')
                source_to_loaded[name]=p
        base_hashes={name:tensor_hash(p) for name,p in source_to_loaded.items()}
        config=LoraConfig(r=4,lora_alpha=8,lora_dropout=0,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM')
        model=get_peft_model(model,config,autocast_adapter_dtype=True)
        adapter_initialized=True
        trainables=[p for name,p in model.named_parameters() if p.requires_grad]
        assert trainables and all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)
        model.config.use_cache=False
        write(run/'dtype-trainable-audit.json',{'native_source_dtypes_preserved':True,'base_tensor_count':len(source_to_loaded),
            'original_unused_MTP_tensors_preserved_frozen':len(auxiliary),
            'trainable_parameters':sum(p.numel() for p in trainables),'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad],
            'trainable_dtypes':sorted({str(p.dtype) for p in trainables}),'resources':guard()})
        def tokens(row):
            text=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompt=tokenizer(text,add_special_tokens=False)['input_ids'];target=tokenizer(row['target'],add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
            if len(prompt)+len(target)>256:raise ValueError('No silent truncation; input exceeds256')
            return prompt,target
        for row in train+probe:tokens(row)
        def evaluate(phase):
            model.eval()
            for row in probe:
                charge('generation',phase+':'+row['id']);prompt,_=tokens(row)
                with torch.no_grad():
                    output=model.generate(input_ids=torch.tensor([prompt],device='cuda:0'),max_new_tokens=64,do_sample=False,
                        use_cache=True,pad_token_id=tokenizer.eos_token_id)
                answer=tokenizer.decode(output[0,len(prompt):],skip_special_tokens=True)
                try:exact=json.loads(answer)==json.loads(row['target'])
                except ValueError:exact=False
                record={'phase':phase,'id':row['id'],'split':row['split'],'user_request':row['user_request'],
                    'actual_answer':answer,'expected_executable_correction':row['target'],'exact_JSON_repair':exact}
                report['predictions'].append(record);event(run,'LoRA '+phase+' actual answer',record)
        evaluate('baseline')
        model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        optimizer=torch.optim.AdamW(trainables,lr=1e-4,weight_decay=0)
        for row in train:
            charge('update',row['id']);charge('training_forward',row['id'])
            prompt,target=tokens(row);ids=torch.tensor([prompt+target],device='cuda:0')
            optimizer.zero_grad(set_to_none=True)
            output=model(input_ids=ids,use_cache=False,logits_to_keep=len(target)+1)
            logits=output.logits[:,:-1,:].float();truth=torch.tensor([target],device='cuda:0')
            if logits.shape[1]!=truth.shape[1]:raise ValueError('Completion-only causal-shift mismatch')
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),truth.reshape(-1))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(trainables,1.0);optimizer.step();torch.cuda.synchronize()
            metrics={'step':attempts,'id':row['id'],'loss':float(loss.detach()),'resources':guard()}
            report['updates'].append(metrics);event(run,'LoRA actual optimizer update',metrics)
            del output,logits,truth,ids,loss
        artifact=ROOT/'artifacts'/run.name/'adapter';artifact.mkdir(parents=True)
        model.save_pretrained(artifact,safe_serialization=True);tokenizer.save_pretrained(artifact)
        report['adapter_path']=str(artifact.relative_to(ROOT))
        model.gradient_checkpointing_disable();evaluate('final')
        # Roundtrip a saved adapter through a separately named PEFT instance
        # on the same immutable base. This is a persistence check, not a new arm.
        charge('roundtrip','default-before-reload');prompt,_=tokens(train[0])
        fixed=torch.tensor([prompt],device='cuda:0')
        with torch.no_grad():expected=model(input_ids=fixed,use_cache=False,logits_to_keep=1).logits.detach().clone()
        model.load_adapter(str(artifact),adapter_name='roundtrip',is_trainable=False)
        model.set_adapter('roundtrip');charge('roundtrip','loaded-adapter')
        with torch.no_grad():restored=model(input_ids=fixed,use_cache=False,logits_to_keep=1).logits.detach()
        if not torch.equal(expected,restored):raise ValueError('Saved/reloaded adapter logits differ')
        report['save_reload_logits_exact']=True;model.set_adapter('default')
        # Parameter objects point to frozen base tensors even after adapter wrapping.
        assert all(tensor_hash(p)==base_hashes[name] for name,p in source_to_loaded.items())
        baseline=[p for p in report['predictions'] if p['phase']=='baseline' and p['split']=='TEST']
        final=[p for p in report['predictions'] if p['phase']=='final' and p['split']=='TEST']
        report.update(completed=True,verdict='LORA_CANDIDATE_TRAINED_NOT_PRODUCTION_OR_RETENTION_QUALIFIED',
            base_weights_exactly_unchanged=True,optimizer_steps=attempts,forward_calls=forward_calls,
            baseline_TEST_exact=sum(p['exact_JSON_repair'] for p in baseline)/len(baseline),
            final_TEST_exact=sum(p['exact_JSON_repair'] for p in final)/len(final))
    except Exception as exc:
        report['failures'].append(type(exc).__name__+': '+str(exc));report['verdict']='LORA_PILOT_STOPPED_NOT_PASS'
        (run/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        if model is not None and adapter_initialized:
            try:
                partial=ROOT/'artifacts'/run.name/'unqualified-partial-adapter';partial.mkdir(parents=True)
                model.save_pretrained(partial,safe_serialization=True)
                report['partial_adapter_path']=str(partial.relative_to(ROOT))
            except Exception as save_error:report['partial_save_error']=str(save_error)
    finally:
        report.update(elapsed_seconds=time.time()-begun,optimizer_attempts=attempts,forward_calls=forward_calls)
        write(run/'result.json',report);event(run,'LoRA pilot terminal',{k:v for k,v in report.items() if k not in ('predictions','updates')})
    print(json.dumps({'run':str(run),**{k:v for k,v in report.items() if k not in ('predictions','updates')}},ensure_ascii=False),flush=True)
    return 0 if report['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
