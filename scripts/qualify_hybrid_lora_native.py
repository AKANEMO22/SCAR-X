"""Actual same-native-process scale0/1 LoRA qualification, no live modification."""
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import secrets,subprocess,sys,time,uuid
from urllib.request import Request,urlopen
import psutil

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.store import Store,encode


def controls():
    result=[]
    for i,(q,a) in enumerate([('37+28','65'),('91-17','74'),('6*9','54'),('81/3','27')]):
        result.append({'id':f'math-{i}','family':'math','messages':[{'role':'user','content':'Calculate '+q+'. Reply with the integer result only.'}],'expected':a})
    for i,(word,n) in enumerate([('lamp',7),('boat',3),('tree',8),('book',2)]):
        result.append({'id':f'json-{i}','family':'JSON','messages':[{'role':'user','content':f'Return only JSON with exactly keys "item" and "count": item is "{word}" and count is integer {n}.'}],'expected':{'item':word,'count':n}})
    for i,(subject,value) in enumerate([('parcel','amber'),('ticket','violet'),('crate','silver'),('folder','green')]):
        result.append({'id':f'source-{i}','family':'source','messages':[{'role':'user','content':f'Observation: The {subject} has color {value}. What color is the {subject}? Reply with one word only.'}],'expected':value})
    return result


def main():
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-lora-native-portability-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    with (ROOT/'local/lora-native-portability-v1-execution.json').open('x',encoding='utf-8') as f:json.dump({'run':str(run)},f)
    (ROOT/'local/current-lora-native-portability.txt').write_text(str(run),encoding='utf-8')
    data_dir=Path((ROOT/'local/current-qualified-repair-data.txt').read_text(encoding='utf-8').strip());rows=json.loads((data_dir/'verified-data.json').read_text(encoding='utf-8'))
    probes=[{'id':r['id'],'family':'repair','split':r['split'],'messages':r['messages'],'expected':json.loads(r['target'])} for r in [r for r in rows if r['split']=='DEV'][:4]+[r for r in rows if r['split']=='TEST']]+controls()
    (run/'locked-probes.json').write_text(encode(probes),encoding='utf-8')
    adapter=ROOT/'artifacts/20261115T074627Z-native-lora-repair-v1-5c09fa/adapter-f32.gguf'
    assert hashlib.sha256(adapter.read_bytes()).hexdigest()=='2c92092edaf5781f91a6c788e56787bc5a15b0456dbe041898d865b0bf674c12'
    native=SharedNative(ROOT);native.check();args=native.worker.cmdline().copy();token=secrets.token_hex(24)
    def arg(flag,value):args[args.index(flag)+1]=str(value)
    arg('--port',18786);arg('--api-key',token)
    if '--alias' in args:arg('--alias','scarx-lora-native-qualification')
    args+=['--lora',str(adapter),'--lora-init-without-apply']
    (run/'manifest.json').write_text(encode({'source_git_sha':os.environ.get('SCARX_NATIVE_LORA_SOURCE_SHA'),
        'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('scripts/qualify_hybrid_lora_native.py','experiments/hybrid-lora-native-portability-v1.md')},
        'native_exe_sha256':hashlib.sha256(Path(args[0]).read_bytes()).hexdigest(),
        'adapter_sha256':hashlib.sha256(adapter.read_bytes()).hexdigest(),'max_calls':64,'max_seconds':900,
        'live_native_pid_unchanged':native.worker.pid,'scale_values':[0,1],'base_native_precision':'unchanged BF16/F32'}),encoding='utf-8')
    process=None;owner_created=None;trace=Store(ROOT/'local/hybrid-v1');started=time.time();records=[]
    report={'run_id':run.name,'completed':False,'failures':[],'training_calls':0,'ALIF_updates':0,'live_adapter_changed':False,'production_or_retention_PASS':False}
    def physical():
        out=subprocess.check_output([str(Path(os.environ['WINDIR'])/'System32/nvidia-smi.exe'),'--id=0','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=3,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        free,total=[int(v.strip())*1024**2 for v in out.strip().split(',')]
        if free<512*1024**2:raise MemoryError('Dedicated reserve below512MiB')
        return {'dedicated_free_bytes':free,'dedicated_total_bytes':total}
    def call(path,body=None):
        with urlopen(Request('http://127.0.0.1:18786'+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}),timeout=60) as r:return json.load(r)
    try:
        physical()
        with (run/'native.stdout.log').open('wb') as out,(run/'native.stderr.log').open('wb') as err:
            process=subprocess.Popen(args,cwd=ROOT,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        owner_created=psutil.Process(process.pid).create_time();deadline=time.time()+90
        while True:
            if process.poll() is not None:raise RuntimeError('Owned qualification worker exited during load')
            try:
                if call('/health').get('status')=='ok':break
            except Exception:
                if time.time()>deadline:raise TimeoutError('Native loading readiness')
                time.sleep(.25)
        adapters=call('/lora-adapters');(run/'adapter-list.json').write_text(encode(adapters),encoding='utf-8')
        for probe in probes:
            for scale in (0,1):
                if time.time()-started>900:raise TimeoutError('Portability budget')
                while native.status()['cognition']['phase']=='serving':
                    if time.time()-started>900:raise TimeoutError('Foreground retains GPU priority')
                    time.sleep(.25)
                native.check();resources=physical()
                with (run/'attempts.jsonl').open('a',encoding='utf-8') as f:f.write(encode({'attempt':len(records)+1,'probe':probe['id'],'scale':scale,'epoch':time.time()})+'\n')
                began=time.time();raw=call('/v1/chat/completions',{'model':'scarx-lora-native-qualification','messages':probe['messages'],
                    'max_tokens':64,'temperature':0,'repeat_penalty':1.05,'chat_template_kwargs':{'enable_thinking':False},'lora':[{'id':0,'scale':scale}]})
                text=raw['choices'][0]['message'].get('content') or ''
                if isinstance(probe['expected'],dict):
                    try:correct=json.loads(text)==probe['expected']
                    except ValueError:correct=False
                else:correct=text.strip().casefold().rstrip('.')==probe['expected'].casefold()
                row={'probe':probe['id'],'family':probe['family'],'split':probe.get('split'),'scale':scale,'actual_answer':text,'expected':probe['expected'],
                    'exact_correct':correct,'seconds':time.time()-began,'resources':resources,'raw_result':raw}
                records.append(row)
                trace.event('work_trace.activity',{'kind':'actual_native_LoRA_qualification','title':'Native LoRA '+probe['id']+' scale'+str(scale),
                    'run_id':run.name,'question':probe['messages'][-1]['content'],'actual_answer':text,'expected':probe['expected'],'scale':scale,'correct':correct,'SNN_updates':0},'orchestrator')
        report.update(completed=True,verdict='NATIVE_PORTABILITY_MEASURED_NOT_AUTOMATIC_PROMOTION',calls=len(records),
            by_family={family:{str(scale):sum(r['exact_correct'] for r in records if r['family']==family and r['scale']==scale) for scale in (0,1)} for family in ('repair','math','JSON','source')})
    except Exception as exc:report['failures'].append(type(exc).__name__+': '+str(exc));report['verdict']='NATIVE_PORTABILITY_STOPPED_NOT_PASS'
    finally:
        if process is not None and process.poll() is None:
            owned=psutil.Process(process.pid)
            if owned.create_time()==owner_created and Path(owned.exe()).resolve()==Path(args[0]).resolve():
                process.terminate();process.wait(timeout=15) # Only this exact disposable worker.
        native.check();report.update(elapsed_seconds=time.time()-started,records=records,
            owned_worker_cleanup_complete=process is None or process.poll() is not None)
        (run/'result.json').write_text(encode(report),encoding='utf-8')
        trace.event('work_trace.activity',{'title':'Native LoRA portability terminal','run_id':run.name,**{k:v for k,v in report.items() if k!='records'}},'orchestrator');trace.close()
    print(encode({'run':str(run),**{k:v for k,v in report.items() if k!='records'}}),flush=True)


if __name__=='__main__':main()
