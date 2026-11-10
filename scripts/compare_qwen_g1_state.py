"""Actual bounded Qwen4B reference on48public prompts, no serving proxy."""
from pathlib import Path
import sys,json,os,time,socket,subprocess,urllib.request,urllib.error,hashlib
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import psutil
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    if any((p.info['name'] or '').lower()=='llama-server.exe' for p in psutil.process_iter(['name'])):raise ValueError('Another llama-server is running; not starting reference')
    cap=apply_memory_limit(4_500_000_000)
    info=json.loads((root/'artifacts/g1-state-prompts-reviewed.json').read_text(encoding='utf-8'));raw=(root/info['data']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==info['sha256'];cases=json.loads(raw)['items']
    ref=json.loads((root/'artifacts/qwen4b-reference-download.json').read_text(encoding='utf-8'));assert ref['available'] and ref['sha256_verified']
    modelpath=Path(ref['local_path']);assert modelpath.stat().st_size==ref['expected_bytes']
    run=new_run(root,'qwen-g1-state-reference-v1',{'protocol':'experiments/qwen-g1-state-reference-v1.md',
        'display_title':'Qwen4B đối chiếu48prompt mới · reference thật, sẽ tắt sau lượt',
        'model_scope':'Isolated Qwen reference; explicitly up to4.5GB, not SCAR-X under2GB or a serving proxy',
        'cases':48,'max_tokens':64,'threads':2,'temperature':0,'new_training':False},data_description='Exact48public synthetic reviewed prompts, no private transcript or prior modeloutputs sent to Qwen',data_files=[info['data']])
    job=root/'local/qwen-g1-state'/run.name;job.mkdir(parents=True)
    state_info=json.loads((root/'artifacts/g1-conversational-state-latest.json').read_text(encoding='utf-8'))
    private=root/'local/g1-state-audit'/Path(state_info['run']).name/'current_private-outputs.json'
    measured=json.loads(private.read_text(encoding='utf-8'))
    measured={r['case_id']:r for r in measured if r['state_condition']=='canonicalreset' and not r['case_id'].startswith('known-')}
    exe=Path(os.environ['LOCALAPPDATA'])/'hermes/runtimes/llamacpp/b10679/cpu/llama-server.exe'
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    command=[str(exe),'-m',str(modelpath),'--host','127.0.0.1','--port',str(port),'-c','1024','-t','2','-tb','2','-b','64','-ub','64','-ngl','0','-np','1','--no-mmap','--no-webui']
    proc=None;outcomes=[];error=None;began=time.perf_counter()
    with ResourceMonitor(limit=4_500_000_000,interval=.1) as monitor,(job/'stdout.log').open('wb') as out,(job/'stderr.log').open('wb') as err:
        try:
            proc=subprocess.Popen(command,cwd=job,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if os.name=='nt':psutil.Process(proc.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            base=f'http://127.0.0.1:{port}'
            while True:
                try:json.load(urllib.request.urlopen(base+'/health',timeout=1));break
                except (OSError,urllib.error.HTTPError):
                    if proc.poll() is not None:raise RuntimeError('Qwen reference startup exited')
                    if time.perf_counter()-began>90:raise TimeoutError('Qwen startup90s')
                    time.sleep(.2)
            for case in cases:
                if time.perf_counter()-began>300:raise TimeoutError('Reference300s deadline')
                payload={'model':'qwen4b-reference','messages':[{'role':'user','content':case['prompt']}],'max_tokens':64,'temperature':0,'seed':919,'stream':False}
                request=urllib.request.Request(base+'/v1/chat/completions',json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                started=time.perf_counter();response=json.load(urllib.request.urlopen(request,timeout=30));choice=response['choices'][0]
                timestamp=datetime.now(timezone.utc).isoformat();text=choice['message'].get('content') or ''
                row={'id':case['id'],'prompt':case['prompt'],'one_style_target':case['target'],'generated':text,
                    'finish_reason':choice.get('finish_reason'),'usage':response.get('usage'),'seconds':time.perf_counter()-started,'measured_at_utc':timestamp}
                outcomes.append(row)
                with (run/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                previous=measured[case['id']];assert previous['prompt']==case['prompt']
                append(run,'answer',{'scope':'Archived SCAR-X current-snapshot/canonicalreset vs freshly measured Qwen4B',
                    'case_id':case['id'],'prompt':case['prompt'],'target':case['target'],'generated':previous['generated'],
                    'seconds':previous['seconds'],'neural_ticks':previous['neural_ticks'],'active':previous['active'],'allocated':previous['allocated'],
                    'checkpoint':'Archived private SCAR-Xsnapshot in local/; not a new serving response',
                    'input_scope':'Same exact single public user prompt. SCAR-X measured earlier on canonicalreset clone; no private context sent to Qwen.',
                    'qwen':{'generated':text,'finish_reason':row['finish_reason'],'run':run.relative_to(root).as_posix(),'measured_at_utc':timestamp,
                        'scope':'Actual Qwen3-4B reference response; no tools. Different training/memory budgets; not truth oracle or SCAR-Xbackend.'}})
                progress(run,'Qwen đối chiếu thực · '+case['id'],len(outcomes),48,metrics={'generated_characters':len(text),'finish_reason':row['finish_reason'],'response_seconds':row['seconds'],'reference_ram_target_bytes':4500000000})
                monitor.check()
        except Exception as exc:error=type(exc).__name__+': '+str(exc)[:180]
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        result={'run':run.relative_to(root).as_posix(),'data':info['data'],'model':ref['base_model'],'model_sha256':ref['expected_sha256'],
            'decision':'QWEN_REFERENCE_MEASURED' if error is None and len(outcomes)==48 else 'QWEN_REFERENCE_INCOMPLETE',
            'outputs':outcomes,'completed':len(outcomes),'expected':48,'truncated':sum(r['finish_reason']=='length' for r in outcomes),
            'server_stopped':bool(proc is None or proc.poll() is not None),'error':error,'resources':monitor.report(),'cap':cap,
            'scarx_uses_qwen_proxy':False,'matched_pretraining_or_ram':False,'hermes_configuration_changed':False,
            'private_input_sent':False,'new_training':False,'normal_chat_or_vision_equivalence':False}
        (run/'report.md').write_text('# Actual Qwen4B reference on48public utterances\n\nNo private prompts, tools, retrieval or SCAR-Xproxy. Qwenmemory measured separately aboveSCAR-Xtarget. Exact matching toone style target is not a conversationqualitygrade.\n\n'+json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/qwen-g1-state-reference-latest.json',result)
        print(json.dumps({'run':str(run),'completed':len(outcomes),'server_stopped':result['server_stopped'],'error':error}),flush=True)


if __name__=='__main__':main()
