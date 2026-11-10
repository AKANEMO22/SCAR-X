"""Fill47missing public references AFTER Torchcontrols stop; never a proxy."""
from pathlib import Path
import os,sys,json,time,hashlib,socket,subprocess,urllib.request,urllib.error
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import psutil
from train_g1_word_score_inputs import source_data
from generate_g1_state_paraphrases import INTENTS
from scarx.app.observatory import archived_single_prompt_references
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress,_process
from scarx.runtime.work_trace import append,activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    control=json.loads((root/'artifacts/g1-word-learnability-control-latest.json').read_text(encoding='utf-8'))
    if len(control.get('trials',[]))!=6 or not control.get('all_cold_exact'):raise ValueError('Complete all6controls first')
    journal=json.loads((root/'local/experiment-status'/(Path(control['run']).name+'.json')).read_text(encoding='utf-8'))
    if journal['status']!='completed' or _process(journal)[0]:raise ValueError('Torchcontrol coordinator must be verified terminal first')
    if any((p.info['name'] or '').lower()=='llama-server.exe' for p in psutil.process_iter(['name'])):raise ValueError('Another llama-server is running; leave it untouched')
    train,dev,prior,_,sources=source_data();diagnostics=[]
    for intent in INTENTS:diagnostics += [r for r in train if r['intent']==intent][:2]
    cases=dev+prior+diagnostics;references=archived_single_prompt_references(root)
    missing=[r for r in cases if r['prompt'] not in references]
    assert len(cases)==len({r['prompt'] for r in cases})==95 and len(missing)==47
    source_file=root/control['run']/'control-28001-known-digest128-v1.json'
    saved=json.loads(source_file.read_text(encoding='utf-8'));record=saved['record']
    assert record['seed']==28001 and record['input_encoding']=='known-digest128-v1' and record['method']=='isolated_surrogate_bptt'
    previous={r['prompt']:r for rows in saved['outputs'].values() for r in rows}
    assert all(previous[c['prompt']]['target']==c['target'] for c in missing)
    ref=json.loads((root/'artifacts/qwen4b-reference-download.json').read_text(encoding='utf-8'));modelpath=Path(ref['local_path'])
    assert modelpath.stat().st_size==ref['expected_bytes']
    with modelpath.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==ref['expected_sha256']
    cap=apply_memory_limit(4_500_000_000)
    run=new_run(root,'qwen-word-control-reference-v1',{'protocol':'experiments/qwen-word-control-reference-v1.md',
        'display_title':'Qwen4B reference thật · bù47publicprompts còn thiếu',
        'observability_kind':'external_reference','model_scope':'Separate CPUreference up to4.5GB; not mainSCAR-X/2GB or a servingproxy. Torchcontrols already stopped.',
        'cases':47,'max_tokens':64,'context':1024,'threads':2,'temperature':0,'seed':919,'new_training':False},
        data_description='Only31dev+16train authored/reviewed publicprompts, no privatehistory/guards or modelweights sent.',data_files=sources+[source_file.relative_to(root).as_posix()])
    write_json(run/'cases.json',missing);write_json(run/'answer-source.json',{'scarx_control_run':control['run'],'scope':'Archived actualcontrol outputs; not new SNNinference'})
    directory=root/'local/qwen-word-controls'/run.name;directory.mkdir(parents=True)
    exe=Path(os.environ['LOCALAPPDATA'])/'hermes/runtimes/llamacpp/b10679/cpu/llama-server.exe'
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    command=[str(exe),'-m',str(modelpath),'--host','127.0.0.1','--port',str(port),'-c','1024','-t','2','-tb','2','-b','64','-ub','64','-ngl','0','-np','1','--no-mmap','--no-webui']
    process=None;outcomes=[];error=None;began=time.perf_counter()
    (run/'report.md').write_text('# Actual Qwen reference\n\nReference in progress, isolated4.5GB budget, no mainmodelchange.\n',encoding='utf-8')
    with ResourceMonitor(limit=4_500_000_000,interval=.1) as monitor,(directory/'stdout.log').open('wb') as stdout,(directory/'stderr.log').open('wb') as stderr:
        try:
            process=subprocess.Popen(command,cwd=directory,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if os.name=='nt':psutil.Process(process.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            base=f'http://127.0.0.1:{port}'
            while True:
                try:json.load(urllib.request.urlopen(base+'/health',timeout=1));break
                except (OSError,urllib.error.HTTPError):
                    if process.poll() is not None:raise RuntimeError('Owned Qwen startup exited')
                    if time.perf_counter()-began>90:raise TimeoutError('Startup90s cap')
                    time.sleep(.2)
            for case in missing:
                if time.perf_counter()-began>360:raise TimeoutError('Reference360s cap')
                payload={'model':'qwen4b-reference','messages':[{'role':'user','content':case['prompt']}],'temperature':0,'seed':919,'max_tokens':64,'stream':False}
                request=urllib.request.Request(base+'/v1/chat/completions',json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                t0=time.perf_counter();response=json.load(urllib.request.urlopen(request,timeout=30));choice=response['choices'][0];generated=choice['message'].get('content') or ''
                timestamp=datetime.now(timezone.utc).isoformat();row={'id':case['id'],'prompt':case['prompt'],'one_style_target':case['target'],'generated':generated,
                    'finish_reason':choice.get('finish_reason'),'usage':response.get('usage'),'seconds':time.perf_counter()-t0,'measured_at_utc':timestamp}
                outcomes.append(row)
                with (run/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                old=previous[case['prompt']]
                append(run,'answer',{'scope':'ISOLATED CONTROL ARCHIVE seed28001/dense vs actual freshQwen4B · NOT MAIN','control_only':True,
                    'case_id':case['id'],'prompt':case['prompt'],'target':case['target'],'generated':old['generated'],'seconds':old['seconds'],
                    'neural_ticks':old['free_generation_ticks'],'active':48,'allocated':48,'checkpoint':'Archived ALIF-BPTT control, never imported into main',
                    'input_scope':'Same single public usertext; ALIF-BPTT measured earlier, Qwenfresh independentCPUreference. Not matchedtraining/memory or truthlabels.',
                    'qwen':{'generated':generated,'finish_reason':row['finish_reason'],'run':run.relative_to(root).as_posix(),'measured_at_utc':timestamp,
                        'scope':'Actual isolated Qwen3-4B, context1024/max_tokens64/temp0, no system/history/tools; not SCAR-Xbackend or truthoracle.'}})
                progress(run,'Qwen actualreference · '+case['id'],len(outcomes),47,metrics={'finish_reason':row['finish_reason'],'seconds':row['seconds'],'reference_memory_limit':4500000000});monitor.check()
        except Exception as e:error=type(e).__name__+': '+str(e)[:180]
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        result={'run':run.relative_to(root).as_posix(),'scarx_control_run':control['run'],'decision':'QWEN_REFERENCE_MEASURED' if error is None and len(outcomes)==47 else 'QWEN_REFERENCE_INCOMPLETE',
            'outputs':outcomes,'completed':len(outcomes),'expected':47,'truncated':sum(r['finish_reason']=='length' for r in outcomes),
            'model':ref['base_model'],'model_sha256':ref['expected_sha256'],'max_tokens':64,'context':1024,'temperature':0,'seed':919,
            'server_stopped':process is None or process.poll() is not None,'error':error,'resources':monitor.report(),'cap':cap,
            'new_training':False,'scarx_uses_qwen_proxy':False,'hermes_configuration_changed':False,'private_input_sent':False,'normal_chat_or_vision_equivalence':False}
        (run/'report.md').write_text('# Actual Qwen reference on47missing publicprompts\n\nPaired with archived isolatedALIF-BPTT outputs, not mainSCAR-X. Same publicsingletext, different training/RAM/decoderbudget. Rawresponses not truthlabels, truncations retained; ownedserver stopped.\n\n'+json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/qwen-word-control-reference-latest.json',result)
        activity(run,'Hoàn tất Qwenreference độc lập',result['decision']+'; rawoutputs không thành nhãn học, server sở hữu đã tắt.',evidence={'completed':len(outcomes),'expected':47,'truncated':result['truncated'],'server_stopped':result['server_stopped']})
        print(json.dumps({'run':result['run'],'completed':len(outcomes),'expected':47,'truncated':result['truncated'],'server_stopped':result['server_stopped'],'error':error},ensure_ascii=False))


if __name__=='__main__':main()
