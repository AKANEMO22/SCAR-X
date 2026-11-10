"""ActualQwen4B64sameDEVprompts; never traininglabels/mainproxy."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,socket,urllib.request,urllib.error
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import psutil
from direct_words_common import source_data
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress,_process
from scarx.runtime.work_trace import append,activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    scarx=json.loads((root/'artifacts/direct-word-bridge-latest.json').read_text(encoding='utf-8'));j=json.loads((root/'local/experiment-status'/(Path(scarx['run']).name+'.json')).read_text(encoding='utf-8'))
    assert j['status']=='completed' and not _process(j)[0] and len(scarx['trials'])==15 and scarx['all_cold_exact']
    if any((p.info['name'] or '').lower()=='llama-server.exe' for p in psutil.process_iter(['name'])):raise ValueError('Other llama-server exists; leave it untouched')
    receipt,_,cases,_,files=source_data();assert len(cases)==64
    # Fixedseed/method BEFORE examining its quality, no bestmodel selection.
    source_file=root/scarx['run']/'word-36313-correct.json';saved=json.loads(source_file.read_text(encoding='utf-8'));previous={r['id']:r for r in saved['outputs']['development']};assert len(previous)==64
    descriptor=json.loads((root/'artifacts/qwen4b-reference-download.json').read_text(encoding='utf-8'));modelpath=Path(descriptor['local_path']);assert modelpath.stat().st_size==descriptor['expected_bytes']
    with modelpath.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==descriptor['expected_sha256']
    cap=apply_memory_limit(4_500_000_000);run=new_run(root,'qwen-direct-word-reference-v1',{'protocol':'experiments/direct-word-bridge-v1.md','display_title':'Qwen4B · actual64sameDEVprompts, nolearning',
        'observability_kind':'external_reference','model_scope':'IndependentCPU4.5GBreference ABOVE SCAR-X2GB. Not labels/proxy/mainmodel, allwordtrials already terminal.',
        'cases':64,'temperature':0,'seed':919,'max_tokens':64,'context':1024,'threads':2,'new_training':False,'scarx_pair_seed':36313,'scarx_pair_method':'correct'},
        data_description='64reviewedpublicDEVsingleuserprompts ONLY; no targets/family/SCAR-Xoutputs or privatecontext sent to Qwen.',data_files=files+[source_file.relative_to(root).as_posix()])
    write_json(run/'answer-source.json',{'source_run':scarx['run'],'scope':'Archived actualneuralwords, not newinference/Qwen-trainedSCAR-X'})
    private=root/'local/qwen-direct-words'/run.name;private.mkdir(parents=True);exe=Path(os.environ['LOCALAPPDATA'])/'hermes/runtimes/llamacpp/b10679/cpu/llama-server.exe';assert exe.is_file()
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    command=[str(exe),'-m',str(modelpath),'--host','127.0.0.1','--port',str(port),'-c','1024','-t','2','-tb','2','-b','64','-ub','64','-ngl','0','-np','1','--no-mmap','--no-webui']
    (run/'report.md').write_text('# ActualQwen4B reference\n\nIndependent64sameDEVprompts, progress. Not labels/proxy/equalRAM or exactstylegrade.\n',encoding='utf-8')
    activity(run,'Đang chạyQwen4B đối chiếu64câu thật','Chỉsinglepublicprompt, khôngnhãn/family/câuSCAR-X làminput. Seed36313/correct paircốđịnh, actualneuraloutputs đãlưu. Qwenbudget4.5GB riêng, không mainproxy/9B.',status='started',evidence={'expected':64,'scarx_pair_source':source_file.relative_to(root).as_posix(),'reference_cap_bytes':4500000000})
    process=None;outcomes=[];error=None;began=time.perf_counter()
    with ResourceMonitor(limit=4_500_000_000,interval=.1) as monitor,(private/'stdout.log').open('wb') as out,(private/'stderr.log').open('wb') as err:
        try:
            process=subprocess.Popen(command,cwd=private,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if os.name=='nt':psutil.Process(process.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            base=f'http://127.0.0.1:{port}'
            while True:
                try:json.load(urllib.request.urlopen(base+'/health',timeout=1));break
                except (OSError,urllib.error.HTTPError):
                    if process.poll() is not None:raise RuntimeError('OwnedQwenstartup exited')
                    if time.perf_counter()-began>90:raise TimeoutError('Startup90s')
                    time.sleep(.2)
            for case in cases:
                if time.perf_counter()-began>600:raise TimeoutError('Qwen600sreferencecap')
                payload={'model':'qwen4b-reference','messages':[{'role':'user','content':case['prompt']}],'temperature':0,'seed':919,'max_tokens':64,'stream':False}
                request=urllib.request.Request(base+'/v1/chat/completions',json.dumps(payload).encode(),headers={'Content-Type':'application/json'});start=time.perf_counter();response=json.load(urllib.request.urlopen(request,timeout=30));choice=response['choices'][0]
                r={'id':case['id'],'prompt':case['prompt'],'generated':choice['message'].get('content') or '','finish_reason':choice.get('finish_reason'),'usage':response.get('usage'),'seconds':time.perf_counter()-start,'measured_at_utc':datetime.now(timezone.utc).isoformat()};outcomes.append(r)
                with (run/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
                old=previous[case['id']];assert old['prompt']==case['prompt'] and old['target']==case['target']
                append(run,'answer',{'scope':'DIRECT WORD BRIDGE ARCHIVE · seed36313/correct vs actualQwen4B · NOT MAIN',
                    'case_id':case['id'],'prompt':case['prompt'],'target':case['target'],'generated':old['generated'],'seconds':old['seconds'],'neural_ticks':old['actual_ticks'],
                    'active':48,'allocated':48,'checkpoint':'Archiveddirectwordseed36313/correct, nevermainimport',
                    'input_scope':'SamePUBLICsingleprompt, SCAR-X earlierneuralwords and independentfreshQwen. Differenttraining/RAM/decoderbudget, no exactstylegrade/equivalence.',
                    'qwen':{'generated':r['generated'],'finish_reason':r['finish_reason'],'run':run.relative_to(root).as_posix(),'measured_at_utc':r['measured_at_utc'],
                        'scope':'ActualQwen3-4B CPUref/temp0/64tokens/ctx1024, no tools/history/target. Not SCAR-Xproxy/truthlabel.'}})
                progress(run,'Qwenactualreference '+case['id'],len(outcomes),64,metrics={'finish_reason':r['finish_reason'],'seconds':r['seconds'],'reference_cap_bytes':4500000000});monitor.check()
                if len(outcomes)%8==0:print(json.dumps({'completed':len(outcomes),'expected':64,'truncated':sum(x['finish_reason']=='length' for x in outcomes)}),flush=True)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)[:180]
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        result={'run':run.relative_to(root).as_posix(),'source_run':scarx['run'],'decision':'QWEN_DIRECT_WORD_REFERENCE_MEASURED' if error is None and len(outcomes)==64 else 'QWEN_DIRECT_WORD_REFERENCE_INCOMPLETE',
            'outputs':outcomes,'completed':len(outcomes),'expected':64,'truncated':sum(x['finish_reason']=='length' for x in outcomes),'model':descriptor['base_model'],'model_sha256':descriptor['expected_sha256'],
            'temperature':0,'seed':919,'context':1024,'max_tokens':64,'server_stopped':process is None or process.poll() is not None,'error':error,'resources':monitor.report(),'cap':cap,
            'new_training':False,'scarx_uses_qwen_proxy':False,'Qwen_outputs_used_as_training_labels':False,'hermes_configuration_changed':False,'normal_chat_or_vision_equivalence':False,'private_input_sent':False}
        (run/'report.md').write_text('# ActualQwen4B64publicpromptreference\n\nRawoutputs/truncationretained; not traininglabels, exactstylegrade or equalcost. Ownedserver stopped.\n\n'+json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/qwen-direct-word-reference.json',result)
        activity(run,'Hoàn tấtQwen4B đối chiếu độc lập',result['decision']+'; '+str(len(outcomes))+'/64, truncated'+str(result['truncated'])+', ownedserverstopped. KhônghọctừQwenlabels.',evidence={'completed':len(outcomes),'expected':64,'server_stopped':result['server_stopped'],'reference_run':result['run']})
        print(json.dumps({k:result[k] for k in ('run','decision','completed','truncated','server_stopped','error','resources')},ensure_ascii=False))


if __name__=='__main__':main()
