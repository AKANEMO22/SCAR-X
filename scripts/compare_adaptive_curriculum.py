"""Actual copied SCAR-X / isolated Qwen on fixed reviewed developmentwording."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,socket,sqlite3,io,zipfile,urllib.request,urllib.error
from datetime import datetime,timezone
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import psutil
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.product_acceptance import canonical_reply,weights_sha256
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import append,activity
from scarx.runtime.experiment_status import progress


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    if any((p.info['name'] or '').lower()=='llama-server.exe' for p in psutil.process_iter(['name'])):raise ValueError('Existing llama-server, leave untouched')
    receipt=json.loads((root/'artifacts/adaptive-curriculum-reviewed.json').read_text(encoding='utf-8'));path=(root/receipt['data']).resolve()
    assert path.is_relative_to(root/'data/candidates') and hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
    data=json.loads(path.read_text(encoding='utf-8'));assert data['all_rows_read'] and not data['automatic_learning_allowed']
    cases=[r for r in data['items'] if r['modality']=='chat' and r['id'].endswith(('_01','_02'))];assert len(cases)==24
    descriptor=json.loads((root/'artifacts/qwen4b-reference-download.json').read_text(encoding='utf-8'));modelpath=Path(descriptor['local_path'])
    assert modelpath.stat().st_size==descriptor['expected_bytes']
    with modelpath.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==descriptor['expected_sha256']
    exe=Path(os.environ['LOCALAPPDATA'])/'hermes/runtimes/llamacpp/b10679/cpu/llama-server.exe';assert exe.is_file()
    cap=apply_memory_limit(4_500_000_000)
    run=new_run(root,'adaptive-curriculum-reference-v1',{'protocol':'experiments/adaptive-curriculum-reference-v1.md','display_title':'24câu AGY mới · SCAR-X copy / Qwen4B thật · không học','observability_kind':'external_reference','model_scope':'Development only. SCAR-X copied privateweights/canonicalstate; Qwen independent4.5GB CPUref, no labels/proxy. No visionimagebenchmark.','cases':24,'temperature':0,'seed':919,'context':1024,'max_tokens':128,'threads':2,'new_training':False},data_description='Fixed24reviewedpublicchatwordings only. No privatecontent/labels/weights sent to Qwen.',data_files=[receipt['data']])
    private=root/'local/adaptive-reference'/run.name;private.mkdir(parents=True)
    with sqlite3.connect((root/'local/generation1/generation1.sqlite').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(private/'source.sqlite') as dest:source.backup(dest)
    with sqlite3.connect(private/'source.sqlite') as db:digest,payload=db.execute('SELECT sha,payload FROM runtime WHERE singleton=1').fetchone()
    assert hashlib.sha256(payload).hexdigest()==digest
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        raw=z.read('model.zip');metadata=json.loads(z.read('session.json'));assert hashlib.sha256(raw).hexdigest()==metadata['sha256']
    model=UnifiedCore.loads(raw);before=model.dumps();fingerprint=weights_sha256(model)
    activity(run,'Đo24câu chat mới trên SCAR-X copy/Qwen','Danh sách cố định01/02 trước đo; không học, không restore phiên thật. Qwen128token/ref4.5GB riêng, không labels. Không benchmarkvision bằng modeltext-only.',status='started',evidence={'protocol':'experiments/adaptive-curriculum-reference-v1.md','cases':24,'source_data':receipt['data']})
    (run/'report.md').write_text('# Actual24developmentreferences\n\nNo training or finaltest/equivalence claim.\n',encoding='utf-8')
    rows=[];process=None;error=None;start=time.perf_counter()
    with ResourceMonitor(limit=4_500_000_000,interval=.1) as monitor,(private/'stdout.log').open('wb') as out,(private/'stderr.log').open('wb') as err:
        try:
            for case in cases:
                began=time.perf_counter();answer=canonical_reply(model,case['prompt']);rows.append({'id':case['id'],'prompt':case['prompt'],'family':case['family'],'language':case['language'],'scarx':{'generated':answer['text'],'stop_reason':answer.get('stop_reason'),'seconds':time.perf_counter()-began,'scope':'Copiedcurrentmodel, independentcanonicalstate, no learning/retrieval.'}});monitor.check()
            assert model.dumps()==before and weights_sha256(model)==fingerprint
            write_json(run/'scarx-outputs.json',rows)
            with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            command=[str(exe),'-m',str(modelpath),'--host','127.0.0.1','--port',str(port),'-c','1024','-t','2','-tb','2','-b','64','-ub','64','-ngl','0','-np','1','--no-mmap','--no-webui']
            process=subprocess.Popen(command,cwd=private,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            psutil.Process(process.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print(json.dumps({'run':run.relative_to(root).as_posix(),'owned_qwen_pid':process.pid,'scarx_answers':len(rows)}),flush=True)
            base=f'http://127.0.0.1:{port}';startup=time.perf_counter()
            while True:
                try:json.load(urllib.request.urlopen(base+'/health',timeout=1));break
                except (OSError,urllib.error.HTTPError):
                    if process.poll() is not None or time.perf_counter()-startup>90:raise RuntimeError('Qwen startup failed/90s')
                    monitor.check();time.sleep(.2)
            for i,row in enumerate(rows):
                if time.perf_counter()-start>420:raise TimeoutError('Reference420s cap')
                request=urllib.request.Request(base+'/v1/chat/completions',json.dumps({'model':'qwen4b-reference','messages':[{'role':'user','content':row['prompt']}],'temperature':0,'seed':919,'max_tokens':128,'stream':False}).encode(),headers={'Content-Type':'application/json'})
                began=time.perf_counter();response=json.load(urllib.request.urlopen(request,timeout=30));choice=response['choices'][0]
                row['qwen']={'generated':choice['message'].get('content') or '','finish_reason':choice.get('finish_reason'),'seconds':time.perf_counter()-began,'measured_at_utc':datetime.now(timezone.utc).isoformat(),'usage':response.get('usage')}
                with (run/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                append(run,'answer',{'scope':'DEVELOPMENT24 · currentSCAR-X COPY / Qwen4B reference · NOT MAIN LEARNING','case_id':row['id'],'prompt':row['prompt'],'generated':row['scarx']['generated'],'seconds':row['scarx']['seconds'],'allocated':model.core.config.hidden,'checkpoint':'Copiedcurrentweights; source kept private, unchanged','input_scope':'FixednewAGYpublicsingleprompt. SCAR-X canonicalclone, independentQwen. No answerlabel/exactstylegrade or finaltest; Qwen identifies itself, not SCAR-X.','qwen':{**row['qwen'],'run':run.relative_to(root).as_posix(),'scope':'ActualQwen3-4B CPU/temp0/128tokens; different RAM/training budget, not labels/proxy.'}})
                progress(run,'Đã đo thật '+row['id'],i+1,24,metrics={'completed':i+1,'expected':24,'seconds':row['qwen']['seconds'],'new_training':False});monitor.check()
                if (i+1)%4==0:print(json.dumps({'completed':i+1,'expected':24}),flush=True)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)[:240]
        finally:
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        result={'run':run.relative_to(root).as_posix(),'decision':'ADAPTIVE_REFERENCE_MEASURED' if error is None and sum('qwen' in r for r in rows)==24 else 'ADAPTIVE_REFERENCE_INCOMPLETE','rows':rows,'completed':sum('qwen' in r for r in rows),'expected':24,'truncated':sum(r.get('qwen',{}).get('finish_reason')=='length' for r in rows),'source_model_unchanged':model.dumps()==before and weights_sha256(model)==fingerprint,'main_training':False,'main_restored':False,'qwen_used_as_labels_or_proxy':False,'private_content_sent':False,'quality_grade':None,'normal_chat_or_vision_proven':False,'server_stopped':process is None or process.poll() is not None,'error':error,'resources':monitor.report(),'cap':cap,'seconds':time.perf_counter()-start}
    write_json(run/'result.json',result);write_json(root/'artifacts/adaptive-curriculum-reference.json',result)
    (run/'report.md').write_text('# Actual24SCAR-X copy/Qwen developmentreferences\n\nNot labels/proxy or sameRAM/training budget. Sourcecopy unchanged.\n\n'+json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    activity(run,'Hoàn tất đối chiếu24câu mới',result['decision']+'; không dạy/restore main hoặc học từ Qwen, ownedreference đã dừng.',status='completed' if error is None else 'failed',evidence={k:result[k] for k in ('completed','expected','truncated','server_stopped','source_model_unchanged','error')})
    print(json.dumps({k:result[k] for k in ('run','decision','completed','truncated','server_stopped','error','seconds')}))
    return int(error is not None)


if __name__=='__main__':raise SystemExit(main())
