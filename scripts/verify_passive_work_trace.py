"""Publish actual source/check activity and a serving inference; no UI controls."""
import hashlib,json,subprocess,sys,time,urllib.request
from pathlib import Path

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    run=new_run(root,'passive-work-trace-acceptance-v1',{
        'display_title':'Trace tự động · cập nhật kỹ thuật và kiểm tra serving thật',
        'observability_kind':'software_acceptance',
        'model_scope':'Kiểm tra phần mềm riêng; một inference thật trên lõi live, không huấn luyện mới',
        'protocol':'experiments/passive-work-trace-acceptance-v1.md'})
    (run/'report.md').write_text('# Passive work trace acceptance\n\nActual software checks and one serving inference. Work log is separate from neural activity. Result pending; not a model-quality benchmark.\n',encoding='utf-8')
    names=['src/scarx/runtime/work_trace.py','src/scarx/app/observatory.py',
        'src/scarx/app/generation_web/index.html','src/scarx/app/generation_web/research.js',
        'src/scarx/app/generation_web/research.css']
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in names}
    activity(run,'Đã cập nhật trace thụ động','Thêm nhật ký kỹ thuật, tách khỏi câu trả lời và số đo học; các phần tự cập nhật mỗi 1,5 giây.',evidence={'sha256':hashes})
    began=time.perf_counter();activity(run,'Kiểm tra toàn bộ phần mềm','Đang chạy pytest trong subprocess; đây là kiểm thử phần mềm, không là kết quả huấn luyện model.',status='started')
    progress(run,'Đang chạy pytest; không cần thao tác trên website',0,2)
    completed=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
    output=completed.stdout.decode('utf-8',errors='replace');(run/'pytest.log').write_text(output,encoding='utf-8')
    summary='\n'.join(output.strip().splitlines()[-3:])
    activity(run,'Kết quả kiểm thử phần mềm',summary,status='completed' if completed.returncode==0 else 'failed',evidence={'exit_code':completed.returncode,'seconds':round(time.perf_counter()-began,3),'output_sha256':hashlib.sha256(completed.stdout).hexdigest()})
    if completed.returncode:raise RuntimeError('Pytest failed; actual output retained in run/pytest.log')
    base='http://127.0.0.1:8765'
    def get(path):return json.load(urllib.request.urlopen(base+path,timeout=30))
    before=get('/api/status');token=before['token']
    if get('/api/trace')['controller']['queued']:raise RuntimeError('Live teaching active; cannot assert read-only update counter')
    payload=json.dumps({'dataset':'g1_state','case_id':'greeting_vi_01'}).encode()
    request=urllib.request.Request(base+'/api/observe-answer',payload,headers={'Content-Type':'application/json','X-SCARX-Token':token})
    generated=json.load(urllib.request.urlopen(request,timeout=30))
    progress(run,'Đã gọi model live thật; kiểm tra GET chỉ đọc',1,2)
    traces=[get('/api/trace') for _ in range(3)];after=get('/api/status')
    answer=next(r for r in traces[-1]['answers'] if r.get('case_id')=='g1-state/greeting_vi_01')
    assert answer['generated']==generated.get('neural_draft',generated['text']) and answer['neural_ticks']>0
    assert answer['qwen']['run'].startswith('runs/') and answer.get('target')
    assert all(t['activities'] and t['controller']['updates']==before['updates'] and not t['controller']['queued'] for t in traces)
    assert after['updates']==before['updates']
    measured=[r for r in traces[-1]['learning'] if isinstance(r.get('loss_mean'),(int,float))]
    assert measured,'Existing real learning measurements must remain readable'
    evidence={'case_id':answer['case_id'],'neural_ticks':answer['neural_ticks'],'updates_before':before['updates'],
        'updates_after':after['updates'],'existing_measured_learning_records':len(measured),'ram_bytes':after['ram_bytes']}
    assert after['ram_bytes']<2_000_000_000
    activity(run,'Đã xác nhận luồng tự cập nhật','Một inference serving thật đã xuất hiện kèm đáp án mẫu và Qwen đã đo. Ba lần đọc trace không tạo bài học hoặc cập nhật trọng số; log học thật cũ vẫn còn.',evidence=evidence)
    progress(run,'Hoàn tất kiểm tra API; không cần chọn hoặc bấm chạy',2,2,metrics=evidence)
    result={'run':run.relative_to(root).as_posix(),'decision':'PASS_SOFTWARE_ACCEPTANCE_ONLY','live_model_replaced':False,
        'normal_chat_or_vision_proven':False,'new_training':False,'checks':evidence,'pytest_summary':summary,
        'pytest_exit_code':completed.returncode,'source_sha256':hashes,'browser_check':'separate Computer Use observation required'}
    (run/'report.md').write_text('# Passive work trace acceptance\n\n'+summary+'\n\nOne real serving inference, three read-only trace fetches, existing actual learning measurements retained. No new training or model-quality claim. See result.json and pytest.log. Browser verification is separate.\n',encoding='utf-8')
    write_json(run/'result.json',result);write_json(root/'artifacts/passive-work-trace-latest.json',result)
    print(json.dumps({'run':result['run'],'decision':result['decision'],'pytest':summary,'checks':evidence},ensure_ascii=False))


if __name__=='__main__':main()
