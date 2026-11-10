"""Actual held-reader atomic replace check and preserved incident receipt."""
from pathlib import Path
import sys,json,threading,time,uuid,os,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.runtime.atomic_file import replace_with_retry
from scarx.runtime.experiment_status import _process
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.work_trace import activity


def main():
    previous='20261109T192017Z-g1-word-learnability-control-v1-446c16';journal=json.loads((root/'local/experiment-status'/(previous+'.json')).read_text(encoding='utf-8'))
    assert journal['status']=='stopped_without_result' and not _process(journal)[0]
    original=root/'runs'/previous
    failure={'source_run':'runs/'+previous,'failure_kind':'windows_atomic_replace_sharing','error':'PermissionError WinError5 at experiment_status._write os.replace',
        'terminal_verified':True,'quality_evaluated':False,'last_worker_responses':448,'budget_consumed_seconds':journal['updated']-journal['started'],
        'technical_retry_count':1,'main_model_changed':False,'cause_scope':'Concurrent observer reader/atomic replace sharing is tested separately; no neural failure inferred.'}
    (root/'artifacts/g1-word-control-io-failure.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8')
    (original/'failure.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8')
    run=new_run(root,'g1-observer-atomic-recovery-v1',{'display_title':'Sửa ghi traceWindows · readerlock / atomicreplace',
        'observability_kind':'software_acceptance','model_scope':'Technical IO check only, no modellearning or quality measurement.'})
    directory=root/'local'/('atomic-replace-'+uuid.uuid4().hex[:8]);directory.mkdir();assert directory.resolve().is_relative_to((root/'local').resolve())
    target=directory/'journal.json';source=directory/'next.tmp';target.write_bytes(b'{"state":"old-complete"}');source.write_bytes(b'{"state":"new-complete"}')
    started=threading.Event();done=threading.Event();errors=[]
    def writer():
        started.set()
        try:replace_with_retry(source,target)
        except Exception as e:errors.append(repr(e))
        finally:done.set()
    with target.open('rb') as reader:
        thread=threading.Thread(target=writer);thread.start();assert started.wait(1)
        time.sleep(.04);while_reader_held=done.is_set()
        assert reader.read()==b'{"state":"old-complete"}'
    thread.join(2);assert done.is_set() and not errors and target.read_bytes()==b'{"state":"new-complete"}'
    result={'run':run.relative_to(root).as_posix(),'decision':'ATOMIC_REPLACE_READER_RECOVERY_VERIFIED','software_only':True,
        'real_held_reader_test':True,'replace_blocked_until_reader_closed':not while_reader_held,'complete_payload_preserved':True,
        'runtime_retry_bounded':True,'prior_failure':failure,'new_training':False,'normal_chat_or_vision_proven':False,
        'math_and_protocol_unchanged':True,'source_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['src/scarx/runtime/atomic_file.py','src/scarx/runtime/experiment_status.py','scripts/word_control_worker.py']}}
    (run/'report.md').write_text('# Actual Windows held-reader recovery\n\nAn open reader preserves the old complete payload while a bounded-retry writer publishes a complete new file after reader close. No modellearning.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    write_json(run/'result.json',result);write_json(root/'artifacts/g1-observer-atomic-recovery.json',result)
    activity(run,'Đã sửa và kiểm thật readerlock Windows','Boundedretry giữ ghi nguyên tử khi filetrace đang được đọc. Lỗi/448responsepartial cũ được giữ; không có devresult và không đổi toán họcneural.',evidence={'replace_blocked_until_reader_closed':not while_reader_held,'payload_complete':True,'prior_active_seconds':failure['budget_consumed_seconds']})
    print(json.dumps({'run':result['run'],'decision':result['decision'],'real_reader_blocked_replace':not while_reader_held,'prior_failed_active_seconds':failure['budget_consumed_seconds']}))


if __name__=='__main__':main()
