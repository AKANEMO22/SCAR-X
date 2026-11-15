"""Actual local GPU candidate reading/answer against the reported image.

No teaching labels, source-image export or user's original session mutation.
"""
from datetime import datetime,timezone
import base64,hashlib,json,sqlite3
from pathlib import Path
import sys,time,uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.image_question import prepare_image_question
from scarx.hybrid.store import Store


def main():
    image=Path(sys.argv[1]).resolve();raw=image.read_bytes()
    if len(raw)>2_000_000:raise ValueError('Image bound')
    run=ROOT/'local/image-question-candidate'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]);run.mkdir(parents=True)
    (ROOT/'local/current-image-question-candidate.txt').write_text(str(run),encoding='utf-8')
    db=Path.home()/'AppData/Local/hermes/state.db'
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as c:
        record=c.execute("SELECT system_prompt,system_prompt_hash FROM sessions WHERE title='Greeting #14' ORDER BY started_at DESC LIMIT 1").fetchone()
        system=record[0]
        if not system:
            value=c.execute('SELECT prompt FROM system_prompts WHERE hash=?',(record[1],)).fetchone()
            system=value[0] if value else None
    if not system:raise ValueError('Cannot recreate original host instructions')
    messages=[{'role':'system','content':system},{'role':'user','content':'chào bạn'},
        {'role':'assistant','content':'Chào bạn! Tôi sẵn sàng hỗ trợ. Bạn cần gì?'},
        {'role':'user','content':[{'type':'text','text':'câu này làm ntn giải thích chi tiết'},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}}]}]
    native=SharedNative(ROOT);trace=Store(ROOT/'local/hybrid-v1')
    class Adapter:
        def generate(self,items,**kwargs):
            start=time.monotonic();out=native.generate({'messages':items,**kwargs})
            return {'text':out['choices'][0]['message'].get('content') or '', 'usage':out.get('usage'),
                    'seconds':time.monotonic()-start,'raw':out}
    reading=prepare_image_question({'messages':messages},messages,Adapter(),trace)
    start=time.monotonic()
    if reading.get('grounded_answer'):
        answer=reading['grounded_answer']['content'];out={'producer':'runtime rendering of documentation facts',
            'choice_id':reading['grounded_answer']['choice_id'],'selection_usage':reading['grounded_answer']['selection_usage']}
    else:
        out=native.generate({'messages':messages,'max_tokens':768,'tool_choice':'none'})
        answer=out['choices'][0]['message'].get('content') or ''
    report={'image_sha256':hashlib.sha256(raw).hexdigest(),'reading':reading,'answer':answer,'raw':out,
        'answer_seconds':time.monotonic()-start,'model_calls':2,'new_training':False,
        'original_Hermes_instructions_preserved':True,'actual_Hermes_acceptance':False}
    (run/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    trace.event('work_trace.activity',{'title':'Candidate đọc ảnh và đối chiếu tài liệu local',
        'actual_answer':answer,'reading_reference_ids':reading['reference_ids'],'new_training':False,'model_calls':2},'orchestrator')
    trace.close()
    print(json.dumps({'run':str(run),**report},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
