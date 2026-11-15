"""Two actual frozen-GPU calls: original image versus literal transcription.

Private local receipts only. No training/reference answer supplied to model.
"""
from datetime import datetime,timezone
import base64,hashlib,json
from pathlib import Path
import sys,time,uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.store import Store


def main():
    image=Path(sys.argv[1]).resolve();raw=image.read_bytes()
    if len(raw)>2_000_000:raise ValueError('Image byte bound')
    run=ROOT/'local/image-question-diagnostic'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]);run.mkdir(parents=True)
    (ROOT/'local/current-image-question-native-diagnostic.txt').write_text(str(run),encoding='utf-8')
    transcript='The UI contains plotOutput("distPlot"). What must the server define for the plot to appear?\nA. input$distPlot <- plotOutput()\nB. ui$distPlot <- hist()\nC. output$distPlot <- renderPlot({...})\nD. Nothing; the plot appears automatically'
    probes=[('original_image',[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}},
                               {'type':'text','text':'câu này làm ntn giải thích chi tiết'}]),
            ('literal_transcription',transcript+'\n\nGiải thích chi tiết bằng tiếng Việt.')]
    native=SharedNative(ROOT);trace=Store(ROOT/'local/hybrid-v1')
    report={'image_sha256':hashlib.sha256(raw).hexdigest(),'model':'originalQwen0.8B BF16/F32 CUDA',
        'reference_answer_in_prompt':False,'new_training':False,'records':[]}
    for name,content in probes:
        start=time.monotonic();out=native.generate({'messages':[{'role':'user','content':content}],'max_tokens':512})
        answer=out['choices'][0]['message'].get('content') or ''
        row={'probe':name,'answer':answer,'seconds':time.monotonic()-start,'raw':out}
        report['records'].append(row)
        (run/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        trace.event('work_trace.activity',{'title':'Kiểm ảnh thật so với chữ chép nguyên văn',
            'probe':name,'actual_answer':answer,'seconds':row['seconds'],'new_training':False,
            'scope':'GPU/native diagnostic; not general image-quality benchmark'},'orchestrator')
        print(json.dumps({'probe':name,'answer':answer,'seconds':row['seconds']},ensure_ascii=False),flush=True)
    trace.close()


if __name__=='__main__':main()
