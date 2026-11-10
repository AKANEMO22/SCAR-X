import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json
import sys
from scarx.neural.language import WordLanguageCore,tokenize
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.evaluation.runs import new_run,write_json

if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
root=Path(__file__).resolve().parents[1];cap=apply_memory_limit()
selection=json.loads((root/'runs/20261108T190315Z-language-vision-pilot-v2-6a889e/language-selection.json').read_text(encoding='utf-8'))['chosen']
run=new_run(root,'language-decoder-audit-v1',{'checkpoint':selection['checkpoint'],'width':4,'length_penalty':.6,'max_words':32},
 data_description='Self-authored dialogue development split only; no training/test reuse or weight updates',data_files=['data/dialogue/pilot-v1.json'])
model=WordLanguageCore.load(root/selection['checkpoint']);fingerprint=model.core.fingerprint()
corpus=json.loads((root/'data/dialogue/pilot-v1.json').read_text(encoding='utf-8'))
rows=[]
with ResourceMonitor(interval=.5) as monitor:
    for family in corpus['families']:
        for prompt in family['development']:
            greedy=model.generate(prompt); beam=model.generate_beam(prompt)
            row={'prompt':prompt,'target':family['answer'],'greedy':greedy['text'],'beam':beam['text'],
                 'greedy_exact':tokenize(greedy['text'])==tokenize(family['answer']),'beam_exact':tokenize(beam['text'])==tokenize(family['answer']),'beam_stopped':beam['stopped']}
            rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True);monitor.check()
    result={'outputs':rows,'greedy_development_exact':sum(r['greedy_exact'] for r in rows)/len(rows),
            'beam_development_exact':sum(r['beam_exact'] for r in rows)/len(rows),'weights_unchanged':model.core.fingerprint()==fingerprint,
            'resources':monitor.report(),'windows_limit':cap,'test_opened':False,'production_replaced':False}
    result['development_gate']=result['beam_development_exact']>=.75 and all(r['beam_stopped'] for r in rows)
    write_json(run/'result.json',result)
    lines=['# Language decoder audit','',f"Greedy devexact={result['greedy_development_exact']:.3f}; beam devexact={result['beam_development_exact']:.3f}; gate={result['development_gate']}",'',
           '| Prompt | Greedy | Beam |','|---|---|---|']
    lines+=['| '+r['prompt']+' | '+r['greedy']+' | '+r['beam']+' |' for r in rows]
    lines+=['','No weight updates or response lookup. Test unopened, production unchanged.']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('RESULT '+str(run),flush=True)
