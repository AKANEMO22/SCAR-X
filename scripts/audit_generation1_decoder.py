from pathlib import Path
import sys
import json
import math
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.neural.unified import UnifiedCore
from scarx.neural.language import tokenize
from scarx.evaluation.runs import new_run,write_json
release=json.loads((root/'artifacts/generation1-release.json').read_text())
run=new_run(root,'generation1-decoder-audit',{'source':release['checkpoint'],'protocol':'experiments/generation1-decoder-audit.md'},data_files=['data/dialogue/generation1.json'])
corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'))
results={}
for stop in (False,True):
    model=UnifiedCore.loads((root/release['checkpoint']).read_bytes());outputs=[]
    for family in corpus['families']:
        for prompt in family['development']:
            result=model.generate(prompt,stop_at_sentence=stop)
            outputs.append({'prompt':prompt,'output':result['text'],'exact':tokenize(result['text'])==tokenize(family['answer']),'stop_reason':result['stop_reason']})
    results['sentence' if stop else 'eos']={'development_exact':sum(r['exact'] for r in outputs)/len(outputs),'outputs':outputs}
write_json(run/'result.json',{'decoders':results,'test_opened':False,'weights_updated':False,'learning_improvement_claim':False})
(run/'report.md').write_text('# Generation1 decoder audit\n\n'+json.dumps(results,ensure_ascii=False,indent=2)+'\n\nDevelopment only; no test reopened, no learning improvement claim.\n',encoding='utf-8')
print(json.dumps({'run':run.relative_to(root).as_posix(),'eos_dev':results['eos']['development_exact'],'sentence_dev':results['sentence']['development_exact']}))
