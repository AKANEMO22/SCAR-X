"""Actual readonly intervention on all immutable v1 word-score checkpoints."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,itertools
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from train_g1_word_score_inputs import source_data
from generate_g1_state_paraphrases import INTENTS
from scarx.evaluation.word_conditioning import measure
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def main():
    began=time.perf_counter();cap=apply_memory_limit();train,dev,prior,_,sources=source_data();diagnostics=[]
    for intent in INTENTS:diagnostics += [r for r in train if r['intent']==intent][:2]
    # Explicit same family order as protocol across the three splits.
    splits={name:sorted(rows,key=lambda r:INTENTS.index(r['intent'])) for name,rows in [('train16',diagnostics),('development',dev),('prior',prior)]}
    old='20261109T175033Z-g1-word-score-transfer-v1-1b1604';records=[]
    run=new_run(root,'g1-word-conditioning-audit-v1',{'protocol':'experiments/g1-word-conditioning-audit-v1.md',
        'display_title':'Đọc checkpointv1 · câu hỏi có giúp dự đoán từ đầu?',
        'model_scope':'Readonly firsttoken diagnostic on nine old48ALIFmodels; no training/mainchanges',
        'model_seeds':[27103,27109,27127],'conditions':['frozen','full','shuffled'],'interventions':['real','zero','cross_family']},
        data_description='Same public95cases and ALL9oldv1models; no fittedhead/labels in input or newtraining.',data_files=sources)
    (run/'report.md').write_text('# Word conditioning audit\n\nReadonly firsttoken intervention in progress; no fullutterance/quality claim.\n',encoding='utf-8')
    with ResourceMonitor(interval=.1) as monitor:
        for seed,condition in itertools.product((27103,27109,27127),('frozen','full','shuffled')):
            path=root/'artifacts/word-score-transfer'/old/f'{seed}-{condition}.zip';payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest()
            values={split:measure(payload,cases) for split,cases in splits.items()}
            assert digest==hashlib.sha256(path.read_bytes()).hexdigest()
            record={'seed':seed,'condition':condition,'source_sha256':digest,'splits':{split:v['summary'] for split,v in values.items()},'source_unchanged':True}
            records.append(record);write_json(run/f'{seed}-{condition}.json',values)
            progress(run,f'{seed}/{condition} · readonlyfirsttoken',len(records),9,metrics={'dev_real':values['development']['summary']['real']['first_token_accuracy'],
                'dev_zero':values['development']['summary']['zero']['first_token_accuracy']})
            monitor.check()
            if time.perf_counter()-began>120:raise TimeoutError('Registered120s conditioningaudit limit')
        summary={condition:{split:{mode:{key:float(np.mean([r['splits'][split][mode][key] for r in records if r['condition']==condition])) for key in records[0]['splits'][split][mode]} for mode in ('real','zero','cross_family')} for split in splits} for condition in ('frozen','full','shuffled')}
        gains={condition:{split:{mode:interval([r['splits'][split]['real']['first_token_accuracy']-r['splits'][split][mode]['first_token_accuracy'] for r in records if r['condition']==condition]) for mode in ('zero','cross_family')} for split in splits} for condition in ('frozen','full','shuffled')}
        result={'run':run.relative_to(root).as_posix(),'source_run':'runs/'+old,'decision':'READONLY_CONDITIONING_DIAGNOSTIC_ONLY',
            'summary':summary,'paired_gains':gains,'all_state_unchanged':True,'new_training':False,'deployed':False,
            'normal_chat_or_vision_proven':False,'resources':monitor.report(),'cap':cap,
            'limitations':['Firstword only; shared targetfirstwords confound accuracy, sensitivity is not useful semantics.',
                'Batched RNG is newdiagnostic, not original v1 decoder scores. Three oldfreshmodels percondition; no newvalidation.']}
        (run/'report.md').write_text('# Word conditioning audit\n\nReadonly firstword sensitivity and usefulness, never a full model reply or learning result.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-word-conditioning-latest.json',result)
        activity(run,'Đã đo can thiệp câu hỏi trên checkpointv1','Không học thêm hoặc sửa fileweights; so real/zero/crossfamily chỉ ở từ đầu, không phải câu trả lời đầy đủ.',evidence={'models':9,'cases_per_model':95,'all_state_unchanged':True,'seconds':monitor.report()['elapsed_seconds']})
        print(json.dumps({'run':result['run'],'decision':result['decision'],'summary':summary},ensure_ascii=False))


if __name__=='__main__':main()
