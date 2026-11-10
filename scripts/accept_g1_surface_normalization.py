"""Registered fixed optional surface normalizer on existing rehearsal clones."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,itertools
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.unified import UnifiedCore
from scarx.neural.language import tokenize
from scarx.learning.dialogue import normalized
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from rehearse_g1_basics import score,digit_score,STATES


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    import psutil
    if any((p.info['name'] or '').lower()=='llama-server.exe' for p in psutil.process_iter(['name'])):raise ValueError('Qwen reference must finish first')
    cap=apply_memory_limit();began=time.perf_counter()
    source=json.loads((root/'artifacts/g1-basic-rehearsal-latest.json').read_text(encoding='utf-8'))
    private_source=(root/source['conditions'][0]['candidate_private_path']).parent
    source_payload=(private_source/'source-model.zip').read_bytes()
    old_data=json.loads((private_source/'source-outputs.json').read_text(encoding='utf-8'))
    cases={r['case_id']:{'id':r['case_id'],'prompt':r['prompt'],'target':r['target'],'group':r['group']} for r in old_data['outputs'] if r['group']!='personal_guard'}
    guards={r['id']:r for r in old_data['guards']}
    for r in old_data['proofs']:guards[r['id']]={**r,'group':'personal_guard'}
    allcases=list(cases.values())+list(guards.values());baseline=score(source_payload,allcases)
    run=new_run(root,'g1-terminal-surface-acceptance-v1',{'protocol':'experiments/g1-terminal-surface-acceptance-v1.md',
        'display_title':'G1 · dấu câu đầu vào + kiểm guard runtime · không train thêm',
        'model_scope':'Existing private rehearsal checkpoints only; optional input normalization, no serving update',
        'source_run':source['run'],'modes':['readout','full'],'orders':[1701,1709,1721],'new_training':False})
    private=root/'local/g1-surface-acceptance'/run.name;private.mkdir(parents=True)
    write_json(private/'source-baseline.json',baseline)
    from sklearn.datasets import load_digits
    digits=load_digits();split=json.loads((root/'runs/20261109T000625Z-generation1-joint-bd6c92/vision-split.json').read_text(encoding='utf-8'))
    images=[(int(i),digits.data[i],int(digits.target[i])) for i in split['diagnostic_test'][:48]];old_digits=digit_score(source_payload,images)
    old_digit_accuracy=float(np.mean([r['predicted']==r['target'] for r in old_digits]));records=[]
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'))
    def surface_words(text):
        words=tokenize(text)
        while words and words[-1] in ('.','!','?'):words.pop()
        return tuple(words)
    known={surface_words(p) for f in corpus['families'] for split in ('train','development','acceptance') for p in f[split]}
    exposure=[r['id'] for r in cases.values() if r['group']=='antigravity' and surface_words(r['prompt']) in known]
    with ResourceMonitor(interval=.1) as monitor:
        for previous in source['conditions']:
            for enabled in (False,True):
                path=root/previous['candidate_private_path'];payload=path.read_bytes();model=UnifiedCore.loads(payload);fingerprint=model.core.fingerprint()
                if enabled:model.encoding['surface_normalization']='terminal-punctuation-v1'
                candidate=model.dumps();assert model.core.fingerprint()==fingerprint
                actual=score(candidate,allcases);states={};passed=True
                for state in STATES:
                    values=[r for r in actual if r['state']==state];old=[r for r in baseline if r['state']==state]
                    known_accuracy=float(np.mean([r['exact'] for r in values if r['group']=='known']))
                    agy=float(np.mean([r['exact'] for r in values if r['group']=='antigravity']));old_agy=float(np.mean([r['exact'] for r in old if r['group']=='antigravity']))
                    old_correct={r['case_id'] for r in old if r['group']=='personal_guard' and normalized(r['generated'])==normalized(r['target'])}
                    lost=[r['case_id'] for r in values if r['case_id'] in old_correct and normalized(r['generated'])!=normalized(r['target'])]
                    statepass=known_accuracy>=(.8 if state=='canonicalreset' else .75) and agy>=.25 and agy-old_agy>=.1 and not lost
                    states[state]={'known':known_accuracy,'antigravity':agy,'source_antigravity':old_agy,'lost_personal_runtime_guards':len(lost),'pass':statepass};passed&=statepass
                new_digits=digit_score(candidate,images);digitlost=sum(old['predicted']==old['target'] and new['predicted']!=new['target'] for old,new in zip(old_digits,new_digits))
                digit_accuracy=float(np.mean([r['predicted']==r['target'] for r in new_digits]));passed&=digitlost==0 and digit_accuracy-old_digit_accuracy>=-.03
                name=f"{previous['mode']}-{previous['curriculum_order_seed']}-surface{int(enabled)}"
                candidate_path=private/(name+'.zip');candidate_path.write_bytes(candidate)
                record={'mode':previous['mode'],'curriculum_order_seed':previous['curriculum_order_seed'],'surface_normalization':enabled,
                    'states':states,'screen_pass':bool(passed),'lost_old_correct_digits':digitlost,'digit_accuracy':digit_accuracy,
                    'weights_unchanged_from_rehearsal':True,'candidate_private_path':candidate_path.relative_to(root).as_posix(),
                    'candidate_sha256':hashlib.sha256(candidate).hexdigest(),'original_source_sha256':previous['source_sha256'],
                    'original_source_private_path':(private_source/'source-model.zip').relative_to(root).as_posix(),
                    'elapsed_seconds':time.perf_counter()-began}
                records.append(record);write_json(private/(name+'-outputs.json'),{'outputs':actual,'digits':new_digits})
                for row in actual:
                    if row['group']!='antigravity' or row['state']!='public_idle_probes':continue
                    append(run,'answer',{**row,'scope':f'G1 clone · {name} · not serving',
                        'input_scope':'Public prompt, optional terminal .!? normalization in encoder; targets never used at inference. Personal guard outputs stay private.',
                        'checkpoint':'private candidate in local/','qwen':None})
                progress(run,name+' · guards',len(records),12,metrics={'screen_pass':record['screen_pass'],
                    'reset_known':states['canonicalreset']['known'],'reset_antigravity':states['canonicalreset']['antigravity'],
                    'lost_runtime_guards':sum(r['lost_personal_runtime_guards'] for r in states.values()),'lost_digits':digitlost})
                monitor.check()
                if time.perf_counter()-began>180:raise TimeoutError('Registered180s surface acceptance limit')
        mode_pass={mode:all(r['screen_pass'] for r in records if r['mode']==mode and r['surface_normalization']) for mode in ('readout','full')}
        chosen=next((mode for mode in ('readout','full') if mode_pass[mode]),None)
        selected=next((r for r in records if r['mode']==chosen and r['curriculum_order_seed']==1701 and r['surface_normalization']),None) if chosen else None
        result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'decision':'PRODUCT_NORMALIZER_READY_FOR_SEPARATE_GUARDED_TRANSACTION' if chosen else 'SURFACE_SCREEN_FAILED_KEEP_LIVE',
            'conditions':records,'all_order_passes':mode_pass,'selected':selected,
            'terminal_punctuation_overlap_cases':exposure,'terminal_punctuation_overlap_count':len(exposure),
            'original_rehearsal_failure_rewritten':False,'old_strict_token_metrics_unchanged':True,
            'personal_runtime_normalization_existed_before_this_run':True,'new_training':False,'deployed':False,
            'normal_chat_or_vision_proven':False,'lifelong_P2_pass':False,'resources':monitor.report(),'cap':cap}
        (run/'report.md').write_text('# G1 optional surface normalization acceptance\n\nOriginal rehearsal failures unchanged; actual inference only, personalguards match pre-existing runtime policy. No training or automaticdeployment.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-surface-acceptance-latest.json',result)
        print(json.dumps({'run':str(run),'decision':result['decision'],'all_order_passes':mode_pass,
            'states':selected['states'] if selected else None,'word_surface_overlap':len(exposure)}),flush=True)


if __name__=='__main__':main()
