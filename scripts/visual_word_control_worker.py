"""One isolated ALIF-BPTT visualword child; never imported by serving."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,argparse
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from visual_word_control_common import source_data,calibrated,sample_stream,verify_baseline,SEEDS,atomic
from train_visual_word_readout import macro
from scarx.evaluation.vision_fixtures import expected,image_uri
from scarx.learning.dialogue import normalized
from scarx.neural.language import tokenize
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import append


def evaluate(payload,cases,images,scenes,run,seed,split,ablation=None,teacher=True):
    from scarx.evaluation.controls import VisualWordLearnabilityControl
    m=VisualWordLearnabilityControl.loads(payload);rows=[]
    for case in cases:
        used=case['scene_id']
        if ablation=='swapped_image':used=next(s['id'] for s in scenes if (case['kind']=='object_count' or (len(s['objects'])==1 if case['kind'].startswith('single_') else len(s['objects'])>1)) and expected(s,case['kind'],case['language'])!=case['target'])
        start=time.perf_counter();ticks=m.reference.ticks
        reply=m.generate(case['prompt'],images[used],ablation=ablation if ablation in ('blank_image','blank_question') else None)
        elapsed=time.perf_counter()-start;actual_ticks=m.reference.ticks-ticks
        metrics=m.reference.teacher_metrics(case['prompt'],images[case['scene_id']],case['target']) if teacher else None
        rows.append({**{k:case[k] for k in ('id','scene_id','prompt','target','kind','language','novel_composition')},'image_used':used,'generated':reply['text'],
            'exact':normalized(reply['text'])==normalized(case['target']),'strict_exact':tokenize(reply['text'])==tokenize(case['target']),
            'stopped':reply['stopped'],'teacher':metrics,'actual_ticks':actual_ticks,'seconds':elapsed})
    if ablation is None:
        for r in rows[:8]:append(run,'answer',{'scope':f'ISOLATED ALIF-BPTT VISUAL CONTROL · seed{seed}/{split} · NOT MAIN/LOCAL',
            'case_id':r['id'],'prompt':r['prompt'],'target':r['target'],'generated':r['generated'],'image_data_uri':image_uri(images[r['scene_id']]),
            'image_source':'Owned renderedscene '+r['scene_id'],'input_scope':'Same48ALIF question/RGB input and neural wordhead as32; hidden optimized ONLY in BPTTcontrol. No renderer/goldtarget at freeinference.',
            'qwen':None,'qwen_missing_reason':'ExistingQwen4Btext-only is not a visionreference; no substitute.',
            'neural_ticks':r['actual_ticks'],'seconds':r['seconds'],'active':48,'allocated':48,'checkpoint':'Isolated visual ALIF-BPTT CONTROL, never imported into main'})
    summary={'macro_exact':macro(rows),'strict_macro_exact':macro(rows,'strict_exact'),'micro_exact':float(np.mean([r['exact'] for r in rows])),
        'novel_composition_macro':macro([r for r in rows if r['novel_composition']]),'seen_composition_macro':macro([r for r in rows if not r['novel_composition']]),
        'by_kind':{k:macro([r for r in rows if r['kind']==k]) for k in ('single_color','single_shape','object_count','left_color')},
        'by_language':{l:macro([r for r in rows if r['language']==l]) for l in ('vi','en')},'stopped_fraction':float(np.mean([r['stopped'] for r in rows])),
        'actual_ticks':sum(r['actual_ticks'] for r in rows)}
    if teacher:summary.update(teacher_nll=float(np.mean([r['teacher']['nll'] for r in rows])),teacher_token_accuracy=float(np.mean([r['teacher']['token_accuracy'] for r in rows])))
    return summary,rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args();path=Path(args.job).resolve()
    if not path.is_relative_to(root/'runs') or path.stat().st_size>65536:raise ValueError('Registered visualjob required')
    job=json.loads(path.read_text(encoding='utf-8'));run=path.parent;seed=job['seed'];assert seed in SEEDS
    cap=apply_memory_limit();began=time.perf_counter();receipt,data,train,dev,images,vocab,source_rows,_=source_data()
    with ResourceMonitor(interval=.1) as monitor:
        from scarx.evaluation.controls import VisualWordLearnabilityControl,torch
        torch.set_num_threads(1)
        if job.get('mode')=='cold':
            artifact=(root/job['checkpoint']).resolve()
            if not artifact.is_relative_to(root/'artifacts/isolated-visual-controls') or hashlib.sha256(artifact.read_bytes()).hexdigest()!=job['sha256']:raise ValueError('Coldcheckpoint provenance')
            m=VisualWordLearnabilityControl.loads(artifact.read_bytes());case=train[0]
            learn=m.learn(case['prompt'],images[case['scene_id']],case['target']);reply=m.generate(case['prompt'],images[case['scene_id']]);draw=int(m.reference.sampler_rng.integers(10000))
            atomic(run/f'cold-visual-{seed}.json',{'learn':learn,'reply':reply,'sampler_draw':draw,'final_sha256':hashlib.sha256(m.dumps()).hexdigest()});return
        reference=calibrated(vocab,seed,train,images);baseline=verify_baseline(reference,source_rows,seed);m=VisualWordLearnabilityControl(reference)
        initial={n:getattr(reference.core,n).copy() for n in ('w_in','w_rec','w_out')};previous={n:a.copy() for n,a in initial.items()};losses=[];counts={};ticks_before=reference.ticks
        for i,case in enumerate(sample_stream(reference,train,1536),1):
            losses.append(m.learn(case['prompt'],images[case['scene_id']],case['target'])['loss']);key=case['kind']+'/'+case['language']+'/'+case['target'];counts[key]=counts.get(key,0)+1
            if i%64==0:
                core=reference.core;neural={'computed':48,'allocated':48,'tick':reference.ticks,'spikes':core.spikes.astype(int).tolist(),'voltage':np.round(core.voltage,4).tolist(),'sample_kind':'Actual authoritative ALIF trajectory in isolated visual BPTT CONTROL'}
                atomic(run/'visual-control-progress.json',{'seed':seed,'responses':i,'loss_mean':float(np.mean(losses[-64:])),'elapsed_seconds':time.perf_counter()-began,'backward_passes':m.backward_passes,'neural':neural})
                if not all(np.isfinite(a).all() for a in core.arrays().values()):raise ValueError('Nonfinite visualcontrol state')
                monitor.check()
            if i%256==0:
                delta={n:float(np.linalg.norm(getattr(core,n)-a)) for n,a in previous.items()}
                append(run,'learning',{'scope':f'ISOLATED ALIF-BPTT VISUAL CONTROL · seed{seed} · NOT MAIN/LOCAL','prompt':f'Visualcontrol experiences{i-255}–{i}',
                    'passes':1,'sample_count':256,'loss_mean':float(np.mean(losses[-256:])),'losses':[float(np.mean(losses[-256:]))],
                    'update_steps':core.updates,'update_unit':'local head wordtargets cumulative · CONTROL ONLY','hidden_update_steps':m.optimizer_steps,
                    'weight_delta':delta,'attempted_weight_delta':delta,'changed_synapses':{n:int(np.count_nonzero(getattr(core,n)!=a)) for n,a in previous.items()},
                    'accepted':True,'rolled_back':False,'guard_checked':False,'reason':f'ISOLATED CONTROL: {m.backward_passes} globaltemporal surrogate backwardpasses, head still outputlocal. No mainupdate/guard; qualitypending.'})
                previous={n:getattr(core,n).copy() for n in previous}
            if time.perf_counter()-began>1500:raise TimeoutError('Registered1500s visualcontrol trialcap')
        train_seconds=time.perf_counter()-began;training_ticks=reference.ticks-ticks_before
        assert counts==baseline['stratum_counts'] and reference.core.updates==baseline['word_updates'] and training_ticks==baseline['training_ticks'] and m.optimizer_steps==1536
        payload=m.dumps();artifact=root/'artifacts/isolated-visual-controls'/run.name;artifact.mkdir(parents=True,exist_ok=True);checkpoint=artifact/f'{seed}.control.zip';checkpoint.write_bytes(payload)
        summaries={};outputs={};causal={};scene_rows=[s for s in data['scenes'] if s['split']=='development']
        for split,cases in (('train864',train),('development',dev)):
            summaries[split],outputs[split]=evaluate(payload,cases,images,scene_rows,run,seed,split)
        for intervention in ('blank_image','blank_question','swapped_image'):
            causal[intervention],outputs[intervention]=evaluate(payload,dev,images,scene_rows,run,seed,intervention,intervention,False)
        assert m.dumps()==payload
        copy=VisualWordLearnabilityControl.loads(payload);case=train[0];learn=copy.learn(case['prompt'],images[case['scene_id']],case['target']);reply=copy.generate(case['prompt'],images[case['scene_id']]);draw=int(copy.reference.sampler_rng.integers(10000))
        cold={'learn':learn,'reply':reply,'sampler_draw':draw,'final_sha256':hashlib.sha256(copy.dumps()).hexdigest()}
        record={'seed':seed,'condition':'full','method':'isolated_visual_surrogate_bptt','control_only':True,'hidden_unchanged':False,
            'experiences':1536,'word_updates':reference.core.updates,'training_ticks':training_ticks,'stratum_counts':counts,'summary':summaries,'causal':causal,
            'optimizer_steps':m.optimizer_steps,'backward_passes':m.backward_passes,'surrogate_forward_ticks':m.surrogate_forward_ticks,
            'max_graph_state_error':m.max_graph_state_error,'max_graph_feature_error':m.max_graph_feature_error,'max_graph_loss_error':m.max_graph_loss_error,
            'zero_gradient_responses':m.zero_gradient_responses,'hidden_delta_in':float(np.linalg.norm(reference.core.w_in-initial['w_in'])),'hidden_delta_rec':float(np.linalg.norm(reference.core.w_rec-initial['w_rec'])),
            'optimizer_array_bytes':sum(getattr(m,n).nbytes for n in ('m_in','v_in','m_rec','v_rec')),'reference_array_bytes':reference.memory_bytes(),
            'checkpoint':checkpoint.relative_to(root).as_posix(),'sha256':hashlib.sha256(payload).hexdigest(),'cold_expected':cold,
            'mean_training_loss':float(np.mean(losses)),'train_seconds':train_seconds,'seconds':time.perf_counter()-began,'resources':monitor.report(),'cap':cap,'torch_version':torch.__version__}
        monitor.check()
        if time.perf_counter()-began>1500:raise TimeoutError('Registered1500s visualcontrol incl evaluation')
        atomic(run/f'control-visual-{seed}.json',{'record':record,'outputs':outputs});print(json.dumps({'seed':seed,'dev_macro':summaries['development']['macro_exact'],'seconds':record['seconds']}),flush=True)


if __name__=='__main__':main()
