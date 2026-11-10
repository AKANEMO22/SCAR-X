"""Actual37learning and freevisualwords; fixedsource/ablation, not mainmodel."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,argparse
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from direct_visual_common import source_data,sample_stream,make,load,reference,SEEDS,METHODS
from visual_word_control_common import atomic
from train_visual_word_readout import macro
from scarx.evaluation.visual_information import ridge
from scarx.evaluation.vision_fixtures import expected,image_uri
from scarx.neural.language import tokenize
from scarx.learning.dialogue import normalized
from scarx.runtime.work_trace import append
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor


def evaluate(payload,method,cases,images,scenes,run,seed,split,ablation=None,teacher=True):
    m=load(payload,method);ref=reference(m);rows=[]
    for case in cases:
        used=case['scene_id']
        if ablation=='swapped_image':used=next(s['id'] for s in scenes if (case['kind']=='object_count' or (len(s['objects'])==1 if case['kind'].startswith('single_') else len(s['objects'])>1)) and expected(s,case['kind'],case['language'])!=case['target'])
        start=time.perf_counter();ticks=ref.ticks;reply=m.generate(case['prompt'],images[used],ablation=ablation if ablation in ('blank_image','blank_question') else None);seconds=time.perf_counter()-start;actual_ticks=ref.ticks-ticks
        metric=ref.teacher_metrics(case['prompt'],images[case['scene_id']],case['target']) if teacher else None
        rows.append({**{k:case[k] for k in ('id','scene_id','prompt','target','kind','language','novel_composition')},'image_used':used,'generated':reply['text'],
            'exact':normalized(reply['text'])==normalized(case['target']),'strict_exact':tokenize(reply['text'])==tokenize(case['target']),
            'stopped':reply['stopped'],'teacher':metric,'seconds':seconds,'actual_ticks':actual_ticks})
    if split in ('train','development'):
        for r in rows[:8]:append(run,'answer',{'scope':f'DIRECT VISUAL37 · seed{seed}/{method}/{split} · RESEARCH, NOT MAIN','control_only':method=='bptt_control',
            'case_id':r['id'],'prompt':r['prompt'],'target':r['target'],'generated':r['generated'],'image_data_uri':image_uri(images[r['scene_id']]),'image_source':'Ownedscene '+r['scene_id'],
            'input_scope':'Same32question/RGB→ALIF48/cache→autoregressiveneuralwords. Proposal30feedback/localeligibility; methodno_image/no_question zerosinput. No renderer/class/target at inference.',
            'neural_ticks':r['actual_ticks'],'seconds':r['seconds'],'active':48,'allocated':48,'checkpoint':f'Directvisual37/seed{seed}/{method}, nevermainimport',
            'qwen':None,'qwen_missing_reason':'ExistingQwen4Btext-only is NOT a visionreference; no image-as-text substitute. Actualtextcomparison36 remains separate.'})
    summary={'macro_exact':macro(rows),'micro_exact':float(np.mean([r['exact'] for r in rows])),'strict_macro_exact':macro(rows,'strict_exact'),
        'novel_composition_macro':macro([r for r in rows if r['novel_composition']]),'seen_composition_macro':macro([r for r in rows if not r['novel_composition']]),
        'by_kind':{k:macro([r for r in rows if r['kind']==k]) for k in ('single_color','single_shape','object_count','left_color')},'by_language':{l:macro([r for r in rows if r['language']==l]) for l in ('vi','en')},
        'stopped_fraction':float(np.mean([r['stopped'] for r in rows])),'actual_ticks':sum(r['actual_ticks'] for r in rows)}
    if teacher:summary.update(teacher_token_accuracy=float(np.mean([r['teacher']['token_accuracy'] for r in rows])),teacher_nll=float(np.mean([r['teacher']['nll'] for r in rows])))
    return summary,rows


def bos(model,cases,images):
    out=[]
    for case in cases:model.prepare(case['prompt'],images[case['scene_id']]);out.append(model.word_feature(1,0));model.learner.segment_active=False
    return np.stack(out)


def trial(job):
    seed,method,runpath=job
    if seed not in SEEDS or method not in METHODS:raise ValueError('Registered37trial')
    run=Path(runpath);began=time.perf_counter();cap=apply_memory_limit()
    if method=='bptt_control':
        from scarx.evaluation.controls import torch
        torch.set_num_threads(1)
    receipt,data,train,dev,images,vocab,_,_=source_data();m=make(vocab,seed,method,train,images);ref=reference(m);initial_payload=m.dumps()
    initial={n:getattr(ref.core,n).copy() for n in ('w_in','w_rec','w_out')};previous={n:a.copy() for n,a in initial.items()};losses=[];counts={};exposures=[];case_index={r['id']:i for i,r in enumerate(train)};ticks_before=ref.ticks
    with ResourceMonitor(interval=.1) as monitor:
        for i,case in enumerate(sample_stream(ref,train,1536),1):
            loss=m.learn(case['prompt'],images[case['scene_id']],case['target']);losses.append(loss['loss']);exposures.append(case_index[case['id']]);counts[case['id']]=counts.get(case['id'],0)+1
            if i%64==0:
                neural={'computed':48,'allocated':48,'tick':ref.ticks,'spikes':ref.core.spikes.astype(int).tolist(),'voltage':np.round(ref.core.voltage,4).tolist(),'sample_kind':f'Actual37 ALIFtrainingwordtrajectory/{method}, not main/thoughttranslation'}
                atomic(run/f'heartbeat-visual37-{seed}-{method}.json',{'seed':seed,'method':method,'experiences':i,'loss_mean':float(np.mean(losses[-64:])),'hidden_updates':ref.learner.hidden_updates,'seconds':time.perf_counter()-began,'neural':neural});monitor.check()
            if i%256==0:
                delta={n:float(np.linalg.norm(getattr(ref.core,n)-a)) for n,a in previous.items()}
                append(run,'learning',{'scope':f'DIRECT VISUAL37 · seed{seed}/{method} · RESEARCH, NOT MAIN','prompt':f'Visualexperiences{i-255}–{i}',
                    'passes':1,'sample_count':256,'loss_mean':float(np.mean(losses[-256:])),'losses':[float(np.mean(losses[-256:]))],'update_steps':ref.core.updates,'update_unit':'outputlocalwordtargets cumulative',
                    'hidden_update_steps':ref.learner.hidden_updates,'weight_delta':delta,'attempted_weight_delta':delta,'changed_synapses':{n:int(np.count_nonzero(getattr(ref.core,n)!=a)) for n,a in previous.items()},
                    'accepted':True,'rolled_back':False,'guard_checked':False,'reason':('GLOBAL/TEMPORAL BPTT onlyisolatedcontrol. ' if method=='bptt_control' else 'Approvedone-stepreadoutfeedback × own-neuron/cacheforwardeligibility, no network/time backward. ')+ 'Ownedrendererlabels not modelanswers, no mainupdate/guard.'})
                previous={n:getattr(ref.core,n).copy() for n in previous}
            if time.perf_counter()-began>300:raise TimeoutError('37trial300s')
        train_seconds=time.perf_counter()-began;training_ticks=ref.ticks-ticks_before;payload=m.dumps();artifact=root/'artifacts/direct-visual-bridge'/run.name;artifact.mkdir(parents=True,exist_ok=True)
        checkpoint=artifact/f'{seed}-{method}.visual.zip';checkpoint.write_bytes(payload);summaries={};outputs={};causal={};devscenes=[s for s in data['scenes'] if s['split']=='development']
        for split,cases in (('train',train),('development',dev)):summaries[split],outputs[split]=evaluate(payload,method,cases,images,devscenes,run,seed,split)
        if method=='correct':
            for intervention in ('blank_image','blank_question','swapped_image'):causal[intervention],outputs[intervention]=evaluate(payload,method,dev,images,devscenes,run,seed,intervention,intervention,False)
        learned=reference(load(payload,method));original=reference(load(initial_payload,method));ft=bos(learned,train,images);fd=bos(learned,dev,images);it=bos(original,train,images);idd=bos(original,dev,images)
        labels=np.array([ref.ids[tokenize(r['target'])[0]] for r in train]);devlabels=np.array([ref.ids[tokenize(r['target'])[0]] for r in dev]);indices=np.array(exposures)
        probe=ridge(ft,fd,labels,ref.core.config.outputs,indices,mask=ref.mask(0));before_probe=ridge(it,idd,labels,ref.core.config.outputs,indices,mask=ref.mask(0))
        probe_score=float(np.mean(probe['dev']==devlabels));initial_score=float(np.mean(before_probe['dev']==devlabels))
        lesion=load(payload,method);lr=reference(lesion)
        for n in ('w_in','w_rec'):getattr(lr.core,n)[:]=initial[n]
        summaries['restore_initial_hidden'],outputs['restore_initial_hidden']=evaluate(lesion.dumps(),method,dev,images,devscenes,run,seed,'restore_initial_hidden')
        assert m.dumps()==payload
        clone=load(payload,method);case=train[0];learn=clone.learn(case['prompt'],images[case['scene_id']],case['target']);reply=clone.generate(dev[0]['prompt'],images[dev[0]['scene_id']]);draw=int(reference(clone).sampler_rng.integers(10000))
        cold={'learn':learn,'reply':reply,'sampler_draw':draw,'full_state_sha256':hashlib.sha256(clone.dumps()).hexdigest()}
        row={'seed':seed,'method':method,'experiences':1536,'word_updates':ref.core.updates,'hidden_updates':ref.learner.hidden_updates,'training_ticks':training_ticks,
            'summary':summaries,'causal':causal,'firstword_resethead_probe':probe_score,'initial_firstword_resethead_probe':initial_score,'probe_gain':probe_score-initial_score,
            'stratum_counts':counts,'array_bytes':ref.memory_bytes(),'checkpoint_bytes':len(payload),'core_fingerprint':ref.core.fingerprint(),'head_sha256':hashlib.sha256(ref.head.dumps()).hexdigest(),
            'checkpoint':checkpoint.relative_to(root).as_posix(),'sha256':hashlib.sha256(payload).hexdigest(),'cold_expected':cold,'mean_training_loss':float(np.mean(losses)),
            'hidden_delta_in':float(np.linalg.norm(ref.core.w_in-initial['w_in'])),'hidden_delta_rec':float(np.linalg.norm(ref.core.w_rec-initial['w_rec'])),
            'train_seconds':train_seconds,'seconds':time.perf_counter()-began,'resources':monitor.report(),'cap':cap,'control_only':method=='bptt_control','candidate_network_backward':False}
        if method=='bptt_control':row.update(backward_passes=m.backward_passes,surrogate_forward_ticks=m.surrogate_forward_ticks,max_graph_state_error=m.max_graph_state_error,max_graph_feature_error=m.max_graph_feature_error,max_graph_loss_error=m.max_graph_loss_error)
        monitor.check()
        if time.perf_counter()-began>300:raise TimeoutError('37trial300s incl evaluation/probe')
        atomic(run/f'visual37-{seed}-{method}.json',{'record':row,'outputs':outputs,'neural_sample':neural});return row,neural


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args();path=Path(args.job).resolve();assert path.is_relative_to(root/'runs') and path.stat().st_size<65536
    job=json.loads(path.read_text(encoding='utf-8'));seed,method,runpath=job['trial'];assert Path(runpath).resolve()==path.parent
    if job['mode']=='cold':
        apply_memory_limit()
        if method=='bptt_control':
            from scarx.evaluation.controls import torch
            torch.set_num_threads(1)
        checkpoint=(root/job['checkpoint']).resolve();assert checkpoint.is_relative_to(root/'artifacts/direct-visual-bridge') and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==job['sha256']
        _,_,train,dev,images,_,_,_=source_data();m=load(checkpoint.read_bytes(),method);case=train[0];learn=m.learn(case['prompt'],images[case['scene_id']],case['target']);reply=m.generate(dev[0]['prompt'],images[dev[0]['scene_id']]);draw=int(reference(m).sampler_rng.integers(10000))
        atomic(path.parent/f'cold-visual37-{seed}-{method}.json',{'learn':learn,'reply':reply,'sampler_draw':draw,'full_state_sha256':hashlib.sha256(m.dumps()).hexdigest()});return
    r,_=trial(tuple(job['trial']));print(json.dumps({'seed':seed,'method':method,'dev':r['summary']['development']['macro_exact']}),flush=True)


if __name__=='__main__':main()
