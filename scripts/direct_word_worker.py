"""Actual fixedscreen wordlearning, no teacher or serving proxy."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,argparse
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from direct_words_common import source_data,make,load,reference,stream,macro,SEEDS,METHODS
from visual_word_control_common import atomic
from scarx.neural.language import tokenize
from scarx.learning.dialogue import normalized
from scarx.evaluation.visual_information import ridge
from scarx.evaluation.direct_word_data import FAMILIES
from scarx.runtime.work_trace import append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def evaluate(payload,method,cases,run,seed,split):
    m=load(payload,method);ref=reference(m);rows=[]
    for case in cases:
        began=time.perf_counter();ticks=ref.ticks;reply=m.generate(case['prompt']);seconds=time.perf_counter()-began;actualticks=ref.ticks-ticks
        metric=ref.teacher_metrics(case['prompt'],case['target']);rows.append({'id':case['id'],'prompt':case['prompt'],'target':case['target'],'family':case['family'],'language':case['language'],
            'generated':reply['text'],'exact':normalized(reply['text'])==normalized(case['target']),'strict_exact':tokenize(reply['text'])==tokenize(case['target']),
            'stopped':reply['stopped'],'teacher':metric,'seconds':seconds,'actual_ticks':actualticks})
    if split in ('train','development'):
        for r in rows[:8]:append(run,'answer',{'scope':f'DIRECT WORD BRIDGE · seed{seed}/{method}/{split} · RESEARCH, NOT MAIN','control_only':method=='bptt_control',
            'case_id':r['id'],'prompt':r['prompt'],'target':r['target'],'generated':r['generated'],'seconds':r['seconds'],'neural_ticks':r['actual_ticks'],'active':48,'allocated':48,
            'input_scope':'Actual singleusertext→legacyG1hash128→ALIF48→autoregressiveneuralwords. No intent/language/target at inference. Small8stylepilot, not normalchat.',
            'qwen':None,'qwen_missing_reason':'FreshQwen64sameprompts measured separately aftertrials; not a proxy/label.',
            'checkpoint':f'Experimental directword/seed{seed}/{method}, not mainrelease'})
    return {'macro_exact':macro(rows),'micro_exact':float(np.mean([r['exact'] for r in rows])),'strict_macro_exact':macro(rows,'strict_exact'),
        'by_family':{f:macro([r for r in rows if r['family']==f]) for f in FAMILIES},'by_language':{l:macro([r for r in rows if r['language']==l]) for l in ('vi','en')},
        'stopped_fraction':float(np.mean([r['stopped'] for r in rows])),'teacher_nll':float(np.mean([r['teacher']['nll'] for r in rows])),
        'teacher_token_accuracy':float(np.mean([r['teacher']['token_accuracy'] for r in rows]))},rows


def bos_features(model,cases):
    out=[]
    for case in cases:model.prepare(case['prompt']);out.append(model.word(1,0));model.learner.segment_active=False
    return np.stack(out)


def trial(job):
    seed,method,runpath=job;assert seed in SEEDS and method in METHODS;run=Path(runpath);cap=apply_memory_limit();began=time.perf_counter()
    if method=='bptt_control':
        from scarx.evaluation.controls import torch
        torch.set_num_threads(1)
    receipt,train,dev,vocab,_=source_data();m=make(vocab,seed,method,train);ref=reference(m);initial_payload=m.dumps();initial={n:getattr(ref.core,n).copy() for n in ('w_in','w_rec','w_out')};previous={n:a.copy() for n,a in initial.items()}
    losses=[];counts={};indices=[];index={r['id']:i for i,r in enumerate(train)};ticks_before=ref.ticks
    with ResourceMonitor(interval=.1) as monitor:
        for i,case in enumerate(stream(ref,train,1024),1):
            losses.append(m.learn(case['prompt'],case['target'])['loss']);counts[case['id']]=counts.get(case['id'],0)+1;indices.append(index[case['id']])
            if i%64==0:
                neural={'computed':48,'allocated':48,'tick':ref.ticks,'spikes':ref.core.spikes.astype(int).tolist(),'voltage':np.round(ref.core.voltage,4).tolist(),
                    'sample_kind':f'Actualwordtraining ALIFtrajectory/{method}, proposal30research, NOT main'}
                atomic(run/f'heartbeat-word-{seed}-{method}.json',{'seed':seed,'method':method,'responses':i,'loss_mean':float(np.mean(losses[-64:])),'seconds':time.perf_counter()-began,'hidden_updates':ref.learner.hidden_updates,'neural':neural});monitor.check()
            if i%256==0:
                delta={n:float(np.linalg.norm(getattr(ref.core,n)-a)) for n,a in previous.items()}
                append(run,'learning',{'scope':f'DIRECT WORD BRIDGE · seed{seed}/{method} · RESEARCH, NOT MAIN','prompt':f'Wordresponses{i-255}–{i}',
                    'passes':1,'sample_count':256,'loss_mean':float(np.mean(losses[-256:])),'losses':[float(np.mean(losses[-256:]))],
                    'update_steps':ref.core.updates,'update_unit':'outputlocalwordtargets cumulative','hidden_update_steps':ref.learner.hidden_updates,
                    'weight_delta':delta,'attempted_weight_delta':delta,'changed_synapses':{n:int(np.count_nonzero(getattr(ref.core,n)!=a)) for n,a in previous.items()},
                    'accepted':True,'rolled_back':False,'guard_checked':False,'reason':('BPTT ONLYISOLATEDCONTROL; ' if method=='bptt_control' else 'Approvedone-stepreadoutfeedback/localforwardeligibility; ')+ 'No servingguard/mainweightupdate. Ownedstylelabels not factualtruth.'})
                previous={n:getattr(ref.core,n).copy() for n in previous}
            if time.perf_counter()-began>300:raise TimeoutError('Registeredwordtrial300s')
        train_seconds=time.perf_counter()-began;training_ticks=ref.ticks-ticks_before;payload=m.dumps();artifact=root/'artifacts/direct-word-bridge'/run.name;artifact.mkdir(parents=True,exist_ok=True)
        checkpoint=artifact/f'{seed}-{method}.words.zip';checkpoint.write_bytes(payload);summaries={};outputs={}
        for split,cases in (('train',train),('development',dev)):summaries[split],outputs[split]=evaluate(payload,method,cases,run,seed,split)
        learned=reference(load(payload,method));original=reference(load(initial_payload,method));ft=bos_features(learned,train);fd=bos_features(learned,dev);it=bos_features(original,train);idd=bos_features(original,dev)
        labels=np.array([ref.ids[tokenize(r['target'])[0]] for r in train]);testlabels=np.array([ref.ids[tokenize(r['target'])[0]] for r in dev]);exposures=np.array(indices)
        probe=ridge(ft,fd,labels,ref.core.config.outputs,exposures,mask=ref.mask(0));initial_probe=ridge(it,idd,labels,ref.core.config.outputs,exposures,mask=ref.mask(0))
        probe_score=float(np.mean(probe['dev']==testlabels));initial_score=float(np.mean(initial_probe['dev']==testlabels))
        lesion=load(payload,method);lesion_ref=reference(lesion)
        for n in ('w_in','w_rec'):getattr(lesion_ref.core,n)[:]=initial[n]
        lesion_summary,lesion_out=evaluate(lesion.dumps(),method,dev,run,seed,'restore_initial_hidden');summaries['restore_initial_hidden']=lesion_summary;outputs['restore_initial_hidden']=lesion_out
        assert m.dumps()==payload;copy=load(payload,method);case=train[0];learn=copy.learn(case['prompt'],case['target']);reply=copy.generate(dev[0]['prompt']);draw=int(reference(copy).sampler_rng.integers(10000))
        cold={'learn':learn,'reply':reply,'sampler_draw':draw,'state_sha256':hashlib.sha256(copy.dumps()).hexdigest()}
        row={'seed':seed,'method':method,'experiences':1024,'summary':summaries,'firstword_resethead_probe':probe_score,'initial_firstword_resethead_probe':initial_score,'probe_gain':probe_score-initial_score,
            'stratum_counts':counts,'word_updates':ref.core.updates,'hidden_updates':ref.learner.hidden_updates,'training_ticks':training_ticks,
            'core_fingerprint':ref.core.fingerprint(),'head_sha256':hashlib.sha256(ref.head.dumps()).hexdigest(),'array_bytes':ref.memory_bytes(),'checkpoint_bytes':len(payload),
            'hidden_delta_in':float(np.linalg.norm(ref.core.w_in-initial['w_in'])),'hidden_delta_rec':float(np.linalg.norm(ref.core.w_rec-initial['w_rec'])),
            'checkpoint':checkpoint.relative_to(root).as_posix(),'sha256':hashlib.sha256(payload).hexdigest(),'cold_expected':cold,'mean_training_loss':float(np.mean(losses)),
            'train_seconds':train_seconds,'seconds':time.perf_counter()-began,'resources':monitor.report(),'cap':cap,'control_only':method=='bptt_control','candidate_network_backward':False}
        if method=='bptt_control':row.update(backward_passes=m.backward_passes,surrogate_forward_ticks=m.surrogate_forward_ticks,max_graph_state_error=m.max_graph_state_error,max_graph_feature_error=m.max_graph_feature_error,max_graph_loss_error=m.max_graph_loss_error)
        monitor.check()
        if time.perf_counter()-began>300:raise TimeoutError('Wordtrial300s incl evaluation/probe')
        atomic(run/f'word-{seed}-{method}.json',{'record':row,'outputs':outputs,'neural_sample':neural});return row,neural


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args();path=Path(args.job).resolve();assert path.is_relative_to(root/'runs') and path.stat().st_size<65536
    job=json.loads(path.read_text(encoding='utf-8'));seed,method,runpath=job['trial'];assert Path(runpath).resolve()==path.parent
    if job['mode']=='cold':
        apply_memory_limit()
        if method=='bptt_control':
            from scarx.evaluation.controls import torch
            torch.set_num_threads(1)
        checkpoint=(root/job['checkpoint']).resolve();assert checkpoint.is_relative_to(root/'artifacts/direct-word-bridge') and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==job['sha256']
        _,train,dev,_,_=source_data();m=load(checkpoint.read_bytes(),method);case=train[0];learn=m.learn(case['prompt'],case['target']);reply=m.generate(dev[0]['prompt']);draw=int(reference(m).sampler_rng.integers(10000))
        atomic(path.parent/f'cold-word-{seed}-{method}.json',{'learn':learn,'reply':reply,'sampler_draw':draw,'state_sha256':hashlib.sha256(m.dumps()).hexdigest()});return
    r,_=trial(tuple(job['trial']));print(json.dumps({'seed':seed,'method':method,'dev':r['summary']['development']['macro_exact']}),flush=True)


if __name__=='__main__':main()
