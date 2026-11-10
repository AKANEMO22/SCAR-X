"""One isolated Torch ALIF control job, never a serving entrypoint."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,sys,json,time,hashlib,tempfile
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.work_trace import append
from scarx.runtime.atomic_file import replace_with_retry
from scarx.evaluation.score_word import ScoreWord
from scarx.evaluation.fixed_word_codes import fit
from train_g1_word_score_inputs import source_data
from train_g1_word_score_transfer import evaluate
from preflight_g1_word_control import stream
from generate_g1_state_paraphrases import INTENTS


def atomic(path,value):
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
        replace_with_retry(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    path=Path(args.job).resolve()
    if not path.is_relative_to(root/'runs') or path.stat().st_size>100000:raise ValueError('Job outside registered run')
    job=json.loads(path.read_text(encoding='utf-8'));run=path.parent;seed=job['seed'];encoding_name=job['input_encoding']
    if seed not in (28001,28019,28027) or encoding_name not in ('legacy','known-digest128-v1'):raise ValueError('Unregistered job')
    cap=apply_memory_limit();began=time.perf_counter();train,dev,prior,vocab,_=source_data()
    with ResourceMonitor(interval=.1) as monitor:
        from scarx.evaluation.controls import WordLearnabilityControl,torch
        torch.set_num_threads(1)
        encoding=None if encoding_name=='legacy' else fit([r['prompt'] for r in train],encoding_name)
        reference=ScoreWord(vocab,seed,'full',encoding);reference.calibrate([r['prompt'] for r in train]);model=WordLearnabilityControl(reference)
        initial={n:getattr(reference.core,n).copy() for n in ('w_in','w_rec','w_out')};previous={n:a.copy() for n,a in initial.items()};losses=[]
        for index,row in enumerate(stream(train,seed,1024),1):
            result=model.learn(row['prompt'],row['target']);losses.append(result['loss'])
            if index%64==0:
                core=reference.core;neural={'computed':48,'allocated':48,'tick':core.ticks,'spikes':core.spikes.astype(int).tolist(),
                    'voltage':np.round(core.voltage,4).tolist(),'sample_kind':'authoritativeNumPytrajectory in ISOLATED BPTT control'}
                atomic(run/'worker-progress.json',{'seed':seed,'input_encoding':encoding_name,'responses':index,
                    'loss_mean':float(np.mean(losses[-64:])),'elapsed_seconds':time.perf_counter()-began,
                    'backward_passes':model.backward_passes,'neural':neural})
                monitor.check()
                if any(not np.isfinite(a).all() for a in core.arrays().values()):raise ValueError('Nonfinite control arrays')
            if index%256==0:
                delta={n:float(np.linalg.norm(getattr(core,n)-a)) for n,a in previous.items()}
                append(run,'learning',{'scope':f'ISOLATED SURROGATE-BPTT CONTROL · {encoding_name}/seed{seed} · NOT MAIN',
                    'prompt':f'Control training responses{index-255}–{index}','passes':1,'sample_count':256,
                    'loss_mean':float(np.mean(losses[-256:])),'losses':[float(np.mean(losses[-256:]))],
                    'update_steps':core.updates,'update_unit':'word target cumulative · ISOLATED CONTROL',
                    'hidden_update_steps':core.hidden_updates,'weight_delta':delta,'attempted_weight_delta':delta,
                    'changed_synapses':{n:int(np.count_nonzero(getattr(core,n)!=a)) for n,a in previous.items()},
                    'accepted':True,'rolled_back':False,'guard_checked':False,
                    'reason':f'CONTROL ONLY: global temporal gradients, {model.backward_passes} backwardpasses. No live guard, no mainweights change; qualitypending.'})
                previous={n:getattr(core,n).copy() for n in previous}
            if time.perf_counter()-began>900:raise TimeoutError('Registered900s isolatedtrial cap')
        training_seconds=time.perf_counter()-began;payload=model.dumps()
        artifact=root/'artifacts/isolated-word-controls'/run.name;artifact.mkdir(parents=True,exist_ok=True)
        checkpoint=artifact/f'{seed}-{encoding_name}.control.zip';checkpoint.write_bytes(payload)
        # Inference uses identical existing NumPy MC8 decoder on discarded copies.
        diagnostics=[]
        for intent in INTENTS:diagnostics += [r for r in train if r['intent']==intent][:2]
        summaries={};outputs={};numpy_payload=reference.dumps()
        for split,cases in (('train16',diagnostics),('development',dev),('prior',prior)):
            summaries[split],outputs[split]=evaluate(numpy_payload,cases,seed,run,f'ISOLATED SURROGATE-BPTT CONTROL/{encoding_name}/seed{seed}/{split}')
            if time.perf_counter()-began>900:raise TimeoutError('Registered900s isolatedtrial cap incl evaluation')
        assert model.dumps()==payload
        copied=WordLearnabilityControl.loads(payload);lesson=train[0];outcome=copied.learn(lesson['prompt'],lesson['target']);reply=copied.generate(lesson['prompt'])
        cold_expected={'learn':outcome,'reply':reply,'final_sha256':hashlib.sha256(copied.dumps()).hexdigest()}
        record={'seed':seed,'input_encoding':encoding_name,'condition':'full','method':'isolated_surrogate_bptt','summary':summaries,
            'responses':1024,'word_target_updates':reference.core.updates,'hidden_updates':reference.core.hidden_updates,
            'numpy_forward_ticks':reference.core.ticks,'surrogate_forward_ticks':model.surrogate_forward_ticks,'backward_passes':model.backward_passes,
            'array_bytes':reference.core.memory_bytes(),'max_graph_state_error':model.max_graph_state_error,'max_graph_feature_error':model.max_graph_feature_error,
            'hidden_delta_in':float(np.linalg.norm(reference.core.w_in-initial['w_in'])),'hidden_delta_rec':float(np.linalg.norm(reference.core.w_rec-initial['w_rec'])),
            'train_seconds':training_seconds,'seconds':time.perf_counter()-began,'mean_training_loss':float(np.mean(losses)),
            'checkpoint':checkpoint.relative_to(root).as_posix(),'sha256':hashlib.sha256(payload).hexdigest(),
            'cold_expected':cold_expected,'resources':monitor.report(),'cap':cap,'torch_version':torch.__version__}
        monitor.check();atomic(run/f'control-{seed}-{encoding_name}.json',{'record':record,'outputs':outputs})
        print(json.dumps({'seed':seed,'input_encoding':encoding_name,'seconds':record['seconds'],'dev':summaries['development']['normalized_style_exact']}))


if __name__=='__main__':main()
