"""Separate CUDA local traces and isolated CUDA surrogate-credit control, no SGD."""

from dataclasses import asdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.store import Store, encode
from scarx.hybrid.device_budget import gpu_headroom


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def alignment(candidate, reference):
    c, r = candidate.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    cn, rn = float(np.linalg.norm(c)), float(np.linalg.norm(r))
    mask = np.abs(r) > 1e-8
    return {"candidate_norm":cn,"surrogate_reference_norm":rn,
            "cosine":float(np.dot(c,r)/(cn*rn)) if cn*rn > 0 else None,
            "sign_agreement_on_reference_abs_gt_1e_8":float(np.mean(np.sign(c[mask]) == np.sign(r[mask]))) if mask.any() else None,
            "reference_nonzero_coordinates":int(mask.sum()),"norm_ratio":cn/rn if rn > 0 else None}


def frozen_learning_state(model):
    from diagnose_hybrid_predicate_fit import immutable_state
    return immutable_state(model,None) | {
        name:sha(model.host(getattr(model,name)).tobytes())
        for name in ("syn_trace","syn_energy","gamma")} | {
        "indices":sha(model.host(model.core.indices).tobytes()),"core_updates":model.core.updates}


def reference_stage(run, selected, parent, started):
    from scarx.hybrid.learner import LocalALIF
    from scarx.evaluation.predicate_retention_dev import pair_sequence
    records=[]
    for seed in (260113,260137,260171):
        for state_name in ("initial","final640"):
            assert gpu_headroom()["free_MiB"] >= 512
            path = parent / f"{seed}-current_hidden/final-full-state.bin"
            payload = path.read_bytes() if state_name == "final640" else None
            model = LocalALIF.loads(payload,device="gpu")[0] if payload else LocalALIF(seed).to_device("gpu")
            before=frozen_learning_state(model)
            state={name:model.host(getattr(model.core,name)).copy() for name in ("w_in","w_rec","w_out","bias","indices")}
            state.update(f_mean=model.host(model.f_mean).copy(),f_var=model.host(model.f_var).copy(),
                         config=np.frombuffer(encode(asdict(model.core.config)).encode(),dtype=np.uint8))
            state_file=f"{seed}-{state_name}-state.npz"
            np.savez(run/state_file,**state)
            for case in selected:
                assert time.monotonic()-started<300
                assert gpu_headroom()["free_MiB"] >= 512
                seq=pair_sequence(case)
                p,features,raw,scale,indices=model.forward(seq,head="relevance",track=True)
                xp=model.xp
                error=-xp.asarray(p)
                error[case["target"]]+=1
                signal=(model.core.w_out[:2].T@error)*(xp.abs(xp.asarray(raw))<5)/xp.asarray(scale)
                direction=signal[:,None]*model.e_f
                trace=.9*model.syn_trace+.1*direction
                energy=.99*model.syn_energy+.01*direction**2
                prospective=xp.clip(.008*trace/(xp.sqrt(energy)+1e-4),-.01,.01)
                weights=xp.concatenate((model.core.w_in,model.core.w_rec),axis=1)
                actual=xp.clip(weights+prospective,-3,3)-weights
                local,delta=model.host(direction).copy(),model.host(actual).copy()
                expected_state=[model.host(getattr(model.core,name)).copy()
                                for name in ("voltage","adaptation","spikes","features")]
                model.core.reset_state()
                for name in ("e_v","e_a","e_f"):getattr(model,name).fill(0)
                spikes=[]
                for row in seq:
                    model.step(row,track=True)
                    spikes.append(model.host(model.core.spikes).copy())
                final=np.stack([model.host(getattr(model.core,name))
                                for name in ("voltage","adaptation","spikes","features")])
                np.testing.assert_array_equal(final,np.stack(expected_state))
                name=f"case-{len(records):02d}.npz"
                np.savez(run/name,sequence=seq,spikes=np.stack(spikes),final_states=final,
                         local_direction=local,prospective_delta=delta,probabilities=p,
                         target=np.array(case["target"],np.int64))
                records.append({"id":case["id"],"api":case["api"],"seed":seed,"state":state_name,
                    "state_file":state_file,"case_file":name,"target":case["target"],
                    "source_checkpoint_sha256":sha(payload) if payload else None,
                    "feedback_clipped_fraction":float(np.mean(np.abs(raw)>=5)),
                    "native_loss":-float(np.log(max(float(p[case["target"]]),1e-12))),
                    "local_direction_norm":float(np.linalg.norm(local.astype(np.float64))),
                    "prospective_delta_norm":float(np.linalg.norm(delta.astype(np.float64)))})
            assert frozen_learning_state(model)==before
            del model
    assert len(records)==48
    with (run/'reference-records.json').open('x',encoding='utf-8') as f:f.write(encode(records))
    return records


def control_stage(run):
    import torch
    from scarx.evaluation.controls import hybrid_tape_credit_direction
    if not torch.cuda.is_available():raise RuntimeError("Diagnostic requires isolated TorchCUDA")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert gpu_headroom()["free_MiB"]>=1800
    records=json.loads((run/'reference-records.json').read_text())
    results=[]
    started=time.monotonic()
    trace=Store(ROOT/'local/hybrid-v1')
    try:
        for row in records:
            assert time.monotonic()-started<300
            assert torch.cuda.mem_get_info()[0]>=512*1048576
            with np.load(run/row['state_file'],allow_pickle=False) as z:
                state={key:z[key].copy() for key in z.files if key!='config'}
                state['config']=json.loads(z['config'].tobytes())
            with np.load(run/row['case_file'],allow_pickle=False) as z:
                arrays={key:z[key].copy() for key in z.files}
            result=hybrid_tape_credit_direction(state,arrays['sequence'],arrays['spikes'],
                                               arrays['final_states'],int(arrays['target']),device='cuda:0')
            np.testing.assert_allclose(result['probabilities'],arrays['probabilities'],atol=1e-5,rtol=0)
            assert abs(result['loss']-row['native_loss'])<1e-5
            direction=result.pop('hidden_direction');inputs=result.pop('input_direction')
            file=row['case_file'].replace('.npz','-surrogate.npz')
            np.savez(run/file,hidden_direction=direction,input_direction=inputs,probabilities=result.pop('probabilities'))
            record={**row,**result,'local_vs_surrogate':alignment(arrays['local_direction'],direction),
                'prospective_delta_vs_surrogate':alignment(arrays['prospective_delta'],direction),
                'query_input_credit_norm':float(np.linalg.norm(inputs[:8].astype(np.float64))),
                'record_input_credit_norm':float(np.linalg.norm(inputs[8:].astype(np.float64)))}
            results.append(record)
            trace.event('credit_direction.actual_control_comparison',{
                'run_id':run.name,**record,'new_neural_updates':0,'surrogate_control_only':True},'orchestrator')
        with (run/'control-result.json').open('x',encoding='utf-8') as f:f.write(encode({
            'records':results,'seconds':time.monotonic()-started,'backward_passes':len(results),
            'optimizer_steps':0,'live_model_changed':False,'neural_device':'cuda:0'}))
    finally:trace.close()
    print(encode({'control_records':len(results),'optimizer_steps':0}),flush=True)


def main():
    from scarx.hybrid.resources import ResourceMonitor
    from scarx.evaluation.predicate_retention_dev import FAMILIES
    parent=ROOT/'runs/20261114T035022Z-predicate-retention-dev-v1-032f30'
    data_raw=(parent/'locked-owned-data.json').read_bytes()
    data=json.loads(data_raw);training=sum(data['train_stages'],[])
    selected=[next(r for r in training if r['api']==api and r['target']==target)
              for api in FAMILIES for target in (1,0)]
    assert gpu_headroom()['free_MiB']>=1800
    name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-credit-direction-diagnostic-v1-'+uuid.uuid4().hex[:6]
    run=ROOT/'runs'/name;run.mkdir()
    (run/'locked-source-cases.json').write_text(encode(selected),encoding='utf-8')
    paths=list((ROOT/'src/scarx/hybrid').glob('*.py'))+[
        Path(__file__),ROOT/'scripts/diagnose_hybrid_predicate_fit.py',ROOT/'src/scarx/neural/core.py',
        ROOT/'src/scarx/evaluation/controls.py',ROOT/'src/scarx/evaluation/readout_eligibility.py',
        ROOT/'src/scarx/evaluation/predicate_retention_dev.py',ROOT/'src/scarx/evaluation/tool_query_oracle.py',
        ROOT/'experiments/hybrid-credit-direction-diagnostic-v1.md']
    hashes={}
    for path in paths:
        rel,raw=path.relative_to(ROOT),path.read_bytes();dest=run/'source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);hashes[rel.as_posix()]=sha(raw)
    (run/'manifest.json').write_text(encode({'source_sha256':hashes,'parent_data_sha256':sha(data_raw),
        'selected_ids':[r['id'] for r in selected],'max_backward_passes':48,'max_reference_rollouts':96,
        'max_seconds_per_stage':300,'new_training':False,'backward_only_in_evaluation_controls':True}),encoding='utf-8')
    monitor=ResourceMonitor(limit=None);monitor.start();trace=Store(ROOT/'local/hybrid-v1')
    started=time.monotonic();report={'run_id':name,'failures':[],'new_neural_training':False,'candidate_promoted':False}
    try:
        rows=reference_stage(run,selected,parent,started)
        report['reference_rollouts']=len(rows)*2
        child=ROOT/'local/hybrid-v1/reference-env/Scripts/python.exe'
        with (run/'isolated-control.log').open('wb') as log:
            result=subprocess.run([str(child),str(Path(__file__)),'--control',str(run)],cwd=ROOT,
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,
                timeout=300,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert result.returncode==0,'Isolated control failed; preserve child log'
        control=json.loads((run/'control-result.json').read_text())
        assert control['backward_passes']==48 and control['optimizer_steps']==0
        report['groups']=[]
        for state_name in ('initial','final640'):
            cases=[r for r in control['records'] if r['state']==state_name]
            metric={}
            for kind in ('local_vs_surrogate','prospective_delta_vs_surrogate'):
                values=[r[kind]['cosine'] for r in cases if r[kind]['cosine'] is not None]
                metric[kind]={'cosine_median':float(np.median(values)),'cosine_min':min(values),'cosine_max':max(values),
                             'case_count':len(cases),'nonzero_norm_cases':len(values)}
            report['groups'].append({'state':state_name,**metric})
        for rel,digest in hashes.items():assert sha((ROOT/rel).read_bytes())==digest
        report['verdict']='CREDIT_DIRECTION_DIAGNOSTIC_ONLY_NO_LEARNING_OR_QUALITY_PASS'
    except Exception:
        report['failures'].append(traceback.format_exc());report['verdict']='CREDIT_DIRECTION_DIAGNOSTIC_INCOMPLETE'
    finally:
        report['seconds']=time.monotonic()-started;report['resources']=monitor.close()
        with (run/'result.json').open('x',encoding='utf-8') as f:f.write(encode(report))
        trace.event('credit_direction.completed',{k:v for k,v in report.items() if k!='resources'},'orchestrator');trace.close()
        print(encode({k:v for k,v in report.items() if k!='resources'}),flush=True)
    return not report['failures']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path);a=p.parse_args()
    if a.control:control_stage(a.control.resolve())
    else:raise SystemExit(0 if main() else 1)
