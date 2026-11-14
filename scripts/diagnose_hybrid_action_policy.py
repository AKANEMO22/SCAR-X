"""Recorded-state CUDA action ablations; inference only, no skill-quality claim."""

import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import time
import traceback
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from scarx.hybrid.store import Store,encode
from scarx.hybrid.learner import LocalALIF,ACTIONS
from scarx.hybrid.device_budget import gpu_headroom
from scarx.evaluation.shared_native import SharedNative


def fingerprint(model):
    names=("core.w_in","core.w_rec","core.w_out","core.bias","syn_trace","syn_energy","importance","gamma","f_mean","f_var")
    arrays=model.arrays()
    return {name:hashlib.sha256(model.host(arrays[name]).tobytes()).hexdigest() for name in names}


def main():
    begun=time.monotonic()
    attribution=ROOT/"runs/20261114T102423Z-learning-target-audit-v1-70f370"
    parent=ROOT/"runs/20261114T011841Z-shared-endurance-v1-554c19"
    claims=json.loads((attribution/"new-update-attribution.json").read_text(encoding="utf-8"))
    assert len(claims)==20 and not any(row["unknown"] for row in claims)
    database=(ROOT/"local"/parent.name/"hybrid.sqlite").resolve()
    with sqlite3.connect(database.as_uri()+"?mode=ro",uri=True) as db:
        pointer=json.loads(db.execute("SELECT payload FROM objects WHERE kind='runtime' AND id='active_checkpoint'").fetchone()[0])
        digest,payload=db.execute("SELECT digest,payload FROM checkpoints WHERE id=?",(pointer["id"],)).fetchone()
    assert hashlib.sha256(payload).hexdigest()==digest
    with np.load(io.BytesIO(payload),allow_pickle=False) as saved:
        meta=json.loads(saved["metadata"].tobytes())
    decisions=meta["controller"]["decisions"]
    rows=[]
    ledger=json.loads((attribution/"event-ledger.json").read_text(encoding="utf-8"))
    event_by_id={row["id"]:row for row in ledger}
    for claim in claims:
        recorded=event_by_id[claim["decision_event"]]["payload"]
        decision=decisions[recorded["id"]]
        assert decision["action"]==recorded["action"]=="verify"
        rows.append({"decision_id":recorded["id"],"sequence":decision["sequence"],
                     "allowed":recorded["allowed"],"logged_action":recorded["action"],
                     "logged_proposed_action":recorded["proposed_action"],"target":ACTIONS.index("verify")})
    name=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+"-action-policy-counterfactual-v1-"+uuid.uuid4().hex[:6]
    run=ROOT/"runs"/name;run.mkdir()
    (run/"locked-cases.json").write_text(encode(rows),encoding="utf-8")
    files=list((ROOT/"src/scarx/hybrid").glob('*.py'))+[Path(__file__),ROOT/"src/scarx/neural/core.py",
        ROOT/"src/scarx/evaluation/readout_eligibility.py",ROOT/"experiments/hybrid-action-policy-counterfactual-v1.md"]
    hashes={}
    for path in files:
        rel=path.relative_to(ROOT);dest=run/"source"/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());hashes[rel.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    (run/"manifest.json").write_text(encode({"run_id":name,"source_sha256":hashes,"parent":parent.name,
        "parent_checkpoint_sha256":digest,"locked_cases_sha256":hashlib.sha256((run/"locked-cases.json").read_bytes()).hexdigest(),
        "seed":260113,"max_forward_attempts":80,"new_learning_steps":0,"known_seen_states_not_TEST":True}),encoding="utf-8")
    report={"run_id":name,"failures":[],"attempted_GPU_forwards":0,"cells":[],"GPU_samples":[],
            "live_weights_or_model_source_changed":False,"learning_steps":0,"VLM_ASR_calls":0}
    arrays={}
    trace=Store(ROOT/"local/hybrid-v1")
    try:
        native=SharedNative(ROOT);native.check()
        report["GPU_samples"].append(gpu_headroom())
        if report["GPU_samples"][-1]["free_MiB"]<2048:raise MemoryError("Insufficient startupheadroom for private GPUdiagnostic")
        initial=LocalALIF(seed=260113,mode="frozen").to_device('gpu')
        final,_=LocalALIF.loads(payload,device='gpu')
        initial_bytes=initial.dumps()
        initial_w={key:initial.host(value).copy() for key,value in initial.core.arrays().items() if key in ('w_in','w_rec','w_out','bias','indices')}
        np.testing.assert_array_equal(initial_w['indices'],final.host(final.core.indices))
        for arm in ('initial','final','hidden_reset','readout_reset'):
            model,_=LocalALIF.loads(initial_bytes if arm=='initial' else payload,device='gpu')
            reset=('w_in','w_rec') if arm=='hidden_reset' else ('w_out','bias') if arm=='readout_reset' else ()
            for key in reset:getattr(model.core,key)[:]=model.xp.asarray(initial_w[key])
            before=fingerprint(model)
            probabilities=[];standardized=[];selected=[]
            for index,row in enumerate(rows):
                native.check()
                while native.status().get('cognition',{}).get('phase')=='serving':
                    if time.monotonic()-begun>180:raise TimeoutError("Foreground priority/duration limit")
                    time.sleep(.1)
                if report["attempted_GPU_forwards"]>=80:raise RuntimeError("Forwardattempt cap")
                report["attempted_GPU_forwards"]+=1
                p,f,*_=model.forward(np.asarray(row['sequence'],np.float32),head='action',track=False)
                choice=max((ACTIONS.index(a) for a in row['allowed']),key=lambda k:float(p[k]))
                probabilities.append(p);standardized.append(f);selected.append(ACTIONS[choice])
            assert fingerprint(model)==before
            if model.device!='gpu' or model.xp.cuda.runtime.getDevice()!=0:raise ValueError("Actual neuraldevice mismatch")
            for key in ('w_in','w_rec','w_out','bias'):arrays[arm+'_'+key]=model.host(getattr(model.core,key)).copy()
            arrays[arm+'_probabilities']=np.stack(probabilities)
            arrays[arm+'_standardized']=np.stack(standardized)
            cell={'arm':arm,'selected_actions':selected,'verify_choices':selected.count('verify'),
                  'recorded_states':len(rows),'parameter_fingerprints_unchanged':before,
                  'actual_device':'CUDA0','forward_only_no_updates':True}
            report['cells'].append(cell)
            report['GPU_samples'].append(gpu_headroom())
            if report['GPU_samples'][-1]['free_MiB']<512:raise MemoryError("PhysicalGPUreserve")
            trace.event('action_policy_counterfactual.arm_measured',{'run_id':name,**cell},'orchestrator')
            del model
        assert report['attempted_GPU_forwards']==80
        assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==value for rel,value in hashes.items())
        native.check()
        report.update(verdict='RECORDED_ACTION_ABLATIONS_MEASURED_NOT_UNSEEN_SKILL_PASS',native_owner_unchanged=True,
            final_future_training_relative_to_earlier_states_disclosed=True,formal_quality_or_retention_PASS=False)
    except Exception:
        report['failures'].append(traceback.format_exc());report['verdict']='ACTION_POLICY_DIAGNOSTIC_INCOMPLETE'
    finally:
        report['seconds']=time.monotonic()-begun
        np.savez(run/'actual-CUDA-forward-arrays.npz',**arrays)
        with (run/'result.json').open('x',encoding='utf-8') as out:out.write(encode(report))
        trace.event('action_policy_counterfactual.completed',{k:v for k,v in report.items() if k!='cells'},'orchestrator');trace.close()
    print(encode({k:report[k] for k in ('run_id','verdict','attempted_GPU_forwards','failures','seconds')}))


if __name__=='__main__':main()
