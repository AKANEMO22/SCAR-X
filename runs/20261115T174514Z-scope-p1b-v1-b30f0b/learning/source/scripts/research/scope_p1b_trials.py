"""Registered primary trajectories and reusable stage functions, research only."""
import csv
import json
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"
from pathlib import Path
import sys
import time
import numpy as np
from scope_p1b_common import ALIF, COLS, ROOT, sha
from scope_p1b_resources import memory
sys.path.insert(0,str(ROOT/"src"))
from scarx.runtime.limits import apply_memory_limit


def write_json(path,value):
    def native(x):
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=native)+"\n",encoding="utf-8")


def evaluate(model,X,old=False):
    return np.asarray([model.forward(x,old=old)[0] for x in X])


def score(prob,y):
    correct=prob.argmax(axis=1)==y
    recall=[float(correct[y==k].mean()) for k in (0,1)]
    return {"accuracy":float(correct.mean()),"class_recall":recall,"min_recall":min(recall)}


def old_identity(model):
    raw=model.core.w_in[:8].tobytes()+model.core.w_rec.tobytes()+model.core.indices.tobytes()+model.old_W.tobytes()+model.old_bias.tobytes()
    return sha(raw)


def fit(model,X,y,H,arm,deadline,directory,dev_X,dev_y,old_X,old_y,partner=None,points=None):
    n=len(y);points=points or (0,n//4,n//2,n)
    predictions={};curves=[];diagnostic_E=[];diagnostic_D=[]
    directions=[];banks=[];probs=[];heads=[];biases=[];moments=[];energies=[];weights=[];endpoints=[]
    original_old=old_identity(model);initial=model.core.w_in.copy()
    with (directory/"learning.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["update","loss","before_correct","actual_hidden_delta_L2","proposed_hidden_delta_L2",
                                               "direction_L2","hint_bank_L2","eligibility_L2","elapsed_seconds"])
        writer.writeheader();started=time.perf_counter()
        for index in range(n+1):
            if time.perf_counter()>deadline:raise TimeoutError("Registered worker/global budget")
            if index in points:
                dp=evaluate(model,dev_X);op=evaluate(model,old_X,old=True)
                predictions[f"dev_p_u{index}"]=dp;predictions[f"old_p_u{index}"]=op
                curves.append({"update":index,"dev":score(dp,dev_y),"old":score(op,old_y)})
                write_json(directory/"progress.json",{"pid":os.getpid(),"completed_updates":index,"curves":curves,
                           "forward_calls":model.forwards,"actual_learning_updates":model.updates})
            if index==n:break
            hint_index=int(partner[index]) if arm=="shuffled" else index
            before=model.core.w_in[:,COLS].copy()
            heads.append(model.core.w_out.copy());biases.append(model.core.bias.copy());weights.append(before.copy())
            info=model.learn(X[index],int(y[index]),H[hint_index],arm)
            realized=model.core.w_in[:,COLS]-before
            actual=float(np.linalg.norm(realized))
            if abs(actual-info["hidden_delta_L2"])>1e-6:raise AssertionError("Actual physical update differs")
            if old_identity(model)!=original_old:raise AssertionError("Frozen old/R/topology invariant")
            if not all(np.isfinite(a).all() for a in model.core.arrays().values()):raise FloatingPointError("Nonfinite state")
            if np.max(np.abs(model.core.w_in))>3 or np.max(np.abs(model.core.w_out))>3:raise AssertionError("Bounds")
            writer.writerow({"update":index+1,"loss":info["loss"],"before_correct":info["before_correct"],
                             "actual_hidden_delta_L2":actual,"proposed_hidden_delta_L2":info["hidden_delta_L2"],
                             "direction_L2":info["direction_L2"],"hint_bank_L2":info["hint_bank_L2"],
                             "eligibility_L2":float(np.linalg.norm(model.ef)),"elapsed_seconds":time.perf_counter()-started})
            diagnostic_E.append(model.ef.copy())
            diagnostic_D.append(realized.copy())
            directions.append(info["direction_array"]);banks.append(info["hint_bank_array"])
            probs.append(info["probabilities_before"]);moments.append(model.mom.copy());energies.append(model.energy.copy())
            endpoints.append(np.clip(model.core.features/.1,-5,5).copy())
            if (index+1)%16==0:handle.flush()
    if arm=="readout" and not np.array_equal(initial,model.core.w_in):raise AssertionError("RO hidden changed")
    np.savez(directory/"learning-diagnostics.npz",eligibility=np.asarray(diagnostic_E),actual_hidden_deltas=np.asarray(diagnostic_D),
             directions=np.asarray(directions),hint_banks=np.asarray(banks),p_before=np.asarray(probs),
             head_before=np.asarray(heads),bias_before=np.asarray(biases),mom_after=np.asarray(moments),energy_after=np.asarray(energies),
             weights_before=np.asarray(weights),q_endpoint=np.asarray(endpoints))
    return predictions,curves,initial


def trial(payload):
    start=time.perf_counter();seed=payload["seed"];sign=payload["sign"];arm=payload["arm"]
    directory=Path(payload["directory"]);directory.mkdir(parents=True,exist_ok=False)
    cap=apply_memory_limit(2_000_000_000);model=ALIF(seed,sign)
    initial_blob=model.dumps();(directory/"initial.npz").write_bytes(initial_blob)
    deadline=min(payload["deadline"],start+45)
    try:
        with np.load(payload["data_path"],allow_pickle=False) as z:
            X=z["train_X"];y=z["train_y"];H=z["train_H"];partner=z["partner"]
            dev_X=z["dev_X"];dev_y=z["dev_y"];old_X=z["old_X"];old_y=z["old_y"]
            metadata={key:z[key] for key in ("dev_ids","dev_gap","dev_order","dev_clock")}
        pred,curves,initial_weights=fit(model,X,y,H,arm,deadline,directory,dev_X,dev_y,old_X,old_y,partner)
        final_blob=model.dumps();(directory/"final.npz").write_bytes(final_blob)
        reverted=ALIF.loads(final_blob);reverted.core.w_in[:]=initial_weights
        before=reverted.forwards;rp=evaluate(reverted,dev_X);revert_forwards=reverted.forwards-before
        pred.update({"revert_hidden_p":rp,"dev_y":dev_y,"old_y":old_y,**metadata})
        np.savez(directory/"predictions.npz",**pred)
        data_sha=sha(Path(payload["data_path"]).read_bytes())
        result={"status":"COMPLETED","seed":seed,"sign":sign,"arm":arm,"pid":os.getpid(),
                "updates":model.updates,"hidden_updates":model.hidden_updates,"neural_forwards":model.forwards+revert_forwards,
                "final_dev":curves[-1]["dev"],"final_old":curves[-1]["old"],"curves":curves,
                "hidden_revert":score(rp,dev_y),"hidden_weight_drift_L2":float(np.linalg.norm(model.core.w_in-initial_weights)),
                "initial_checkpoint_sha256":sha(initial_blob),"final_checkpoint_sha256":sha(final_blob),
                "data_sha256":data_sha,"wall_seconds":time.perf_counter()-start,"process_memory":memory(),
                "memory_cap":cap,"checkpoint_kind":"S64_student_only_no_P","teacher_at_inference":False,
                "not_20stage_retention":True}
        write_json(directory/"result.json",result);return result
    except Exception as ex:
        (directory/"partial.npz").write_bytes(model.dumps())
        result={"status":"FAILED_KEEP_PARTIAL","error":repr(ex),"seed":seed,"sign":sign,"arm":arm,
                "updates":model.updates,"hidden_updates":model.hidden_updates,"neural_forwards":model.forwards,
                "wall_seconds":time.perf_counter()-start,"pid":os.getpid(),"process_memory":memory()}
        write_json(directory/"result.json",result);raise
