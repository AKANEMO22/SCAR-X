"""Single registered 24-trajectory P1b. No retries, no production access."""
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scope_p1b_common import ALIF,ARMS,SEEDS,ROOT,cases,old_cases,sha
from scope_p1b_trials import trial,write_json,score
from scope_p1b_statistics import contrasts,gate
from scope_p1b_resources import Monitor,memory
from preflight_scope_p1b import donors
from intake_scope_p1b import serving
sys.path.insert(0,str(ROOT/"src"))
from scarx.runtime.limits import apply_memory_limit


def prepared(seed,folder):
    tr=cases(seed,"train");dv=cases(seed,"dev");old=old_cases();teacher=ALIF(seed,kind="teacher")
    hints=[];zp=[]
    for c in tr:
        p,_,_,h,z=teacher.forward(c["sequence"],tape=True)
        hints.append(np.asarray(h,np.float32)[:,64:120]);zp.append(p)
    payload={"train_X":np.stack([c["sequence"] for c in tr]),"train_y":np.asarray([c["label"] for c in tr]),
        "train_H":np.asarray(hints),"teacher_train_p":np.asarray(zp),"partner":donors(tr),
        "train_ids":np.asarray([c["id"] for c in tr]),"dev_X":np.stack([c["sequence"] for c in dv]),
        "dev_y":np.asarray([c["label"] for c in dv]),"dev_ids":np.asarray([c["id"] for c in dv]),
        "dev_gap":np.asarray([c["gap"] for c in dv]),"dev_order":np.asarray([c["order"] for c in dv]),
        "dev_clock":np.asarray([c["clock"] for c in dv]),"old_X":np.stack([c["sequence"] for c in old]),
        "old_y":np.asarray([c["label"] for c in old]),"old_ids":np.asarray([c["id"] for c in old])}
    path=folder/f"fit-{seed}.npz";np.savez(path,**payload)
    X_hashes={sha(c["sequence"].tobytes()) for c in tr}
    if X_hashes&{sha(c["sequence"].tobytes()) for c in dv}:raise AssertionError("Train/DEV overlap")
    sealed=[]
    for cycle in (1,2):
        test=cases(seed,"test",cycle=cycle)
        if X_hashes&{sha(c["sequence"].tobytes()) for c in test}:raise AssertionError("Train/TEST overlap")
        p=folder/f"sealed-test-{seed}-c{cycle}.npz"
        np.savez(p,X=np.stack([c["sequence"] for c in test]),y=np.asarray([c["label"] for c in test]),
                 ids=np.asarray([c["id"] for c in test]))
        sealed.append({"path":str(p),"sha256":sha(p.read_bytes()),"model_forwarded":False})
    return path,{"seed":seed,"fit_sha256":sha(path.read_bytes()),"sealed_TEST":sealed,
                 "teacher_cache_forwards":teacher.forwards,"train_cases":len(tr),"dev_cases":len(dv)}


def restart(checkpoint,X,ids,folder,study,deadline):
    folder.mkdir(parents=True,exist_ok=False);inputs=folder/"X-only.npz";output=folder/"predictions.npz"
    np.savez(inputs,X=X,ids=ids)
    cmd=[sys.executable,str(ROOT/"scripts/research/restart_scope_p1b_student.py"),"--checkpoint",str(checkpoint),
         "--inputs",str(inputs),"--output",str(output),"--study-root",str(study)]
    before=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,encoding="utf-8",timeout=max(.1,min(45,deadline-time.perf_counter())))
    (folder/"stdout.txt").write_text(p.stdout,encoding="utf-8");(folder/"stderr.txt").write_text(p.stderr,encoding="utf-8")
    if p.returncode:raise RuntimeError("Independent restart failed; keep evidence")
    with np.load(output,allow_pickle=False) as z:
        prob=z["p"];meta=json.loads(z["metadata"].tobytes())
    return prob,{"wall_seconds":time.perf_counter()-before,**meta}


def continuation(records,phase,run,deadline):
    """Only reached if fixed G2 is qualified; outputs remain separately gated."""
    restarts=[];test_scores=[];test_stat_records=[]
    for r in records:
        seed=r["seed"];directory=run/r["relative_directory"]
        with np.load(phase/f"fit-{seed}.npz",allow_pickle=False) as z:
            Xd=z["dev_X"];yd=z["dev_y"];Xi=z["old_X"];yi=z["old_y"];di=z["dev_ids"];oi=z["old_ids"]
        with np.load(phase/f"sealed-test-{seed}-c1.npz",allow_pickle=False) as z:
            Xt=z["X"];yt=z["y"];ti=z["ids"]
        X=np.concatenate((Xd,Xi,Xt));ids=np.concatenate((di,oi,ti))
        p,meta=restart(directory/"final.npz",X,ids,directory/"S-only-restart",run,deadline)
        with np.load(directory/"predictions.npz",allow_pickle=False) as z:
            exact=np.array_equal(p[:len(yd)],z["dev_p_u192"])
        if not exact:raise AssertionError("Fresh process DEV prediction mismatch")
        test=score(p[len(yd)+len(yi):],yt);old=score(p[len(yd):len(yd)+len(yi)],yi)
        restarts.append({"seed":seed,"sign":r["sign"],"arm":r["arm"],"test":test,"old":old,"restart":meta})
        test_scores.append({"seed":seed,"sign":r["sign"],"arm":r["arm"],"prob":p[len(yd)+len(yi):],"y":yt})
        sd=directory/"S-only-test-scores";sd.mkdir()
        positions=[np.flatnonzero(x[:,0]) for x in Xt]
        np.savez(sd/"predictions.npz",dev_p_u192=p[len(yd)+len(yi):],dev_y=yt,
                 dev_order=np.asarray([int(np.sign(x[pos[0],1])) for x,pos in zip(Xt,positions)]),
                 dev_gap=np.asarray([int(pos[1]-pos[0]) for pos in positions]),dev_clock=np.asarray([int(pos[1]) for pos in positions]))
        test_stat_records.append({**r,"relative_directory":str(sd.relative_to(run))})
    write_json(phase/"G3-restarts.json",restarts)
    # Require held-out superiority as well; no candidate selection or updates after TEST.
    ok=True
    for seed in SEEDS:
        for sign in (1,-1):
            group={r["arm"]:r for r in restarts if r["seed"]==seed and r["sign"]==sign}
            b=group["teacher"];ok &= b["test"]["accuracy"]>=.65 and b["test"]["min_recall"]>=.65 and b["old"]["accuracy"]>=.97
            ok &= all(b["test"]["accuracy"]-group[a]["test"]["accuracy"]>=.05-1e-12 for a in ("label","shuffled","readout"))
    test_stats=contrasts(test_stat_records,run)
    write_json(phase/"G3-statistics.json",test_stats)
    ok &= all(v["CI98_333_input_conditional_Bonferroni3"][0]>0 for v in test_stats.values())
    if not ok:return {"G3":"FRESH_TEST_OR_RESTART_NO_GO","G4":"NOT_RUN_GATE"}
    from scope_p1b_cycle import cycle
    cycles=[]
    for r in records:
        if r["arm"]!="teacher":continue
        if deadline-time.perf_counter()<45:
            cycles.append({"seed":r["seed"],"sign":r["sign"],"verdict":"NOT_RUN_RESOURCE"});continue
        rr=cycle({"directory":str(run/r["relative_directory"]),"deadline":deadline,
                  "skill1_dev_accuracy":r["final_dev"]["accuracy"]})
        if rr["verdict"]=="CYCLE2_DEV_QUALIFIED_NEEDS_RESTART":
            directory=run/r["relative_directory"]/"cycle2"
            test1=cases(r["seed"],"test",cycle=1);test2=cases(r["seed"],"test",cycle=2);old=old_cases()
            all_cases=test1+test2+old;X=np.stack([c["sequence"] for c in all_cases]);ids=np.asarray([c["id"] for c in all_cases])
            p,meta=restart(directory/"S2-candidate.npz",X,ids,directory/"restart",run,deadline)
            n1,n2=len(test1),len(test2)
            rr["test1"]=score(p[:n1],np.asarray([c["label"] for c in test1]))
            rr["test2"]=score(p[n1:n1+n2],np.asarray([c["label"] for c in test2]))
            rr["old"]=score(p[n1+n2:],np.asarray([c["label"] for c in old]))
            rr["restart"]=meta
            baseline=next(z["test"]["accuracy"] for z in restarts if z["seed"]==r["seed"] and z["sign"]==r["sign"] and z["arm"]=="teacher")
            retained=all(rr[k]["accuracy"]>=.65 and rr[k]["min_recall"]>=.65 for k in ("test1","test2","old"))
            retained &= rr["test1"]["accuracy"]>=baseline-.03 and rr["old"]["accuracy"]>=.97
            rr["verdict"]="CYCLE2_RESTART_RETAINED" if retained else "CYCLE2_RESTART_NO_GO"
            write_json(directory/"restart-result.json",rr)
        cycles.append({"seed":r["seed"],"sign":r["sign"],**rr})
    write_json(phase/"G4-continuations.json",cycles)
    return {"G3":"FINITE_PILOT_RESTART_QUALIFIED","G4":cycles}


def main():
    start=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True)
    args=ap.parse_args();run=args.run.resolve();phase=run/"learning";phase.mkdir(exist_ok=False)
    pf=json.loads((run/"preflight/result.json").read_text());audit=json.loads((run/"preflight-audit-recovery.json").read_text())
    if not all(pf["gates"].values()) or not audit["audit_pass"] or not all(audit["independently_reconstructed_gates"].values()):raise RuntimeError("Prerequisite failed")
    if json.loads((run/"backend-selection.json").read_text())["backend"]!="cpu":raise RuntimeError("Registration freezes chosen CPU")
    source_names=["experiments/scope-p1b-temporal-transfer-v1.md",*[
        "scripts/research/"+name for name in ("scope_p1b_common.py","scope_p1b_trials.py","scope_p1b_statistics.py",
            "scope_p1b_cycle.py","restart_scope_p1b_student.py","run_scope_p1b_learning.py","audit_scope_p1b_learning.py")]]
    source_hashes={}
    for name in source_names:
        b=(ROOT/name).read_bytes();dest=phase/"source"/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b);source_hashes[name]=sha(b)
    cap=apply_memory_limit(2_000_000_000);monitor=Monitor();monitor.start();before=serving()
    prior=75.;deadline=start+1800-prior
    write_json(phase/"registration.json",{"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "dirty_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True).strip(),"source_hashes":source_hashes,
        "seeds":list(SEEDS),"signs":[1,-1],"arms":list(ARMS),"max_trials":24,"updates_each":192,"backend":"CPU",
        "reserved_previous_execution_seconds":prior,"global_limit_seconds":1800,"cap":cap,"serving_before":before,
        "UTC_first_preparation":dt.datetime.now(dt.timezone.utc).isoformat()})
    models=[]
    try:
        datasets={};data_info=[]
        for seed in SEEDS:datasets[seed],info=prepared(seed,phase);data_info.append(info)
        write_json(phase/"data-lock.json",data_info) # Before first .learn in any child.
        payloads=[]
        for seed in SEEDS:
            for sign in (1,-1):
                for arm in ARMS:
                    directory=phase/"trials"/f"{seed}-{sign}-{arm}"
                    payloads.append({"seed":seed,"sign":sign,"arm":arm,"directory":str(directory),"data_path":str(datasets[seed]),"deadline":deadline})
        write_json(phase/"trial-matrix.json",payloads)
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures={executor.submit(trial,p):p for p in payloads}
            for future in as_completed(futures,timeout=max(1,deadline-time.perf_counter())):
                record=future.result();p=futures[future];record["relative_directory"]=str(Path(p["directory"]).relative_to(run))
                models.append(record);write_json(phase/"completed.json",models)
                print(json.dumps({"completed":len(models),"seed":record["seed"],"sign":record["sign"],"arm":record["arm"],
                      "DEV":record["final_dev"],"updates":record["updates"],"seconds":record["wall_seconds"]}),flush=True)
                if monitor.exceeded:raise MemoryError("Experimenttree exceeded2GB")
        stats=contrasts(models,run);decision=gate(models,stats)
        write_json(phase/"statistics.json",stats);write_json(phase/"G2-decision.json",decision)
        further=continuation(models,phase,run,deadline) if decision["pilot_gate_pass"] else {"G3":"NOT_RUN_G2_FAILED","G4":"NOT_RUN_G2_FAILED"}
        write_json(phase/"conditional-stages.json",further)
        result={"verdict":"PILOT_SIGNAL_CONDITIONAL_ONLY" if decision["pilot_gate_pass"] else "NO_TEACHER_SPECIFIC_BENEFIT_OR_LEARNER_NO_GO",
                "trials_completed":len(models),"primary_learning_updates":sum(r["updates"] for r in models),
                "primary_hidden_updates":sum(r["hidden_updates"] for r in models),
                "primary_neural_forwards":sum(r["neural_forwards"] for r in models),
                "teacher_cache_forwards":sum(d["teacher_cache_forwards"] for d in data_info),"G2":decision,
                "conditional_stages":further,"wall_seconds":time.perf_counter()-start,
                "total_seconds_with_prior_reservation":prior+time.perf_counter()-start,
                "teacher_hand_engineered":True,"population_CI":"INCONCLUSIVE_N3","continual_learning_proven":False,
                "serving_unchanged":serving()==before,"TEST_model_forwarded":bool(decision["pilot_gate_pass"])}
        write_json(phase/"result.json",result);print(json.dumps(result),flush=True)
    except Exception as ex:
        write_json(phase/"failure.json",{"error":repr(ex),"completed_trials":len(models),"wall_seconds":time.perf_counter()-start,
                   "keep_all_partial_results":True});raise
    finally:
        write_json(phase/"resource.json",{"tree":monitor.close(),"parent_OS_peaks":memory(),"serving_after":serving(),
                   "wall_seconds":time.perf_counter()-start,"prior_reserved_seconds":prior})


if __name__=="__main__":main()
