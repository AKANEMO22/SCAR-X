"""Frozen P1b feature extraction, teacher-blind ridge probes and information audit.

No call to learn/proposal, no TEST arrays, no checkpoint mutation or live access.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid
import numpy as np
import psutil
from scope_p1b_common import ALIF,cuda,ROOT
from scope_p1b_resources import Monitor,memory,gpu
sys.path.insert(0,str(ROOT/"src"))
from scarx.runtime.limits import apply_memory_limit

OLD="runs/20261115T174514Z-scope-p1b-v1-b30f0b"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("x",encoding="utf-8") as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    os.replace(temp,path)


def hidden_key(path):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(z["metadata"].tobytes());h=hashlib.sha256()
        for key in ("core_w_in","core_w_rec","core_indices"):
            h.update(z[key].tobytes())
        h.update(json.dumps(meta["config"],sort_keys=True).encode())
        return h.hexdigest()


def matrix(model,X,deadline=None,stop=None):
    f=[];q=[];probs=[];spikes=np.zeros(64,np.int64);sat=0
    for x in X:
        if deadline is not None and time.perf_counter()>deadline:raise TimeoutError("Episode deadline")
        if stop is not None and Path(stop).exists():raise RuntimeError("Research controller requested stop")
        p,_,qq,h,zz=model.forward(x,tape=True)
        f.append(model.host(model.core.features).copy());q.append(qq);probs.append(p)
        spikes+=np.asarray(zz,dtype=np.int64).sum(0)
        sat+=int((np.abs(np.asarray(h))>=5).sum())
    return np.asarray(f),np.asarray(q),np.asarray(probs),spikes,sat


def profile(backend,out,checkpoint,data):
    start=time.perf_counter();out.mkdir(exist_ok=False);cap=apply_memory_limit(800_000_000)
    before=gpu();xp=np;handles=[]
    if backend=="gpu":
        if before["free_MiB"]<1536:raise MemoryError("NVML reserve before CUDA")
        xp,handles=cuda()
    model=ALIF.loads(checkpoint.read_bytes(),xp=xp)
    with np.load(data,allow_pickle=False) as z:X=z["train_X"][:2]
    frozen=hashlib.sha256(model.host(model.core.w_in).tobytes()).hexdigest()
    def sync():
        if xp is not np:xp.cuda.get_current_stream().synchronize()
    matrix(model,X[:1]);sync();dur=[];result=None
    for _ in range(3):
        sync();t=time.perf_counter();result=matrix(model,X);sync();dur.append(time.perf_counter()-t)
    assert frozen==hashlib.sha256(model.host(model.core.w_in).tobytes()).hexdigest()
    np.savez(out/"outputs.npz",f=result[0],q=result[1],p=result[2],spikes=result[3])
    after=gpu()
    if backend=="gpu" and after["free_MiB"]<1024:raise MemoryError("NVML reserve after CUDA")
    record={"backend":backend,"episode_forwards":7,"hidden_updates":0,"median_two_episodes_seconds":statistics.median(dur),
            "repeat_seconds":dur,"wall_seconds":time.perf_counter()-start,"memory":memory(),"cap":cap,
            "GPU_before":before,"GPU_after":after,"full_perprocess_VRAM_peak":"UNKNOWN_WDDM",
            "cupy_pool_bytes":int(xp.get_default_memory_pool().total_bytes()) if xp is not np else 0}
    write(out/"result.json",record);return record


def calibration(payload):
    start=time.perf_counter();apply_memory_limit(350_000_000)
    model=ALIF.loads(Path(payload["checkpoint"]).read_bytes());before=model.forwards
    with np.load(payload["data"],allow_pickle=False) as z:X=z["train_X"][:64]
    matrix(model,X,payload["deadline"],payload["stop"])
    return {"pid":os.getpid(),"episode_forwards":model.forwards-before,"seconds":time.perf_counter()-start,"memory":memory()}


def extract(payload):
    start=time.perf_counter();directory=Path(payload["directory"]);directory.mkdir(exist_ok=False)
    apply_memory_limit(350_000_000);checkpoint=Path(payload["checkpoint"])
    try:
        model=ALIF.loads(checkpoint.read_bytes());before=model.forwards
        weights={k:model.host(v).copy() for k,v in model.core.arrays().items() if k in ("w_in","w_rec","w_out","indices","bias")}
        with np.load(payload["data"],allow_pickle=False) as z:X=z["train_X"];DX=z["dev_X"]
        train=matrix(model,X,payload["deadline"],payload["stop"]);dev=matrix(model,DX,payload["deadline"],payload["stop"])
        assert all(np.array_equal(v,model.host(getattr(model.core,k))) for k,v in weights.items())
        np.savez(directory/"features.npz",train_f=train[0],train_q=train[1],dev_f=dev[0],dev_q=dev[1],
                 train_spikes=train[3],dev_spikes=dev[3],saturation=np.asarray([train[4],dev[4]]))
        result={"status":"DONE","key":payload["key"],"seed":payload["seed"],"pid":os.getpid(),
                "checkpoint_sha256":sha(checkpoint),"data_sha256":sha(Path(payload["data"])),
                "forwards":model.forwards-before,"hidden_updates":0,"memory":memory(),
                "wall_seconds":time.perf_counter()-start,"artifact_sha256":sha(directory/"features.npz")}
        write(directory/"result.json",result);return result
    except Exception as e:
        write(directory/"failure.json",{"error":repr(e),"keep_partial":True,"seconds":time.perf_counter()-start});raise


def metrics(scores,y):
    scores=np.where(np.abs(scores)<=1e-12,0,scores)
    predicted=(scores>0).astype(np.int64);conf=np.zeros((2,2),np.int64)
    for a,b in zip(y,predicted):conf[a,b]+=1
    recalls=np.diag(conf)/conf.sum(1);pos=scores[y==1];neg=scores[y==0]
    auc=float((pos[:,None]>neg[None,:]).mean()+.5*(pos[:,None]==neg[None,:]).mean())
    margin=(2*y-1)*scores
    return {"accuracy":float((predicted==y).mean()),"balanced_accuracy":float(recalls.mean()),
            "negative_recall":float(recalls[0]),"positive_recall":float(recalls[1]),"AUC":auc,
            "mean_signed_margin":float(margin.mean()),"min_signed_margin":float(margin.min()),"confusion":conf.tolist()}


def ridge(X,y,DX):
    x=X.astype(np.float64);dx=DX.astype(np.float64);mean=x.mean(0);scale=np.maximum(x.std(0),1e-6)
    a=np.column_stack(((x-mean)/scale,np.ones(len(x))));b=np.column_stack(((dx-mean)/scale,np.ones(len(dx))))
    sample_weight=np.asarray([len(y)/(2*np.sum(y==k)) for k in y]);penalty=np.eye(65);penalty[-1,-1]=0
    lhs=a.T@(a*sample_weight[:,None])+penalty;rhs=a.T@(sample_weight*(2*y-1))
    coef=np.linalg.solve(lhs,rhs)
    return a@coef,b@coef,coef,mean,scale


def information(f,y,spikes,saturation,n):
    raw=f.astype(np.float64);centered=raw-raw.mean(0);singular=np.linalg.svd(centered,compute_uv=False)
    tol=max(centered.shape)*np.finfo(np.float64).eps*(singular[0] if len(singular) else 0)
    groups={}
    for row,label in zip(f,y):groups.setdefault(row.tobytes(),set()).add(int(label))
    return {"centered_rank":int((singular>tol).sum()),"singular_values":singular.tolist(),
            "variance_sum":float(f.astype(np.float64).var(0).sum()),"observed_never_spiked_units":int((spikes==0).sum()),
            "observed_never_spiked_fraction":float((spikes==0).mean()),"saturated_q_fraction_all_ticks":saturation/(n*32*64),
            "unique_endpoint_vectors":len(groups),"opposite_class_exact_collision_groups":sum(len(v)>1 for v in groups.values()),
            "class_centroid_L2":float(np.linalg.norm(f[y==1].mean(0)-f[y==0].mean(0)))}


def conditional_intervals(summaries,output,phase):
    rng=np.random.default_rng(991116);draws={};point={}
    for seed in (261161,261167,261173):
        with np.load(phase/f"fit-{seed}.npz",allow_pickle=False) as z:
            y=z["dev_y"];strata=list(zip(z["dev_order"],z["dev_gap"],z["dev_clock"]))
        indices=np.empty((10000,len(y)),np.int64)
        for value in sorted(set(strata)):
            ix=np.asarray([i for i,v in enumerate(strata) if v==value]);indices[:,ix]=rng.choice(ix,size=(10000,len(ix)))
        w=np.where(y==1,len(y)/(2*np.sum(y==1)),len(y)/(2*np.sum(y==0)))
        for r in [r for r in summaries if r["seed"]==seed]:
            for kind in ("final","initial"):
                key=r["keys"][kind]
                with np.load(output/"features"/key/"probe-f.npz",allow_pickle=False) as z:score=z["dev_scores"]
                score=np.where(np.abs(score)<=1e-12,0,score);correct=(score>0)==y
                ident=(seed,r["sign"],r["arm"],kind)
                draws[ident]=(correct[indices]*w[indices]).mean(1);point[ident]=float((correct*w).mean())
            with np.load(ROOT/OLD/"learning/trials"/f"{seed}-{r['sign']}-{r['arm']}"/"predictions.npz",allow_pickle=False) as z:
                correct=z["dev_p_u192"].argmax(1)==y
            ident=(seed,r["sign"],r["arm"],"direct")
            draws[ident]=(correct[indices]*w[indices]).mean(1);point[ident]=float((correct*w).mean())
    result=[]
    for sign in (1,-1):
        for arm,kind in (("teacher","direct"),("label","final"),("shuffled","final"),("readout","final"),("teacher","initial")):
            differences=[draws[(seed,sign,"teacher","final")]-draws[(seed,sign,arm,kind)] for seed in (261161,261167,261173)]
            mean=np.mean(differences,axis=0)
            result.append({"sign":sign,"contrast":f"teacher_primary_probe_minus_{arm}_{kind}",
                           "balanced_accuracy_gain":float(np.mean([point[(s,sign,"teacher","final")]-point[(s,sign,arm,kind)] for s in (261161,261167,261173)])),
                           "CI95_input_conditional_fixed_models":np.quantile(mean,[.025,.975]).tolist(),
                           "seed_population_claim":False,"no_multiple_comparison_quality_gate":True})
    return result


def run(output):
    start=time.perf_counter();deadline=start+298;output.mkdir(exist_ok=False)
    cap=apply_memory_limit(700_000_000);monitor=Monitor();monitor.start();phase=ROOT/OLD/"learning"
    rows=read(phase/"completed.json");custody=read(ROOT/OLD/"closure/FINAL_MANIFEST.json")
    source_names=["scripts/research/rc_p1b_diagnostics.py","scripts/research/scope_p1b_common.py",
                  "scripts/research/scope_p1b_resources.py","src/scarx/neural/core.py","src/scarx/hybrid/cuda_alif.py",
                  "experiments/rc-p1b-diagnostics-v1.md","experiments/rc-p1b-diagnostics-provenance-recovery-v1.md"]
    hashes={name:sha(ROOT/name) for name in source_names}
    current=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    git_hashes={}
    for name,expected in hashes.items():
        blob=subprocess.check_output(["git","show",current+":"+name],cwd=ROOT)
        assert blob.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'),"Non-EOL source mutation: "+name
        git_hashes[name]=hashlib.sha256(blob).hexdigest()
        snapshot=output/"source"/name;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes((ROOT/name).read_bytes())
    if psutil.virtual_memory().available<4_000_000_000 or psutil.disk_usage(ROOT.anchor).free<3_000_000_000:raise MemoryError("Host reserve")
    jobs={};subjects=[];inputs={}
    def verify_input(path):
        rel=str(path.relative_to(ROOT/OLD)).replace("\\","/")
        actual=sha(path)
        assert actual==custody["files"][rel],"Historical input custody: "+rel
        inputs[rel]=actual
    verify_input(phase/"completed.json")
    for r in sorted(rows,key=lambda r:(r["seed"],r["sign"],r["arm"])):
        directory=ROOT/OLD/r["relative_directory"];data=phase/f"fit-{r['seed']}.npz"
        verify_input(directory/"predictions.npz");verify_input(data)
        keys={}
        for kind in ("initial","final"):
            checkpoint=directory/(kind+".npz");rel=str(checkpoint.relative_to(ROOT/OLD)).replace("\\","/")
            verify_input(checkpoint)
            key=hidden_key(checkpoint);keys[kind]=key
            jobs.setdefault(key,{"key":key,"seed":r["seed"],"checkpoint":str(checkpoint),"data":str(data),
                                 "directory":str(output/"features"/key),"deadline":deadline,"stop":str(output/"STOP")})
        subjects.append({**r,"feature_keys":keys})
    write(output/"registration.json",{"source_commit":current,"source_hashes":hashes,"committed_blob_hashes":git_hashes,"input_hashes":inputs,
          "source_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
          "cap":cap,"max_seconds":300,"prior_failed_intake_reserved_seconds":2,"unique_hidden_states":len(jobs),"probe_alpha":1,"scope":"TRAIN_DEV_only_NO_hidden_updates_NO_TEST"})
    write(output/"hardware.json",{"physical_cores":psutil.cpu_count(False),"logical_cores":psutil.cpu_count(),
          "RAM":psutil.virtual_memory()._asdict(),"pagefile_pressure":psutil.swap_memory()._asdict(),
          "disk":psutil.disk_usage(ROOT.anchor)._asdict(),"GPU":gpu(),
          "live_backend":"User explicitly stopped Hermes model before this study; do not restart"})
    write(output/"progress.json",{"status":"PROFILING","completed_jobs":0})
    representative=next(iter(jobs.values()));profiles={}
    for backend in ("cpu","gpu"):
        cmd=[sys.executable,__file__,"profile","--backend",backend,"--output",str(output/("profile-"+backend)),
             "--checkpoint",representative["checkpoint"],"--data",representative["data"]]
        with (output/(backend+".stdout.txt")).open("w") as so,(output/(backend+".stderr.txt")).open("w") as se:
            p=subprocess.run(cmd,stdout=so,stderr=se,timeout=max(1,deadline-time.perf_counter()))
        profiles[backend]={"exit_code":p.returncode}
        if p.returncode==0:profiles[backend].update(read(output/("profile-"+backend)/"result.json"))
    if profiles["cpu"]["exit_code"]:raise RuntimeError("CPU profile failed")
    if not profiles["gpu"]["exit_code"]:
        with np.load(output/"profile-cpu/outputs.npz") as a,np.load(output/"profile-gpu/outputs.npz") as b:
            maxerror=max(float(np.max(np.abs(a[k]-b[k]))) for k in ("f","q","p","spikes"))
        assert maxerror<1e-6,"Backend parity"
        if profiles["gpu"]["median_two_episodes_seconds"]<profiles["cpu"]["median_two_episodes_seconds"]:
            raise RuntimeError("CUDA faster: frozen CPU extractor not authorized as optimal; stop for implementation")
    else:maxerror=None
    profile_resource=monitor.close()
    if profile_resource["sampled_tree_peak_rss"]>800_000_000:raise MemoryError("Profile tree cap")
    monitor=Monitor();monitor.start()
    calibration_records=[]
    for workers in (1,2,3):
        if time.perf_counter()>deadline or monitor.peak_rss>700_000_000:raise RuntimeError("Calibration resource limit")
        t=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records=list(pool.map(calibration,[representative]*6))
        elapsed=time.perf_counter()-t
        if time.perf_counter()>deadline or monitor.peak_rss>700_000_000:raise RuntimeError("Calibration post-batch limit")
        calibration_records.append({"workers":workers,"seconds_including_startup":elapsed,"episodes_per_second":384/elapsed,"jobs":records})
    chosen=max(calibration_records,key=lambda r:r["episodes_per_second"])["workers"]
    write(output/"scheduler.json",{"backend":"cpu","profiles":profiles,"max_parity_error":maxerror,
          "calibration":calibration_records,"chosen_workers":chosen,"threads_per_worker":1,
          "calibration_forwards":1152,"no_utilization_target":True})
    (output/"features").mkdir();complete=[]
    if time.perf_counter()>deadline or monitor.peak_rss>700_000_000:raise RuntimeError("Before initial extraction submissions")
    pool=ProcessPoolExecutor(max_workers=chosen);pending=set();iterator=iter(jobs.values())
    try:
        for _ in range(chosen):
            payload=next(iterator,None)
            if payload is not None:pending.add(pool.submit(extract,payload))
        while pending:
            done,pending=wait(pending,timeout=max(.01,deadline-time.perf_counter()),return_when=FIRST_COMPLETED)
            if not done:raise TimeoutError("Extraction controller deadline")
            for future in done:
                complete.append(future.result())
                write(output/"progress.json",{"status":"EXTRACTING_FROZEN_FEATURES","completed_jobs":len(complete),"total_jobs":len(jobs)})
                if monitor.peak_rss>700_000_000 or time.perf_counter()>deadline:raise RuntimeError("D1D2 runtime/RSS limit")
                payload=next(iterator,None)
                if payload is not None:pending.add(pool.submit(extract,payload))
    except Exception:
        (output/"STOP").touch(exist_ok=False)
        for future in pending:future.cancel()
        raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
    write(output/"extraction-results.json",complete);probes={};info={};prediction_records=[];probe_count=0
    for key,payload in jobs.items():
        if time.perf_counter()>deadline or monitor.peak_rss>700_000_000:raise RuntimeError("Probe stage resource limit")
        directory=Path(payload["directory"])
        with np.load(payload["data"],allow_pickle=False) as z:y=z["train_y"];dy=z["dev_y"];ids=z["dev_ids"]
        with np.load(directory/"features.npz",allow_pickle=False) as z:
            info[key]={"TRAIN":information(z["train_f"],y,z["train_spikes"],int(z["saturation"][0]),len(y)),
                       "DEV":information(z["dev_f"],dy,z["dev_spikes"],int(z["saturation"][1]),len(dy))}
            probes[key]={}
            for variant in ("f","q"):
                train_scores,dev_scores,coef,mean,scale=ridge(z["train_"+variant],y,z["dev_"+variant]);probe_count+=1
                np.savez(directory/("probe-"+variant+".npz"),coefficient=coef,mean=mean,scale=scale,train_scores=train_scores,dev_scores=dev_scores)
                probes[key][variant]={"TRAIN":metrics(train_scores,y),"DEV":metrics(dev_scores,dy)}
                for identifier,label,score in zip(ids,dy,np.where(np.abs(dev_scores)<=1e-12,0,dev_scores)):
                    prediction_records.append({"key":key,"seed":payload["seed"],"variant":variant,"case_id":str(identifier),
                                               "label":int(label),"score":float(score),"prediction":int(score>0)})
    summaries=[]
    for r in subjects:
        key=r["feature_keys"]["final"];directory=ROOT/OLD/r["relative_directory"]
        with np.load(output/"features"/key/"features.npz",allow_pickle=False) as z:q=z["dev_q"]
        with np.load(directory/"final.npz",allow_pickle=False) as z:logits=q@z["core_w_out"].T+z["core_bias"]
        p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        with np.load(directory/"predictions.npz",allow_pickle=False) as z:
            error=float(np.max(np.abs(p-z["dev_p_u192"])));dy=z["dev_y"]
        assert error<1e-6,"Frozen extraction differs from saved direct predictions"
        summaries.append({"seed":r["seed"],"sign":r["sign"],"arm":r["arm"],"keys":r["feature_keys"],
            "direct":metrics(p[:,1]-p[:,0],dy),"primary_probe":probes[key]["f"],"secondary_probe_q":probes[key]["q"],
            "initial_hidden_probe":probes[r["feature_keys"]["initial"]]["f"],"direct_reconstruction_max_abs_error":error})
    adverse=[r for r in summaries if r["arm"]=="teacher" and r["sign"]==-1]
    diagnostic=all(r["primary_probe"]["DEV"]["balanced_accuracy"]>=.75 and
                   min(r["primary_probe"]["DEV"][k] for k in ("negative_recall","positive_recall"))>=.65 and
                   r["primary_probe"]["DEV"]["balanced_accuracy"]-r["direct"]["balanced_accuracy"]>=.10 for r in adverse)
    with (output/"probe-predictions.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(prediction_records[0]));writer.writeheader();writer.writerows(prediction_records)
    write(output/"D1-results.json",summaries);write(output/"D2-information.json",info)
    write(output/"probe-metrics-by-hidden.json",probes)
    write(output/"conditional-intervals.json",conditional_intervals(summaries,output,phase))
    elapsed=time.perf_counter()-start
    resource=monitor.close();write(output/"resource.json",{"tree":resource,"profile_tree":profile_resource,"parent":memory(),"wall_seconds":elapsed})
    if elapsed>300 or resource["sampled_tree_peak_rss"]>700_000_000:raise RuntimeError("Resource envelope")
    result={"status":"DONE","adverse_teacher_meets_preregistered_probe_diagnostic":diagnostic,
            "G2":"FAIL_UNCHANGED","hidden_training_updates":0,"diagnostic_head_fits":probe_count,
            "new_feature_forwards":sum(r["forwards"] for r in complete),"profile_forwards":sum(p.get("episode_forwards",0) for p in profiles.values()),
            "calibration_forwards":1152,"unique_hidden_states":len(jobs),"model_seeds":3,
            "TEST_opened":False,"scheduler_workers":chosen,"wall_seconds":elapsed,"source_commit":current,
            "probe_accuracy_is_not_direct_inference":True,"seed_population_CI":"INCONCLUSIVE_N3"}
    write(output/"result.json",result);write(output/"progress.json",result);print(json.dumps(result))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("mode",choices=("run","profile"));ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--backend",choices=("cpu","gpu"));ap.add_argument("--checkpoint",type=Path);ap.add_argument("--data",type=Path)
    args=ap.parse_args()
    if args.mode=="profile":print(json.dumps(profile(args.backend,args.output,args.checkpoint,args.data)))
    else:
        try:run(args.output)
        except Exception as e:
            if args.output.exists():write(args.output/"FAILED.json",{"error":repr(e),"keep_all_results":True,"no_retry_or_replacement_seeds":True})
            raise


if __name__=="__main__":main()
