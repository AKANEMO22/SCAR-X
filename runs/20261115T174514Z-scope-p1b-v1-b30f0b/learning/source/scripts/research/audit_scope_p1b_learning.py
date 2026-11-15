"""Independent raw metrics and literal local-rule audit, never trains/refits."""
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np


def local_reference(X,weights,H):
    v=np.zeros(64);a=v.copy();f=v.copy();ev=np.zeros((64,4));ea=ev.copy();ef=ev.copy();bank=ev.copy();minmargin=1.
    for t,row in enumerate(X.astype(float)):
        theta=.65+.08*a;u=.85*v+weights.astype(float)@row[[0,1,3,4]]
        minmargin=min(minmargin,float(np.min(np.abs(u-theta))))
        s=1/(1+np.exp(-np.clip((u-theta)/.4,-20,20)));d=s*(1-s)/.4*(np.abs((u-theta)/.4)<20)
        candidate_ev=.85*ev+row[[0,1,3,4]][None,:];z_e=d[:,None]*(candidate_ev-.08*ea)
        spikes=(u>=theta).astype(float);v=u-spikes*theta;a=.95*a+spikes;f=.92*f+.08*spikes
        ev=candidate_ev-theta[:,None]*z_e-spikes[:,None]*.08*ea;ea=.95*ea+z_e;ef=.92*ef+.08*z_e
        raw=f/.1;signal=np.zeros(64);signal[8:] = (H[t]-np.clip(raw[8:],-5,5))/.1*(np.abs(raw[8:])<5)
        bank+=signal[:,None]*ef/32
    return np.clip(f/.1,-5,5),ef,bank,minmargin


def quantile(a,q):
    a=sorted(map(float,a));pos=(len(a)-1)*q;i=int(pos);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(pos-i)


def main():
    start=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=args.run;phase=root/"learning";records=json.loads((phase/"completed.json").read_text());fail=[]
    def need(ok,why):
        if not ok:fail.append(why)
    need(len(records)==24,"24 registered primary cells")
    reg=json.loads((phase/"registration.json").read_text())
    for name,h in reg["source_hashes"].items():need(hashlib.sha256((phase/"source"/name).read_bytes()).hexdigest()==h,"source hash "+name)
    cache={};correct={};meta={};initials={};all_model_updates=0;all_hidden=0;max_algebra=0.;max_reference=0.;reference_replays=0
    for seed in reg["seeds"]:
        with np.load(phase/f"fit-{seed}.npz",allow_pickle=False) as z:cache[seed]={k:z[k] for k in z.files}
        data=cache[seed];X=data["train_X"];Y=data["train_y"];H=data["train_H"];partner=data["partner"]
        for i,x in enumerate(X):
            events=np.flatnonzero(x[:,0]);first,last=map(int,events);order=np.sign(x[first,1])
            need(int(order==1 and last-first<=2)==int(Y[i]),"labels from input history")
            need(np.all(x[:,2:]==0),"no future target input")
            need(Y[i]==Y[partner[i]] and last==int(np.flatnonzero(X[partner[i],:,0])[-1]),"conditional shuffle contract")
            # Three actual teacher neuron types; verify trace from input without teacher model code.
            types=[(.358,-.158,28),(.45,.45,14),(.45,-.45,14)];offset=0
            for we,wk,n in types:
                v=a=f=0.;expected=[]
                for row in x:
                    u=.85*v+we*float(row[0])+wk*float(row[1]);theta=.65+.08*a;z=float(u>=theta)
                    v=u-z*theta;a=.95*a+z;f=.92*f+.08*z;expected.append(f/.1)
                need(np.max(np.abs(H[i,:,offset:offset+n]-np.asarray(expected)[:,None]))<1e-5,"teacher causal trace cache")
                offset+=n
        meta[seed]=list(zip(data["dev_order"].tolist(),data["dev_gap"].tolist(),data["dev_clock"].tolist()))
    for r in records:
        directory=root/r["relative_directory"];data=cache[r["seed"]]
        initial=(directory/"initial.npz").read_bytes();final=(directory/"final.npz").read_bytes()
        need(hashlib.sha256(initial).hexdigest()==r["initial_checkpoint_sha256"],"initial checkpoint hash")
        need(hashlib.sha256(final).hexdigest()==r["final_checkpoint_sha256"],"final checkpoint hash")
        with np.load(directory/"initial.npz",allow_pickle=False) as z:ini={k:z[k] for k in z.files}
        with np.load(directory/"final.npz",allow_pickle=False) as z:fin={k:z[k] for k in z.files}
        key=(r["seed"],r["sign"])
        if key in initials:need(all(np.array_equal(v,initials[key][k]) for k,v in ini.items()),"paired initialization arrays")
        else:initials[key]=ini
        need(np.array_equal(ini["core_w_rec"],fin["core_w_rec"]),"fixed recurrent weights")
        need(np.array_equal(ini["core_w_in"][:8],fin["core_w_in"][:8]),"old hidden rows")
        need(np.array_equal(ini["old_W"],fin["old_W"]) and np.array_equal(ini["old_bias"],fin["old_bias"]),"old readout")
        frozen=np.ones((64,64),bool);frozen[:,[0,1,3,4]]=False;frozen[:8]=True
        need(np.array_equal(ini["core_w_in"][frozen],fin["core_w_in"][frozen]),"outside trainability mask")
        need(fin["core_w_in"].shape==(64,64),"S-only checkpoint shape")
        with np.load(directory/"predictions.npz",allow_pickle=False) as z:
            p=z["dev_p_u192"];y=z["dev_y"];rp=z["revert_hidden_p"]
            need(np.array_equal(y,data["dev_y"]),"shared labels")
            accuracy=float(np.mean(p.argmax(axis=1)==y));reverted=float(np.mean(rp.argmax(axis=1)==y))
            need(abs(accuracy-r["final_dev"]["accuracy"])<1e-12,"raw accuracy")
            need(abs(reverted-r["hidden_revert"]["accuracy"])<1e-12,"revert score")
            correct[(r["seed"],r["sign"],r["arm"])]=(p.argmax(axis=1)==y).astype(float)
        log=list(csv.DictReader((directory/"learning.csv").open(encoding="utf-8")))
        need(len(log)==192,"192updates logged")
        with np.load(directory/"learning-diagnostics.npz",allow_pickle=False) as z:d={k:z[k] for k in z.files}
        moment=np.zeros((64,4));energy=np.full((64,4),1e-4)
        for i in range(192):
            prob=d["p_before"][i];label=int(data["train_y"][i]);delta=-prob.astype(float);delta[label]+=1
            W=d["head_before"][i].astype(float);q=d["q_endpoint"][i].astype(float)
            logits=W@q+d["bias_before"][i];pp=np.exp(logits-logits.max());pp/=pp.sum()
            need(np.max(np.abs(pp-prob))<1e-5,"probability pre-update")
            need(abs(-math.log(max(float(prob[label]),1e-12))-float(log[i]["loss"]))<1e-10,"loss")
            dw=np.clip(.04*delta[:,None]*q[None,:],-.025,.025);dw[:,:8]=0
            nextW=d["head_before"][i+1] if i<191 else fin["core_w_out"]
            nextb=d["bias_before"][i+1] if i<191 else fin["core_bias"]
            need(np.max(np.abs(np.clip(W+dw,-3,3)-nextW))<1e-6,"output-local physical update")
            need(np.max(np.abs(np.clip(d["bias_before"][i]+np.clip(.04*delta,-.025,.025),-3,3)-nextb))<1e-6,"bias physical update")
            direction=(W.T@delta)[:,None]*(np.abs(q)<5)[:,None]/.1*d["eligibility"][i]
            if r["arm"] in ("teacher","shuffled"):direction+=d["hint_banks"][i]
            direction[:8]=0
            max_algebra=max(max_algebra,float(np.max(np.abs(direction-d["directions"][i]))))
            if r["arm"]!="readout":
                moment=.9*moment+.1*direction;energy=.99*energy+.01*direction**2
                step=np.clip(.008*moment/(np.sqrt(energy)+1e-4),-.01,.01)
                expected=np.clip(d["weights_before"][i]+step,-3,3)-d["weights_before"][i];expected[:8]=0
            else:expected=np.zeros((64,4))
            max_algebra=max(max_algebra,float(np.max(np.abs(expected-d["actual_hidden_deltas"][i]))))
            need(abs(float(np.linalg.norm(d["actual_hidden_deltas"][i]))-float(log[i]["actual_hidden_delta_L2"]))<1e-6,"actual weightdelta")
            if i in (0,47,95,191):
                hi=int(data["partner"][i]) if r["arm"]=="shuffled" else i
                qr,ef,bank,margin=local_reference(data["train_X"][i],d["weights_before"][i],data["train_H"][hi])
                error=max(float(np.max(np.abs(qr-q))),float(np.max(np.abs(ef-d["eligibility"][i]))),float(np.max(np.abs(bank-d["hint_banks"][i]))))
                max_reference=max(max_reference,error);reference_replays+=1
                need(error<1e-4,"literal ALIF/local trace audit; near-threshold mismatch is not waived")
        all_model_updates+=r["updates"];all_hidden+=r["hidden_updates"]
        if r["arm"]=="readout":need(np.array_equal(ini["core_w_in"],fin["core_w_in"]),"RO invariant")
    need(max_algebra<1e-4,"direction/update algebra tolerance")
    # Independent paired input-conditional bootstrap, signs share each case draw.
    rng=np.random.default_rng(2026111609);boot={a:np.zeros(10000) for a in ("label","shuffled","readout")};effects={a:[] for a in boot}
    for seed in reg["seeds"]:
        blocks=[]
        for category in sorted(set(meta[seed])):
            indices=np.asarray([i for i,k in enumerate(meta[seed]) if k==category]);n=len(indices)
            blocks.append(indices[rng.integers(0,n,size=(10000,n))])
        selected=np.concatenate(blocks,axis=1)
        for a in boot:
            dif=np.stack([correct[(seed,s,"teacher")]-correct[(seed,s,a)] for s in (1,-1)])
            effects[a].extend(dif.mean(axis=1).tolist());boot[a]+=dif.mean(axis=0)[selected].mean(axis=1)/3
    reported=json.loads((phase/"statistics.json").read_text());stats={}
    for arm in boot:
        ci=[quantile(boot[arm],.025),quantile(boot[arm],.975)]
        adj=[quantile(boot[arm],.05/6),quantile(boot[arm],1-.05/6)]
        item={"gain":float(np.mean(effects[arm])),"CI95":ci,"CI98_333":adj,"per_seed_sign_gain":effects[arm]};stats[arm]=item
        old=reported["teacher-minus-"+arm]
        need(abs(item["gain"]-old["gain"])<1e-12 and np.max(np.abs(np.asarray(ci)-old["CI95_input_conditional"]))<1e-12,"independent paired statistics")
        need(np.max(np.abs(np.asarray(adj)-old["CI98_333_input_conditional_Bonferroni3"]))<1e-12,"adjusted interval")
    result={"audit_pass":not fail,"failures":fail,"raw_metrics_and_bootstrap":stats,
        "checked_primary_updates":all_model_updates,"checked_hidden_updates":all_hidden,
        "max_update_algebra_error":max_algebra,"literal_student_reference_rollouts":reference_replays,
        "teacher_reference_episode_rollouts":sum(len(x["train_y"]) for x in cache.values()),"max_reference_error":max_reference,
        "learning_updates_created_by_auditor":0,"model_library_imports":False,
        "wall_seconds":time.perf_counter()-start,"same_author_independent_implementation":True,
        "population_seed_CI":"INCONCLUSIVE_N3"}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result));return 0 if not fail else 2


if __name__=="__main__":raise SystemExit(main())
