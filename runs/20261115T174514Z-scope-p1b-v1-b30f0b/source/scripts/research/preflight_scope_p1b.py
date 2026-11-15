"""Frozen P1b prerequisite checks; computes proposals but never calls learn."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from scope_p1b_common import ALIF, cases, old_cases, SEEDS, ROOT, sha, cuda, np
from scope_p1b_resources import memory
sys.path.insert(0,str(ROOT/"src"))
from scarx.runtime.limits import apply_memory_limit


def donors(data):
    buckets=defaultdict(list)
    for i,c in enumerate(data):buckets[(c["label"],c["clock"])].append(i)
    result=np.arange(len(data))
    for indices in buckets.values():
        ordered=sorted(indices,key=lambda i:(data[i]["order"],data[i]["gap"],data[i]["id"]))
        for i,j in zip(ordered,np.roll(ordered,len(ordered)//2)):result[i]=j
    return result


def main():
    start=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--backend",choices=("cpu","gpu"),default="cpu")
    args=ap.parse_args();run=args.run.resolve()
    if run.exists():raise FileExistsError(run)
    run.mkdir(parents=True);cap=apply_memory_limit(2_000_000_000)
    xp=np;handles=[]
    if args.backend=="gpu":xp,handles=cuda()
    sources=["experiments/scope-p1b-temporal-preflight-v1.md","scripts/research/scope_p1b_common.py",
             "scripts/research/scope_p1b_resources.py","scripts/research/preflight_scope_p1b.py",
             "scripts/research/audit_scope_p1b_preflight.py","src/scarx/neural/core.py",
             "src/scarx/evaluation/readout_eligibility.py"]
    manifest={"source_hashes":{p:sha((ROOT/p).read_bytes()) for p in sources},"seeds":list(SEEDS),
              "learning_updates":0,"backend":args.backend+"_no_training_preflight","memory_cap":cap}
    (run/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    raw={};records=[];physical=[];proposal_D=[];initial_W=[];ratios=[];all_old=[]
    teacher_all=[];witness_all=[];ro_all=[];variation=[];forward_count=0;prefix_ok=True;P_ablated=[]
    for seed in SEEDS:
        data=cases(seed,"qual");partner=donors(data);teacher=ALIF(seed,kind="teacher",xp=xp)
        witness=ALIF(seed,kind="witness",xp=xp)
        (run/f"constructed-teacher-{seed}.npz").write_bytes(teacher.dumps())
        (run/f"constructed-witness-{seed}.npz").write_bytes(witness.dumps())
        teacher_hints=[];teacher_p=[];witness_p=[];teacher_z=[]
        X=np.stack([c["sequence"] for c in data]);Y=np.asarray([c["label"] for c in data])
        P_ablated.extend(Y==0)
        for case in data:
            p,_,_,history,z=teacher.forward(case["sequence"],tape=True)
            wp,*_=witness.forward(case["sequence"])
            teacher_hints.append(np.asarray(history,np.float32)[:,64:120]);teacher_z.append(np.asarray(z,np.float32)[:,64:120])
            teacher_p.append(p);witness_p.append(wp)
            teacher_all.append(int(p.argmax())==case["label"]);witness_all.append(int(wp.argmax())==case["label"])
        H=np.asarray(teacher_hints);teacher_p=np.asarray(teacher_p);witness_p=np.asarray(witness_p)
        old_data=old_cases();raw["old_X"]=np.stack([c["sequence"] for c in old_data]);raw["old_Y"]=np.asarray([c["label"] for c in old_data])
        for kind,model in (("teacher",teacher),("witness",witness)):
            old_p=[]
            for c in old_data:
                pp=model.forward(c["sequence"],old=True)[0];old_p.append(pp)
                all_old.append(int(pp.argmax())==c["label"])
            raw[f"old_p_{seed}_{kind}"]=np.asarray(old_p)
        prefix=data[0]["sequence"].copy();positions=np.flatnonzero(prefix[:,0]);cut=int(positions[0])+1
        changed=prefix.copy();changed[cut:,1]*=-1
        original_history=np.asarray(teacher.forward(prefix,tape=True)[3])
        changed_history=np.asarray(teacher.forward(changed,tape=True)[3])
        prefix_ok &= np.array_equal(original_history[:cut],changed_history[:cut])
        raw[f"prefix_cut_{seed}"]=np.asarray(cut)
        raw[f"prefix_original_{seed}"]=original_history[:cut]
        raw[f"prefix_changed_{seed}"]=changed_history[:cut]
        # Conditional variation and distribution-preserving permutation, no labels enter B's forward.
        for key in sorted(set((c["label"],c["clock"]) for c in data)):
            idx=[i for i,c in enumerate(data) if (c["label"],c["clock"])==key]
            variation.append(float(np.var(H[idx],axis=0).max()))
        if not all(partner[i]!=i and Y[i]==Y[partner[i]] and data[i]["clock"]==data[partner[i]]["clock"] for i in range(len(data))):
            raise AssertionError("Matched partner contract")
        raw[f"X_{seed}"]=X;raw[f"Y_{seed}"]=Y;raw[f"H_{seed}"]=H
        raw[f"Z_{seed}"]=np.asarray(teacher_z);raw[f"teacher_p_{seed}"]=teacher_p;raw[f"witness_p_{seed}"]=witness_p
        raw[f"partner_{seed}"]=partner
        for sign in (1,-1):
            student=ALIF(seed,sign=sign,xp=xp);before=student.host(student.core.w_in).copy();init=student.dumps()
            (run/f"initial-{seed}-{sign}.npz").write_bytes(init)
            local_ratios=[]
            for i,case in enumerate(data):
                p,bank,_,_,_=student.forward(case["sequence"],track=True,hint=H[i])
                DB,_,_,dB,_=student.proposal(case["label"],p,bank,"teacher")
                p2,bank2,_,_,_=student.forward(case["sequence"],track=True,hint=H[partner[i]])
                DC,_,_,dC,_=student.proposal(case["label"],p2,bank2,"shuffled")
                DB,DC,dB,dC=map(student.host,(DB,DC,dB,dC))
                ro_all.append(np.count_nonzero(student.host(student.core.features))==0)
                diff=float(np.linalg.norm(dB-dC));relative=diff/max(float(np.linalg.norm(dB)),1e-12)
                local_ratios.append(relative);physical.append(np.stack([dB,dC]));proposal_D.append(np.stack([DB,DC]))
                initial_W.append(before[:,[0,1,3,4]])
                records.append({"seed":seed,"sign":sign,"case_index":i,"label":case["label"],
                                "gap":case["gap"],"order":case["order"],"clock":case["clock"],
                                "physical_delta_difference":diff,"physical_relative_difference":relative,
                                "hidden_weights_changed":not np.array_equal(before,student.host(student.core.w_in)),"learning_updates":student.updates})
            ratios.append({"seed":seed,"sign":sign,"median":float(np.median(local_ratios))})
            if not np.array_equal(before,student.host(student.core.w_in)) or student.updates:raise AssertionError("No preflight learning")
            forward_count+=student.forwards
        forward_count+=teacher.forwards+witness.forwards
    raw["physical"]=np.asarray(physical);raw["directions"]=np.asarray(proposal_D);raw["initial_W"]=np.asarray(initial_W)
    np.savez(run/"raw.npz",**raw)
    (run/"records.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    gates={"teacher_qualified":float(np.mean(teacher_all))>=.95,"student_witness":float(np.mean(witness_all))>=.95,
           "teacher_P_necessary":float(np.mean(teacher_all))-float(np.mean(P_ablated))>=.05,
           "old_valid_construction":all(all_old),"RO_feature_collision":all(ro_all),
           "conditional_trace_variation":min(variation)>1e-6,
           "physical_update_distinguishable":all(r["median"]>.01 for r in ratios) and max(r["physical_delta_difference"] for r in records)>1e-6,
           "causal_teacher_prefix":prefix_ok}
    result={"verdict":"PREFLIGHT_PASS_NOT_LEARNING_VALIDATION" if all(gates.values()) else "PREFLIGHT_NO_GO",
            "gates":gates,"teacher_accuracy":float(np.mean(teacher_all)),"witness_accuracy":float(np.mean(witness_all)),
            "old_valid_accuracy":float(np.mean(all_old)),"min_conditional_trace_variance":min(variation),
            "physical_relative_difference":ratios,"neural_forwards":forward_count,"learning_updates":0,
            "wall_seconds":time.perf_counter()-start,"memory":memory(),"teacher_hand_engineered":True,
            "TEST_accessed":False,"source_hashes":manifest["source_hashes"]}
    (run/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result));return 0


if __name__=="__main__":raise SystemExit(main())
