"""Data/proposal auditor: no ALIF model invocation or optimizer fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def main():
    start=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    run=args.run;meta=json.loads((run/"manifest.json").read_text());rows=json.loads((run/"records.json").read_text())
    with np.load(run/"raw.npz",allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    fail=[]
    def check(ok,why):
        if not ok:fail.append(why)
    teacher=[];witness=[];variances=[];old_ok=[];P_ablated=[]
    for seed in meta["seeds"]:
        X=raw[f"X_{seed}"];Y=raw[f"Y_{seed}"];H=raw[f"H_{seed}"];Z=raw[f"Z_{seed}"]
        p=raw[f"teacher_p_{seed}"];wp=raw[f"witness_p_{seed}"];partner=raw[f"partner_{seed}"]
        P_ablated.extend(Y==0)
        for kind,h in (("teacher",128),("witness",64)):
            with np.load(run/f"constructed-{kind}-{seed}.npz",allow_pickle=False) as checkpoint:
                check(checkpoint["core_w_in"].shape==(h,64),"teacher/student exact capacity")
                check(all(np.max(np.abs(checkpoint[k]))<=3 for k in ("core_w_in","core_w_rec","core_w_out","core_bias")),"parameter bounds")
                check(np.all(checkpoint["core_indices"][:64]<64),"no P-to-S sparse edge")
                check(checkpoint["core_w_in"].dtype==np.float32,"native FP32")
            old_ok.extend(raw[f"old_p_{seed}_{kind}"].argmax(axis=1)==raw["old_Y"])
        check(np.array_equal(raw[f"prefix_original_{seed}"],raw[f"prefix_changed_{seed}"]),"causal prefix invariance")
        clocks=[]
        for i,seq in enumerate(X):
            positions=np.flatnonzero(seq[:,0]);first,last=map(int,positions)
            order=int(np.sign(seq[first,1]));label=int(order==1 and last-first<=2)
            check(label==int(Y[i]),"independent input-label semantics")
            check(seq[last,1]*seq[first,1]<0,"two event kinds")
            clocks.append(last)
            check(np.all(seq[:,2:]==0),"no codec target channels")
        teacher.extend(p.argmax(axis=1)==Y);witness.extend(wp.argmax(axis=1)==Y)
        check(np.allclose(H[:,0],.08*Z[:,0]/.1,atol=1e-6),"feature filter at first tick")
        for t in range(1,32):check(np.allclose(H[:,t],.92*H[:,t-1]+.8*Z[:,t],atol=1e-6),"spike filter recurrence")
        for key in sorted(set(zip(Y.tolist(),clocks))):
            idx=[i for i in range(len(Y)) if (Y[i],clocks[i])==key]
            variances.append(float(H[idx].var(axis=0).max()))
            perm=[int(partner[i]) for i in idx]
            check(sorted(idx)==sorted(perm) and all(i!=j for i,j in zip(idx,perm)),"within-class-clock bijection")
        # Head law is computed from recorded teacher feature, not copied passed flags.
        f=H[:,-1,:28].mean(axis=1);logit=3*f-.32
        expected=1/(1+np.exp(-logit))
        check(np.max(np.abs(expected-p[:,1]))<2e-6,"teacher head from actual P features")
    actual=raw["physical"];D=raw["directions"];W=raw["initial_W"]
    moment=.1*D;energy=.99*np.float32(1e-4)+.01*D**2
    step=np.clip(.008*moment/(np.sqrt(energy)+1e-4),-.01,.01)
    expected=np.clip(W[:,None]+step,-3,3)-W[:,None];expected[:,:,0:8]=0
    maxdelta=float(np.max(np.abs(expected-actual)))
    check(maxdelta<2e-7,"independent final physical-update reconstruction")
    diff=np.linalg.norm(actual[:,0]-actual[:,1],axis=(1,2))
    denom=np.maximum(np.linalg.norm(actual[:,0],axis=(1,2)),1e-12);ratios=diff/denom
    groups=[]
    for seed in meta["seeds"]:
        for sign in (1,-1):
            idx=[i for i,r in enumerate(rows) if r["seed"]==seed and r["sign"]==sign]
            groups.append({"seed":seed,"sign":sign,"median_relative_physical_difference":float(np.median(ratios[idx]))})
    check(all(not r["hidden_weights_changed"] and r["learning_updates"]==0 for r in rows),"no applied learning")
    gates={"teacher_qualified":float(np.mean(teacher))>=.95,"student_witness":float(np.mean(witness))>=.95,
           "teacher_P_necessary":float(np.mean(teacher))-float(np.mean(P_ablated))>=.05,
           "old_valid_construction":all(old_ok),
           "conditional_trace_variation":min(variances)>1e-6,
           "physical_update_distinguishable":all(g["median_relative_physical_difference"]>.01 for g in groups) and max(diff)>1e-6}
    result={"audit_pass":not fail,"failures":fail,"independently_reconstructed_gates":gates,
            "teacher_accuracy":float(np.mean(teacher)),"witness_accuracy":float(np.mean(witness)),
            "min_conditional_trace_variance":min(variances),"groups":groups,"max_delta_error":maxdelta,
            "model_calls":0,"learning_updates":0,"same_author_independent_algebra_not_empirical_replication":True,
            "wall_seconds":time.perf_counter()-start}
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result));return 0 if not fail else 2


if __name__=="__main__":raise SystemExit(main())
