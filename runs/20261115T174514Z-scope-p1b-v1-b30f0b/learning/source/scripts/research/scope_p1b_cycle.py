"""Registered conditional second-cycle continuation of an existing B trajectory."""
from pathlib import Path
import time
import numpy as np
from scope_p1b_common import ALIF,cases,old_cases,sha
from scope_p1b_trials import fit,evaluate,score,write_json


def cycle(payload):
    start=time.perf_counter();base=Path(payload["directory"]);directory=base/"cycle2"
    directory.mkdir(exist_ok=False);deadline=payload["deadline"]
    student=ALIF.loads((base/"final.npz").read_bytes());seed=student.seed
    plastic=ALIF(seed+1000000,student.sign)
    plastic.kind="plastic";plastic.cycle=2
    plastic.core.w_in[8:,3:5]=plastic.core.w_in[8:,0:2]
    plastic.core.w_in[8:,0:2]=0;plastic.core.stable_in[:]=plastic.core.w_in
    (directory/"P-reset-initial.npz").write_bytes(plastic.dumps())
    tr=cases(seed,"train",cycle=2);dv=cases(seed,"dev",cycle=2);old=old_cases()
    X=np.stack([c["sequence"] for c in tr]);y=np.asarray([c["label"] for c in tr])
    DX=np.stack([c["sequence"] for c in dv]);dy=np.asarray([c["label"] for c in dv])
    OX=np.stack([c["sequence"] for c in old]);oy=np.asarray([c["label"] for c in old])
    pdir=directory/"plastic-learning";pdir.mkdir()
    pred,curves,_=fit(plastic,X,y,np.zeros((len(y),32,56),np.float32),"label",deadline,pdir,DX,dy,OX,oy)
    np.savez(pdir/"predictions.npz",**pred,dev_y=dy,old_y=oy)
    np.savez(directory/"cycle2-inputs.npz",train_X=X,train_y=y,dev_X=DX,dev_y=dy)
    pblob=plastic.dumps();(directory/"P-learned.npz").write_bytes(pblob)
    qualified=curves[-1]["dev"]["accuracy"]>=.95 and curves[-1]["dev"]["min_recall"]>=.65
    if not qualified:
        result={"verdict":"PLASTIC_REUSE_LEARNER_NO_GO","P_updates":plastic.updates,"S_updates_extra":0,
                "P_dev":curves[-1]["dev"],"forward_calls":plastic.forwards,"wall_seconds":time.perf_counter()-start}
        write_json(directory/"result.json",result);return result
    oldtrain=cases(seed,"train");olddev=cases(seed,"dev")
    mixed=oldtrain+tr;XX=np.stack([c["sequence"] for c in mixed]);yy=np.asarray([c["label"] for c in mixed])
    hints=[]
    teacher_forwards_before=plastic.forwards+student.forwards;S_forwards_at_load=student.forwards
    for i,c in enumerate(mixed):
        m=student if i<len(oldtrain) else plastic
        hints.append(np.asarray(m.forward(c["sequence"],tape=True)[3],np.float32)[:,8:64])
    cache_forwards=plastic.forwards+student.forwards-teacher_forwards_before
    order=np.random.default_rng(np.random.SeedSequence([seed,50000])).permutation(len(mixed))
    dev_union=olddev+dv;UX=np.stack([c["sequence"] for c in dev_union]);uy=np.asarray([c["label"] for c in dev_union])
    sdir=directory/"student-consolidation";sdir.mkdir();u_before=student.updates;f_before=student.forwards
    pred,curves,_=fit(student,XX[order],yy[order],np.asarray(hints)[order],"teacher",deadline,sdir,UX,uy,OX,oy)
    np.savez(sdir/"predictions.npz",**pred,dev_y=uy,old_y=oy)
    np.savez(directory/"consolidation-inputs.npz",train_X=XX[order],train_y=yy[order],train_H=np.asarray(hints)[order],dev_X=UX,dev_y=uy)
    pp=pred["dev_p_u384"];old1=score(pp[:len(olddev)],uy[:len(olddev)]);new2=score(pp[len(olddev):],uy[len(olddev):])
    oldbaseline=payload["skill1_dev_accuracy"]
    accepted=old1["accuracy"]>=oldbaseline-.03 and old1["min_recall"]>=.65 and new2["accuracy"]>=.65 and new2["min_recall"]>=.65
    blob=student.dumps();(directory/"S2-candidate.npz").write_bytes(blob)
    # Do not reset a learned P2 or delete its only skill on failed consolidation.
    result={"verdict":"CYCLE2_DEV_QUALIFIED_NEEDS_RESTART" if accepted else "CYCLE2_CONSOLIDATION_NO_GO",
            "P_updates":plastic.updates,"S_updates_extra":student.updates-u_before,"skill1_dev":old1,"skill2_dev":new2,
            "cache_forwards":cache_forwards,"student_forward_extra":student.forwards-f_before,
            "P_total_forwards":plastic.forwards,"S_total_forwards_extra":student.forwards-S_forwards_at_load,
            "P_checkpoint_sha256":sha(pblob),"S2_checkpoint_sha256":sha(blob),"P2_reset_after_learning":False,
            "wall_seconds":time.perf_counter()-start,"capacity":{"S":64,"P":64,"public_output_contracts":2}}
    write_json(directory/"result.json",result);return result
