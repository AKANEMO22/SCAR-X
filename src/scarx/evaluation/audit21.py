from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from ..neural.core import Config,SpikingCore
from .tasks import episodes,accuracy


def audit_phase2(root):
    base=Path(root)/"runs/20261108T152637Z-phase2-v1-7a0790"
    data=json.loads((base/"result.json").read_text())
    rows=[]
    for trial in data["trials"]:
        matrix=trial["continual"]["matrix"]
        diagonal=np.array([row[i] for i,row in enumerate(matrix)])
        final=np.array(matrix[-1]); best=np.array([max(matrix[t][i] for t in range(i,20)) for i in range(20)])
        recomputed=float(np.mean(best-final))
        if abs(recomputed-trial["continual"]["forgetting_mean"])>1e-10: raise AssertionError("Phase2 metric mismatch")
        checkpoint=SpikingCore.load(base/f"{trial['seed']}-{trial['mode']}.npz")
        check=accuracy(checkpoint,list(episodes(2000+trial["seed"],128)))
        if check.tolist()!=trial["correct"]: raise AssertionError("Phase2 checkpoint mismatch")
        rows.append({"seed":trial["seed"],"mode":trial["mode"],"diagonal_acquisition":float(diagonal.mean()),
                     "final_accuracy":float(final.mean()),"best_to_final_forgetting":recomputed,
                     "signed_after_learning_drop":float(np.mean(diagonal-final)),"max_selection_excess":float(np.mean(best-diagonal))})
    interventions=[]
    for seed in (17,29,43):
        for mode in ("readout","full"):
            m=SpikingCore(Config(inputs=24,hidden=48,fan_in=8,plastic_rows=12,seed=seed))
            rng=np.random.default_rng(3000+seed); replay=[]; seen=0; first=None
            task0=list(episodes(6000+seed*100,128,contexts=20,context=0))
            for stage in range(20):
                fresh=list(episodes(4000+seed*100+stage,64,contexts=20,context=stage))
                old=[replay[i] for i in rng.choice(len(replay),min(len(replay),32),replace=False)] if replay else []
                mixed=fresh+old; rng.shuffle(mixed)
                for x,y in mixed: m.forward(x,track=True); m.learn(y,mode=mode)
                m.consolidate()
                for x,y in fresh:
                    seen+=1
                    if len(replay)<128: replay.append((x,y))
                    else:
                        j=int(rng.integers(seen))
                        if j<128: replay[j]=(x,y)
                if stage==0:
                    first={name:getattr(m,name).copy() for name in ("w_in","w_rec","w_out","bias")}
                    first_score=float(accuracy(m,task0).mean())
            final={name:getattr(m,name).copy() for name in first}
            normal=float(accuracy(m,task0).mean())
            for name in ("w_out","bias"): getattr(m,name)[:]=first[name]
            old_head=float(accuracy(m,task0).mean())
            for name in first: getattr(m,name)[:]=final[name]
            for name in ("w_in","w_rec"): getattr(m,name)[:]=first[name]
            old_hidden=float(accuracy(m,task0).mean())
            groups=[(int(np.argmax(x[0,4:])),int(y)) for x,y in replay]
            interventions.append({"seed":seed,"mode":mode,"task0_after_learning":first_score,
                                  "task0_final":normal,"task0_old_head_final_hidden":old_head,
                                  "task0_old_hidden_final_head":old_hidden,
                                  "replay_groups_covered":len(set(groups)),"possible_groups":80,
                                  "missing_groups":80-len(set(groups)),"buffer_size":len(replay),
                                  "readout_l2_drift":float(np.linalg.norm(final['w_out']-first['w_out'])),
                                  "hidden_l2_drift":float(np.linalg.norm(final['w_in']-first['w_in']))})
    return {"all_15_trials_recomputed_and_checkpoints_verified":True,"metrics":rows,"counterfactuals":interventions,
            "interpretation":"Counterfactual head/hidden swaps on task0, not an additive causal decomposition; max-over-checkpoints is optimistic with finite test samples."}
