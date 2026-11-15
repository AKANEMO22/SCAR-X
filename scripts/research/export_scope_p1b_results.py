"""Post-run data export/figure only. No model import or learning."""
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"
import argparse
import csv
import itertools
import json
from pathlib import Path
import time
import numpy as np


def main():
    start=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True)
    args=ap.parse_args();run=args.run;phase=run/"learning";out=run/"summary";out.mkdir(exist_ok=False)
    records=json.loads((phase/"completed.json").read_text());table=[];extra=[];writes=0;changed_updates=0
    with (out/"raw-predictions.csv").open("w",encoding="utf-8",newline="") as f:
        fields=["seed","sign","arm","dataset","update","case_id","label","prediction","p0","p1","correct"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();nrows=0
        for r in sorted(records,key=lambda r:(r["seed"],r["sign"],r["arm"])):
            d=run/r["relative_directory"]
            with np.load(phase/f"fit-{r['seed']}.npz",allow_pickle=False) as data:oldids=data["old_ids"]
            with np.load(d/"predictions.npz",allow_pickle=False) as z:
                for u in (0,48,96,192):
                    for dataset,pkey,ykey,ids in (("DEV",f"dev_p_u{u}","dev_y",z["dev_ids"]),
                                                  ("old_valid",f"old_p_u{u}","old_y",oldids)):
                        for p,y,identifier in zip(z[pkey],z[ykey],ids):
                            pred=int(p.argmax());writer.writerow({"seed":r["seed"],"sign":r["sign"],"arm":r["arm"],
                                "dataset":dataset,"update":u,"case_id":str(identifier),"label":int(y),"prediction":pred,
                                "p0":float(p[0]),"p1":float(p[1]),"correct":int(pred==y)});nrows+=1
                for p,y,identifier in zip(z["revert_hidden_p"],z["dev_y"],z["dev_ids"]):
                    pred=int(p.argmax());writer.writerow({"seed":r["seed"],"sign":r["sign"],"arm":r["arm"],
                        "dataset":"DEV_hidden_revert","update":192,"case_id":str(identifier),"label":int(y),
                        "prediction":pred,"p0":float(p[0]),"p1":float(p[1]),"correct":int(pred==y)});nrows+=1
            with np.load(d/"learning-diagnostics.npz",allow_pickle=False) as z:
                actual=z["actual_hidden_deltas"]
                writes+=int(np.count_nonzero(actual));changed_updates+=int(np.count_nonzero(np.linalg.norm(actual,axis=(1,2))))
    for sign in (1,-1):
        for arm in ("label","teacher","shuffled","readout"):
            rows=sorted([r for r in records if r["sign"]==sign and r["arm"]==arm],key=lambda r:r["seed"])
            table.append({"sign":sign,"arm":arm,"per_seed_DEV":[r["final_dev"]["accuracy"] for r in rows],
                "mean_DEV":float(np.mean([r["final_dev"]["accuracy"] for r in rows])),
                "positive_recall":[r["final_dev"]["class_recall"][1] for r in rows],
                "mean_old_valid":float(np.mean([r["final_old"]["accuracy"] for r in rows])),
                "mean_hidden_revert":float(np.mean([r["hidden_revert"]["accuracy"] for r in rows]))})
    # Exploratory seed-level sensitivity, not a replacement for preregistered decision.
    seeds=sorted(set(r["seed"] for r in records))
    for arm in ("label","shuffled","readout"):
        effects=[]
        for seed in seeds:
            b=[r["final_dev"]["accuracy"] for r in records if r["seed"]==seed and r["arm"]=="teacher"]
            a=[r["final_dev"]["accuracy"] for r in records if r["seed"]==seed and r["arm"]==arm]
            effects.append(float(np.mean(b)-np.mean(a)))
        observed=abs(np.mean(effects))
        flipped=[abs(np.mean(np.asarray(effects)*s)) for s in itertools.product((-1,1),repeat=3)]
        extra.append({"control":arm,"paired_seed_effects":effects,
                      "exact_8_sign_flip_p_descriptive":sum(x>=observed-1e-12 for x in flipped)/8,
                      "not_registered_as_quality_decision":True})
    plot=False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors=["#9aa0a6","#2877b5","#d39a2c","#70777e"]
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,sign in zip(axes,(1,-1)):
            rows=[t for t in table if t["sign"]==sign]
            for i,(row,color) in enumerate(zip(rows,colors)):
                ax.bar(i,row["mean_DEV"]*100,color=color,width=.65)
                ax.scatter([i-.12,i,i+.12],np.asarray(row["per_seed_DEV"])*100,color="#202020",s=18,zorder=3)
                ax.text(i,row["mean_DEV"]*100+2,f"{row['mean_DEV']*100:.2f}",ha="center",fontsize=9)
            ax.set_xticks(range(4),["Label","Teacher","Shuffled","Readout"])
            ax.set_title("Initial readout "+("+" if sign==1 else "−"));ax.set_ylim(0,105)
            ax.axhline(75,color="#666666",ls=":",lw=1);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
        axes[0].set_ylabel("DEV accuracy (%)")
        fig.suptitle("SCAR-X P1b: temporal ALIF transfer")
        fig.text(.5,.01,"Dots: 3 seeds. Majority baseline: 75%. G2 failed; TEST/restart/reuse not run.",ha="center",fontsize=9)
        fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(out/"accuracy-by-readout-sign.png",dpi=160);plt.close(fig);plot=True
    except ImportError:pass
    summary={"table":table,"posthoc_seed_sensitivity":extra,"raw_prediction_rows":nrows,
             "nonzero_hidden_write_events":writes,"updates_with_nonzero_hidden_delta":changed_updates,
             "neural_calls_added":0,"learning_updates_added":0,"plot_created":plot,
             "wall_seconds":time.perf_counter()-start}
    (out/"export-summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary))


if __name__=="__main__":main()
