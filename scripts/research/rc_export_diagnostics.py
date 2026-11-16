"""Saved-result tables and scientific figure. No model calls or fitting."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np


def main():
    start=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(exist_ok=False);records=json.loads((args.run/"D1-results.json").read_text());rows=[]
    for r in sorted(records,key=lambda r:(r["sign"],r["arm"],r["seed"])):
        for method,m in (("direct",r["direct"]),("raw_common_probe",r["primary_probe"]["DEV"]),("q_common_probe",r["secondary_probe_q"]["DEV"]),
                         ("initial_common_probe",r["initial_hidden_probe"]["DEV"])):
            rows.append({"seed":r["seed"],"sign":r["sign"],"arm":r["arm"],"method":method,
                         **{k:m[k] for k in ("accuracy","balanced_accuracy","positive_recall","negative_recall","AUC","mean_signed_margin","min_signed_margin")}})
    with (args.output/"diagnostic-metrics.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,sharey=True,figsize=(11,4.4))
    names=("label","teacher","shuffled","readout")
    for ax,sign in zip(axes,(1,-1)):
        for j,(method,color) in enumerate((("direct","#8793a1"),("raw_common_probe","#2877b5"))):
            for i,arm in enumerate(names):
                values=[r["balanced_accuracy"]*100 for r in rows if r["sign"]==sign and r["arm"]==arm and r["method"]==method]
                x=i+(j-.5)*.34
                ax.bar(x,np.mean(values),width=.32,color=color,label=method if i==0 else None)
                ax.scatter([x-.05,x,x+.05],values,color="#222222",s=16,zorder=3)
        ax.set_xticks(range(4),["Label","Teacher","Shuffled","Readout"]);ax.set_ylim(0,108)
        ax.set_title("Initial readout "+("+" if sign==1 else "−"));ax.axhline(50,color="#777777",ls=":",lw=1)
        ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel("DEV balanced accuracy (%)");axes[0].legend(loc="lower left",fontsize=8)
    fig.suptitle("P1b frozen checkpoints: online head vs fixed common ridge probe")
    fig.text(.5,.015,"Dots: 3 fitted seeds. Probe is diagnostic; P1b G2 remains FAIL. No hidden updates or TEST.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(args.output/"direct-vs-common-probe.png",dpi=160);plt.close(fig)
    receipt={"rows":len(rows),"new_model_calls":0,"new_fits":0,"wall_seconds":time.perf_counter()-start}
    (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt))


if __name__=="__main__":main()
