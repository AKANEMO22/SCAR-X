"""Export measured seed-cluster confidence intervals; no fitted trends."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
summary=json.loads((root/'artifacts/phase21-summary.json').read_text())
data=json.loads((root/summary['lifelong']/'result.json').read_text())
modes=list(data['summaries'])
labels=['Readout\nEMA','Score\nEMA','Score\nmeta','Meta\ndecay','Selective','Random\nreject','SI\nreadout']
fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
for ax,key,title in zip(axes,('final_accuracy','forgetting'),('Final clean accuracy','Mean forgetting')):
    values=[data['summaries'][m][key] for m in modes]
    means=np.array([v['mean'] for v in values])*100
    lower=np.array([v['lower95'] for v in values])*100
    upper=np.array([v['upper95'] for v in values])*100
    ax.bar(np.arange(len(modes)),means,color=['#526b8c','#526b8c','#cc8147','#cc8147','#6b8e81','#8c929b','#526b8c'],width=.65)
    ax.errorbar(np.arange(len(modes)),means,yerr=np.vstack((means-lower,upper-means)),fmt='none',ecolor='#222222',capsize=3)
    ax.set_xticks(np.arange(len(modes)),labels,fontsize=9)
    ax.set_title(title)
    ax.set_axisbelow(True); ax.grid(axis='y',alpha=.2)
    ax.spines[['right','top']].set_visible(False)
    if key=='final_accuracy':
        ax.set_ylim(0,75); ax.set_ylabel('Percent correct')
        ax.axhline(50,color='#888888',ls=':',lw=1.3,label='Chance: 50%')
        ax.axhline(65,color='#9d2f2f',ls='--',lw=1.2,label='Required: 65%')
    else:
        ax.set_ylim(0,16); ax.set_ylabel('Percentage points')
        ax.axhline(3,color='#9d2f2f',ls='--',lw=1.2,label='Required: <= 3 pp')
    ax.legend(frameon=False,fontsize=8,loc='upper left')
fig.suptitle('SCAR-X Phase 2.1 — 8 paired seeds, fixed 48-neuron network',fontsize=13)
fig.supxlabel('Error bars: seed-cluster bootstrap 95% CI. No method passes the joint quality/retention gate.',fontsize=9)
output=root/'artifacts/phase21-figures'; output.mkdir(parents=True,exist_ok=True)
fig.savefig(output/'lifelong_accuracy_forgetting.png',dpi=180)
fig.savefig(output/'lifelong_accuracy_forgetting.svg')
print(output)
