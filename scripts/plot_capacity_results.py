"""Static scientific plots from immutable capacity outputs, no model calls."""
from pathlib import Path
import argparse,json,os,hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'local/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=(ROOT/a.run).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Owned run only')
    audit=json.loads((run/'audit.json').read_text(encoding='utf-8'))
    if audit['integrity']!='PASS':raise ValueError('Completed audit required')
    folder=run/'figures';folder.mkdir(exist_ok=True)
    if (folder/'capacity-direct-ood.png').exists():raise ValueError('Never overwrite scientific figures')
    heights=[48,64,128];arms=('own','groups_up','readout_only');colors=('#006f9b','#b35b24','#598f48')
    tasks=('strict_binding','xor','distractor_binding');points=[]
    fig,axes=plt.subplots(2,3,figsize=(12,6.6),sharey=True,layout='constrained')
    for column,task in enumerate(tasks):
        for row,metric in enumerate(('id_direct','ood_direct')):
            axis=axes[row,column]
            for arm,color in zip(arms,colors):
                values=[audit['means'][str(h)][arm][task][metric]*100 for h in heights]
                axis.plot(heights,values,marker='o',label=arm,color=color)
                points.append({'task':task,'metric':metric,'arm':arm,'heights':heights,'mean_percent':values})
            axis.axhline(65,color='gray',linestyle='--',linewidth=.7);axis.grid(alpha=.2);axis.set_xticks(heights);axis.set_ylim(40,102)
            axis.set_title(task.replace('_',' '));axis.set_ylabel('ID direct (%)' if row==0 else 'OOD direct (%)');axis.set_xlabel('Allocated ALIF neurons')
    axes[0,0].legend(fontsize=8);fig.suptitle('Capacity diagnostic: fixed 1024 updates, 3 exploratory DEV seeds\nSame arm across H; no per-task winner selection',fontsize=11)
    fig.savefig(folder/'capacity-direct-ood.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),sharey=True,layout='constrained')
    for axis,task in zip(axes,tasks):
        for arm,color in zip(arms[:2],colors[:2]):
            values=[100*(audit['means'][str(h)][arm][task]['id_probe']-audit['means'][str(h)]['readout_only'][task]['id_probe']) for h in heights]
            axis.plot(heights,values,color=color,marker='o',label=arm)
            points.append({'task':task,'metric':'learned_minus_frozen_probe_pp','arm':arm,'heights':heights,'mean_pp':values})
        axis.axhline(0,color='gray',linewidth=.7);axis.grid(alpha=.2);axis.set_xticks(heights);axis.set_xlabel('Allocated ALIF neurons');axis.set_title(task.replace('_',' '))
    axes[0].set_ylabel('Learned − frozen probe (percentage points)');axes[0].legend(fontsize=8)
    fig.suptitle('Extra capacity versus actual hidden-learning advantage\nDescriptive means; probe gain does not replace direct skill',fontsize=11)
    fig.savefig(folder/'capacity-hidden-advantage.png',dpi=160);plt.close(fig)
    raw_files=list((run/'jobs').glob('*/result.json'))+list((run/'reused-h48').glob('*/capacity-result.json'))
    with (folder/'source-points.json').open('x',encoding='utf-8') as f:json.dump({'points':points,'new_model_calls':0,
      'source_sha256':{str(p.relative_to(run)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_files}},f,indent=2)
    print(str(folder))
if __name__=='__main__':main()
