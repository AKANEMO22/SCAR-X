"""Actual registered static-alpha results only. Zero training or selection."""
from pathlib import Path
import argparse,hashlib,json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'local/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();run=(ROOT/args.run).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Owned research run required')
    rows=[]
    for path in (run/'jobs').glob('*/result-0.json'):
        r=json.loads(path.read_text(encoding='utf-8'))
        if r['budget']=='equal_updates' and r['profile'] in ('static_0','static_25','static_50','static_75','static_100'):
            r['_raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(r)
    out=run/'figures';out.mkdir(exist_ok=True)
    if (out/'static-credit-screen.png').exists():raise ValueError('Never overwrite published figure')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True,layout='constrained');points=[]
    for axis,task in zip(axes,('strict_binding','xor','distractor_binding')):
        for metric,color,style in [('id_direct','#006c94','-'),('id_probe','#529e45',':'),('ood_direct','#cb5736','--')]:
            means=[];low=[];high=[]
            for alpha in (0,25,50,75,100):
                cells=sorted((r for r in rows if r['task']==task and r['profile']==f'static_{alpha}'),key=lambda r:r['seed'])
                if [r['seed'] for r in cells]!=[140011,140017,140023]:raise ValueError('Incomplete static group')
                values=np.asarray([r['metrics'][metric] for r in cells])*100
                means.append(values.mean());low.append(values.min());high.append(values.max())
                points.append({'task':task,'alpha':alpha/100,'metric':metric,'seed_values':values.tolist(),
                   'raw_sha256':[r['_raw_sha256'] for r in cells]})
            xs=np.linspace(0,1,5);axis.plot(xs,means,style,marker='o',color=color,label=metric)
            if metric=='id_direct':axis.fill_between(xs,low,high,color=color,alpha=.12,label='direct seed range')
        axis.axhline(65,color='gray',linewidth=.7,linestyle='--');axis.set_title(task.replace('_',' '));axis.set_xlabel('Static alpha (0=own, 1=stopped)');axis.set_xticks(np.linspace(0,1,5));axis.grid(alpha=.2)
    axes[0].set_ylabel('Accuracy (%)');axes[0].set_ylim(40,102);axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('SCAR-X: completed static screen; 3 exploratory DEV seeds, not confirmation',fontsize=11)
    fig.savefig(out/'static-credit-screen.png',dpi=160);plt.close(fig)
    with (out/'static-credit-screen.json').open('x',encoding='utf-8') as f:json.dump({'points':points,'new_training_updates':0,'scope':'Descriptive mean; shading is min/max seed range, not confidence interval'},f,indent=2)
    print(str(out/'static-credit-screen.png'))
if __name__=='__main__':main()
