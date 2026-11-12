"""Real frozen C-router traces; descriptive averages, not verbal thought."""
from pathlib import Path
import argparse,json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'local/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=(ROOT/a.run).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Owned run required')
    source=run/'router-analysis-v1.json';d=json.loads(source.read_text(encoding='utf-8'))
    if d['status']!='PASS':raise ValueError('Require validated router traces')
    output=run/'figures/router-alpha.png'
    if output.exists():raise ValueError('Do not overwrite scientific figures')
    output.parent.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(12,5.8),sharey=True,layout='constrained')
    for column,task in enumerate(('strict_binding','xor','distractor_binding')):
        for profile in ('state','state_shuffle','random_alpha','state_signal_shuffle'):
            row=d['summary'][profile][task]
            temporal=row['alpha_per_tick_mean'];neurons=row['alpha_per_neuron_mean']
            axes[0,column].plot(np.arange(1,len(temporal)+1),temporal,label=profile)
            axes[1,column].plot(np.arange(1,len(neurons)+1),neurons,label=profile)
        axes[0,column].set_title(task.replace('_',' '));axes[0,column].set_xlabel('Tick (12 per episode)')
        axes[1,column].set_xlabel('Neuron index (48 allocated)')
        for axis in axes[:,column]:axis.set_ylim(0,1);axis.grid(alpha=.2)
    axes[0,0].set_ylabel('Mean alpha across neurons');axes[1,0].set_ylabel('Mean alpha across ticks')
    axes[0,2].legend(fontsize=7)
    fig.suptitle('Actual fixed-router alpha: final checkpoint TRAIN replay, 0 learning updates\nDifferent trained weights; curves are descriptive, not matched-state causal comparisons',fontsize=10)
    fig.savefig(output,dpi=160);plt.close(fig);print(str(output))
if __name__=='__main__':main()
