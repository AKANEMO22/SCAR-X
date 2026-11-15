"""Plot completed saved20stage trajectories; zero model calls or score changes."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    run=Path((ROOT/'local/current-shared-lora-retention20.txt').read_text(encoding='utf-8').strip())
    result=json.loads((run/'result.json').read_text(encoding='utf-8'))
    assert len(result['stages'])==20 and len(result['updates'])==480
    target=ROOT/'artifacts'/run.name;target.mkdir(exist_ok=True)
    stages=[result['baseline']]+result['stages'];x=[0]+[s['stage']+1 for s in result['stages']]
    fig,axes=plt.subplots(2,1,figsize=(9,7),layout='constrained')
    for api in ('AddAlarm','AddReminder','CompleteReminder','DeleteAlarm'):
        axes[0].plot(x,[100*s['by_API'][api] for s in stages],marker='.',linewidth=1.5,label=api)
    axes[0].set(xlabel='Completed learning stages (0 = initial trained adapter)',ylabel='Exact TEST repair (%)',ylim=(-3,103),
        title='One shared VLM LoRA: known-task retention across 20 stages (not SNN)')
    axes[0].legend(ncol=2,loc='lower right');axes[0].grid(alpha=.2)
    for is_replay,color,label in [(False,'#2266aa','Current stream'),(True,'#cc8833','Replay')]:
        points=[r for r in result['updates'] if r['replay']==is_replay]
        axes[1].scatter([r['step'] for r in points],[r['loss'] for r in points],s=7,alpha=.5,color=color,label=label)
    axes[1].set(xlabel='Actual optimizer update',ylabel='Completion-only cross entropy',title='TRAIN-only updates; 68 unique rows, repeated exposures')
    axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(target/'retention20-and-loss.png',dpi=160)
    print(str(target/'retention20-and-loss.png'))


if __name__=='__main__':main()
