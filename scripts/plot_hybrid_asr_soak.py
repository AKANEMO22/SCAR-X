"""Standalone scientific figure from completed measured component evidence."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/20261114T080220Z-whisper-small-soak-v1-292b32'


def main():
    result=json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    audit=json.loads((RUN/'independent-soak-audit-v1.json').read_text(encoding='utf-8'))
    manifest=json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
    assert result['gate_pass'] and not result['failures'] and audit['mechanical_metrics']['gate']
    events=json.loads((RUN/'observer-audit-snapshot-v1.json').read_text(encoding='utf-8'))
    # Exact completionevent times, not invented evenly spaced measurements.
    calls=[e for e in events if e['kind']=='ASR_small_soak.actual_transcript']
    samples=result['global_free_samples']
    # Event archive lacks SQL created epoch. Active resource sample times give
    # actual resource axis; call counts use actual recorded source order.
    origin=min(row['epoch'] for row in samples)
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.5))
    indices=np.arange(1,len(result['calls'])+1)
    rtf=[row['RTF'] for row in result['calls']]
    axes[0].scatter(indices,rtf,s=13,c='#237d75',alpha=.8,label='Actual completed calls')
    axes[0].axhline(1,color='#b75c58',ls='--',lw=1,label='RTF 1.0')
    axes[0].axhline(audit['mechanical_metrics']['RTF_p95'],color='#4877aa',ls=':',lw=1.4,label='Measured p95')
    axes[0].set(xlabel='Completed ASR call (source order)',ylabel='Wall time / audio duration',
                title='Whisper-small FP32 · CUDA0',ylim=(0,1.08))
    axes[0].legend(loc='upper right',frameon=True,framealpha=.96,facecolor='white',edgecolor='none',fontsize=8)
    active=np.array([row['ASR_active'] for row in samples])
    times=np.array([(row['epoch']-origin)/60 for row in samples])
    free=np.array([row['free_MiB'] for row in samples])
    axes[1].plot(times,free,c='#8a96a5',lw=.7,alpha=.7,label='Global free VRAM samples')
    axes[1].scatter(times[active],free[active],s=7,c='#4877aa',alpha=.7,label='Owned ASR active')
    axes[1].axhline(512,color='#b75c58',ls='--',lw=1,label='512 MiB reserve')
    axes[1].set(xlabel='Measured time since first sample (minutes)',ylabel='Physical free VRAM (MiB)',
                title=f"Sampled minimum {int(free.min())} MiB",ylim=(0,max(free)*1.08))
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16)
    fig.suptitle('30-minute paced ASR component soak · no live promotion',fontsize=13)
    fig.text(.5,.025,
        '165 completed calls · 32 existing utterances (345.76 unique audio seconds) repeated\n'
        'Same resident VLM + mixed 8h workload. Sampling may miss short peaks; not a video-QA or learning benchmark.',
        ha='center',fontsize=8.5,color='#4b5563')
    fig.tight_layout(rect=(0,.13,1,.93))
    destination=RUN/'component-soak-v2.png'
    if destination.exists():raise FileExistsError('Refuse replacing historical figure')
    fig.savefig(destination,dpi=175,facecolor='white')
    fig.savefig(RUN/'component-soak-v2.svg',facecolor='white');plt.close(fig)
    print(destination)


if __name__=='__main__':main()
