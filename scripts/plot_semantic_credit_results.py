"""Scientific figure from immutable actual DEV trajectories, not a mock UI."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = Path((ROOT / 'local/current-semantic-metaplasticity-DEV.txt').read_text(encoding='utf-8').strip())
    diagnosis = Path((ROOT / 'local/current-semantic-three-seed-diagnosis.txt').read_text(encoding='utf-8').strip())
    curves = json.loads((source / 'curves.json').read_text(encoding='utf-8'))
    result = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    measured = json.loads((diagnosis / 'result.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3))
    colors = {'unprotected': '#3375b9', 'protected': '#20846b', 'norm_matched': '#ae7446'}
    names = {'unprotected': 'Signed error', 'protected': 'Importance + stable anchor', 'norm_matched': 'Matched update norm'}
    seeds = ('260113', '260137', '260171')
    for rule in colors:
        values = np.array([curves[seed][rule] for seed in seeds]).mean(axis=2) * 100
        mean = values.mean(axis=0)
        axes[0, 0].plot(np.arange(1, 21), mean, label=names[rule], color=colors[rule])
        axes[0, 0].fill_between(np.arange(1, 21), values.min(axis=0), values.max(axis=0), color=colors[rule], alpha=.12)
    axes[0, 0].set(title='Actual DEV accuracy over 20 stages', xlabel='Learning stage', ylabel='Accuracy (%)', ylim=(0, 100))
    axes[0, 0].set_xticks([1, 5, 10, 15, 20])
    axes[0, 0].legend(fontsize=8)
    for col, (field, threshold, title) in enumerate((('mean_forgetting', 3, 'Final mean forgetting'), ('worst_forgetting', 5, 'Per-seed worst intent forgetting'))):
        ax = axes[1, col]
        for index, rule in enumerate(colors):
            vals = np.array([result['summary'][s][rule][field] for s in seeds]) * 100
            ax.bar(index, vals.mean(), color=colors[rule], alpha=.75)
            ax.scatter(np.full(3, index) + [-.1, 0, .1], vals, color='black', s=18)
        ax.axhline(threshold, color='#bb3e4b', linestyle='--', label=f'Registered limit {threshold} pp')
        ax.set_xticks(range(3), ['Signed error', 'Protected', 'Norm matched'])
        ax.set(title=title, ylabel='Percentage points', ylim=(0, 48))
        ax.legend(fontsize=8)
    ax = axes[0, 1]
    order = ('frozen', 'norm_matched', 'protected', 'unprotected', 'rawPCA64')
    means = [measured['mean_common_head_accuracy'][k] * 100 for k in order]
    ax.bar(range(5), means, color=['#8b94a1', '#ae7446', '#20846b', '#3375b9', '#7463a1'])
    for index, key in enumerate(order):
        ax.scatter(np.full(3, index) + [-.1, 0, .1], [measured['scores'][s][key] * 100 for s in seeds], color='black', s=18)
    ax.set_xticks(range(5), ['Frozen', 'Norm\nmatched', 'Protected', 'Signed\nerror', 'Raw\nPCA64'])
    ax.set(title='Identical GPU-fitted readout · final DEV', ylabel='Accuracy (%)', ylim=(0, 100))
    for a in axes.flat:
        a.grid(axis='y', alpha=.16)
        a.set_axisbelow(True)
    fig.suptitle('SCAR-X ALIF H128 · local learning on CUDA · 3 seeds', fontsize=15)
    fig.text(.5, .012, 'Bars: seed means; dots: seeds; shading: seed range. Reused DEV only; no fresh TEST or SNN-over-RAG PASS.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    path = ROOT / 'artifacts' / source.name / 'credit-retention-and-representation.png'
    fig.savefig(path, dpi=170)
    print(str(path))


if __name__ == '__main__':
    main()
