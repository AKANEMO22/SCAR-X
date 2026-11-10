"""Does vocabulary expansion break previously-correct pairs?

Hypothesis: teach() expands the vocabulary, which changes the input dimension and
therefore every previously learned input->output mapping. If true, the guard is
mostly reacting to vocabulary growth rather than to genuine interference, and the
fix is an ordering/expansion policy rather than a weaker guard.

Compares teaching with a target already fully inside the vocabulary against a
target that requires new tokens. Runs on discarded session copies only.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, normalized  # noqa: E402

SOURCE = ROOT / 'local' / 'generation1'


def trial(prompt, target, repeats=8, expand=True):
    work = Path(tempfile.mkdtemp(prefix='scarx-vocab-')) / 'session'
    shutil.copytree(SOURCE, work, ignore=shutil.ignore_patterns('session.lock'))
    app = Generation(work, background=False)
    try:
        vocab_before = len(app.model.vocabulary)
        guard = app._guard_pairs(prompt)
        g_before = app._guard_scores(app.model, guard) if guard else []
        added = []
        if expand:
            added = [t for t in __import__('scarx.neural.language', fromlist=['tokenize']).tokenize(target)
                     if t not in app.model.ids]
            app.model.expand_vocabulary(target)
        for _ in range(repeats):
            app.model.train_pair(prompt, target)
        g_after = app._guard_scores(app.model, guard) if guard else []
        lost = sum(b and not a for b, a in zip(g_before, g_after))
        return {
            'prompt': prompt, 'target': target,
            'vocab_before': vocab_before, 'vocab_after': len(app.model.vocabulary),
            'vocab_grew': len(app.model.vocabulary) != vocab_before,
            'new_tokens': added,
            'guard_lost': lost, 'guard_gained': sum((not b) and a for b, a in zip(g_before, g_after)),
            'guard_pairs': len(guard),
            'passing_before': sum(g_before),
            'output_after': normalized(app.model.generate(prompt)['text']),
        }
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


cases = [
    # target words already present in the shipped vocabulary
    ('chào bạn', 'chào bạn'),
    ('tạm biệt', 'tạm biệt'),
    ('bạn là ai', 'tôi là scar-x'),
    # target likely needing new tokens
    ('cảm ơn', 'Không có gì.'),
    ('bạn khỏe không', 'Tôi đang hoạt động.'),
]

rows = []
for prompt, target in cases:
    r = trial(prompt, target)
    rows.append(r)
    grow = f"+{r['vocab_after'] - r['vocab_before']}" if r['vocab_grew'] else "0 "
    print(f"vocab{grow:4s} lost={r['guard_lost']} gained={r['guard_gained']} "
          f"passing_before={r['passing_before']}/{r['guard_pairs']}  {prompt!r} -> {target!r}")
    print(f"        out={r['output_after']!r}  new_tokens={r['new_tokens']}")

grew = [r for r in rows if r['vocab_grew']]
same = [r for r in rows if not r['vocab_grew']]
print()
print(f"vocab GREW : n={len(grew)}  mean guard_lost={sum(r['guard_lost'] for r in grew)/max(1,len(grew)):.2f}")
print(f"vocab SAME : n={len(same)} mean guard_lost={sum(r['guard_lost'] for r in same)/max(1,len(same)):.2f}")
json.dump(rows, open(ROOT / 'local' / 'vocab-expansion-effect.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('\nwrote local/vocab-expansion-effect.json')
