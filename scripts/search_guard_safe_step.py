"""Under a fixed, unmodified guard: which pair is lost, and can a smaller step fit?

The guard rejects any update that breaks a currently-passing pair. Rather than
weakening the guard, search the update SIZE: a smaller step may reach the target
while costing nothing. Also names the exact pair that blocks, so the choice is
evidence-based instead of a guess.

No rule, threshold, guard or architecture change. Discarded copies only.
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

LIVE = ROOT / 'local' / 'generation1'
TOP = [('bạn tên gì', 'Tôi là SCAR-X.'), ('cảm ơn', 'Không có gì.'),
       ('chào bạn', 'Chào bạn.'), ('tạm biệt', 'Tạm biệt bạn.')]
STEPS = [1, 2, 3, 4, 6, 8]


def trial(prompt, answer, steps):
    work = Path(tempfile.mkdtemp(prefix='scarx-step-')) / 'session'
    shutil.copytree(LIVE, work, ignore=shutil.ignore_patterns('session.lock'))
    app = Generation(work, background=False)
    try:
        before = normalized(app.model.generate(prompt)['text'])
        guard = app._guard_pairs(prompt)
        g_before = app._guard_scores(app.model, guard)
        pairs_before = list(zip(guard, g_before))
        app.model.expand_vocabulary(answer)
        for _ in range(steps):
            app.model.train_pair(prompt, answer)
        g_after = app._guard_scores(app.model, guard)
        lost_pairs = [(p, t) for (p, t), b, a in zip(guard, g_before, g_after) if b and not a]
        gained = sum((not b) and a for b, a in zip(g_before, g_after))
        after = normalized(app.model.generate(prompt)['text'])
        return {'prompt': prompt, 'answer': answer, 'steps': steps,
                'before': before, 'after': after,
                'reaches_target': normalized(after) == normalized(answer),
                'guard_lost': len(lost_pairs), 'guard_gained': gained,
                'lost_pairs': [{'prompt': p, 'target': t} for p, t in lost_pairs],
                'accepted': len(lost_pairs) == 0}
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


rows = []
print('=== smallest guard-safe step that reaches the target ===')
for prompt, answer in TOP:
    print(f'\n{prompt!r} -> {answer!r}')
    for steps in STEPS:
        r = trial(prompt, answer, steps)
        rows.append(r)
        mark = 'ACCEPT' if r['accepted'] else f"BLOCK({r['guard_lost']})"
        print(f"  steps={steps:<2d} {mark:9s} reaches={str(r['reaches_target']):5s} "
              f"gained={r['guard_gained']}  {r['before']!r} -> {r['after']!r}")
        if r['lost_pairs']:
            lp = r['lost_pairs'][0]
            print(f"           lost: {lp['prompt']!r} (wanted {lp['target']!r})")

ok = [r for r in rows if r['accepted'] and r['reaches_target']]
print(f"\nguard-safe AND reaches target: {len(ok)}/{len(rows)}")
for r in ok:
    print(f"  steps={r['steps']} {r['prompt']!r} -> {r['answer']!r}")

json.dump(rows, open(ROOT / 'local' / 'step-search.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('\nwrote local/step-search.json')
