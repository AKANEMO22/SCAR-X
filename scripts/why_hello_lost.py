"""Why does teaching 'xin chào' still lose 'hello' without being blocked?

The guard only blocks when a pair passes BEFORE and fails AFTER. So either
  (a) 'hello' was not passing before, or
  (b) the guard list used during teach() differs from the one audited.

Print the exact before/after verdicts the guard itself computes for that lesson.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, LearningGuardRejected  # noqa: E402
from scarx.evaluation.product_acceptance import copy_session  # noqa: E402

LESSON = ('xin chào', 'Chào bạn.')

work = copy_session(ROOT / 'local' / 'generation1')
app = Generation(work, background=False)
try:
    pairs = app._guard_pairs(LESSON[0])
    print(f'pairs passed to the guard for {LESSON!r}: {len(pairs)}')
    before = app._guard_scores(app.model, pairs)
    print('\nBEFORE (bitmask 1=content, 2=complete):')
    for (p, t), s in zip(pairs, before):
        print(f'  {s}  {p!r:26s} target={t[:20]!r}')

    hello_idx = [i for i, (p, _) in enumerate(pairs) if p == 'hello']
    print(f"\n'hello' index in guard list: {hello_idx}, "
          f"before score: {[before[i] for i in hello_idx]}")

    try:
        app.teach({'prompt': LESSON[0], 'answer': LESSON[1],
                   'repeats': 16, 'confirmation': True})
        print('\nteach: ACCEPTED')
    except LearningGuardRejected as exc:
        print(f"\nteach: rejected guard_lost={exc.metrics.get('guard_lost')}")

    after = app._guard_scores(app.model, pairs)
    print('\nAFTER:')
    for (p, t), b, a in zip(pairs, before, after):
        flag = ' <-- LOST' if (b & 1) and not (a & 1) else ''
        print(f'  {b}->{a}  {p!r:26s}{flag}')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
