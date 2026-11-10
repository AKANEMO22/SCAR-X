"""Why did the guard ACCEPT a lesson that still lost 'chào bạn'?

This is the exact defect the user flagged: the guard compares against a fixed
probe list, not against the product's own acceptance set. A regression outside
that list is invisible to it.

Read-only analysis on copies; no rule change.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, normalized  # noqa: E402
from scarx.evaluation.product_acceptance import (  # noqa: E402
    copy_session, guard_report, measure, weights_sha256,
)

LIVE = ROOT / 'local' / 'generation1'
# Reproduce the state that accepted 'tạm biệt' and still lost 'chào bạn'.
BEFORE = [('bạn tên gì', 'Tôi là SCAR-X.'), ('cảm ơn', 'Không có gì.')]
LESSON = ('tạm biệt', 'Tạm biệt bạn.')

work = copy_session(LIVE)
app = Generation(work, background=False)
try:
    def correct_set():
        rows = measure(app)
        return {r['prompt'] for r in rows if r['correct']}, rows

    for p, a in BEFORE:
        try:
            app.teach({'prompt': p, 'answer': a, 'repeats': 16, 'confirmation': True})
        except Exception:
            pass
    cs_before, _ = correct_set()

    guard = app._guard_pairs(LESSON[0])
    g_before = app._guard_scores(app.model, guard)
    guard_names_before = {(p, t) for (p, t), ok in zip(guard, g_before) if ok}
    print('guard pairs that PASS before the lesson:')
    for p, t in sorted(guard_names_before):
        print(f'   {p[:44]!r:48s} -> {t[:18]!r}')

    print(f"\nproduct bench correct before: {sorted(cs_before)}")
    print("is 'chào bạn' in the guard list? ->",
          any(normalized(p) == normalized('chào bạn') for p, _ in guard_names_before))
    print("is 'chào bạn' in ANY guard pair?  ->",
          any(normalized(p) == normalized('chào bạn') for p, _ in guard))

    sha_before = weights_sha256(app.model)
    accepted = False
    try:
        app.teach({'prompt': LESSON[0], 'answer': LESSON[1], 'repeats': 16, 'confirmation': True})
        accepted = True
    except Exception as exc:
        print(f'\nlesson rejected: {type(exc).__name__}')

    cs_after, rows_after = correct_set()
    lost = sorted(cs_before - cs_after)
    g_after = app._guard_scores(app.model, guard)
    guard_lost = [(p, t) for (p, t), b, a in zip(guard, g_before, g_after) if b and not a]

    print(f'\naccepted by guard : {accepted}')
    print(f'guard_lost pairs  : {guard_lost}')
    print(f'product newly_lost: {lost}')
    print(f'weights changed   : {sha_before != weights_sha256(app.model)}')
    print()
    print('CONCLUSION:'
          if lost and not guard_lost else
          'CONCLUSION: no divergence')
    if lost and not guard_lost:
        print('  The guard accepted because it only scores its own fixed probe list.')
        print("  'chào bạn' was NOT in that list, so its loss was invisible.")
        print('  This is a guard Coverage bug, not a guard Threshold bug:')
        print('  loosening or tightening the threshold cannot fix it.')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
