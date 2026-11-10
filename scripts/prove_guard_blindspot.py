"""Prove WHY the guard accepted a lesson that lost 'chào bạn'.

Reproduces the exact state (two lessons applied) where the loss happens, then
compares, for that prompt, three things on the same model:
  A. guard scoring  (reset-state clone)   - what the guard protects
  B. product scoring(live continuous state) - what the user sees
  C. the guard's stored target for that prompt

If A fails while B passes, the pair was never "currently passing" from the
guard's point of view, so the guard could not notice losing it. That is the
mechanism, and it is a measurement-semantics bug, not a threshold bug.

Read-only on a copy.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, normalized  # noqa: E402
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, copy_session, guard_report, measure,
)
from scarx.neural.unified import UnifiedCore  # noqa: E402

BEFORE = [('bạn tên gì', 'Tôi là SCAR-X.'), ('cảm ơn', 'Không có gì.')]
LESSON = ('tạm biệt', 'Tạm biệt bạn.')
TARGET_PROMPT = 'chào bạn'

work = copy_session(ROOT / 'local' / 'generation1')
app = Generation(work, background=False)
try:
    for p, a in BEFORE:
        try:
            app.teach({'prompt': p, 'answer': a, 'repeats': 16, 'confirmation': True})
        except Exception as exc:
            print(f'(prep lesson {p!r} rejected: {type(exc).__name__})')

    pairs = app._guard_pairs(LESSON[0])
    stored = {normalized(p): t for p, t in pairs}
    guard_names = {normalized(p) for p, _ in pairs}

    print(f'is {TARGET_PROMPT!r} in the guard list now? ->',
          normalized(TARGET_PROMPT) in guard_names)
    print(f'stored guard target  : {stored.get(normalized(TARGET_PROMPT))!r}')

    def reset_out(prompt):
        c = UnifiedCore.loads(app.model.dumps())
        c.core.reset_state()
        c.last_active[:] = c.ticks
        return c.generate(prompt)['text']

    def product_rows():
        return {r['prompt']: r for r in measure(app, CASES)}

    prod_before = product_rows()
    reset_before = reset_out(TARGET_PROMPT)
    target = stored.get(normalized(TARGET_PROMPT))

    print('\n--- BEFORE the lesson ---')
    print(f'  A. guard (reset clone)  : {reset_before!r}')
    print(f'     stored target        : {target!r}')
    print(f'     guard counts as PASS : {normalized(reset_before) == normalized(target or "")}')
    print(f'  B. product (live state) : {prod_before.get(TARGET_PROMPT, {}).get("text")!r}')
    print(f'     product counts as OK : {prod_before.get(TARGET_PROMPT, {}).get("correct")}')

    accepted = False
    error = None
    try:
        app.teach({'prompt': LESSON[0], 'answer': LESSON[1], 'repeats': 16, 'confirmation': True})
        accepted = True
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'

    prod_after = product_rows()
    reset_after = reset_out(TARGET_PROMPT)

    print('\n--- AFTER the lesson ---')
    print(f'  accepted by guard       : {accepted}' + (f'  ({error[:60]})' if error else ''))
    print(f'  A. guard (reset clone)  : {reset_after!r}  '
          f'PASS={normalized(reset_after) == normalized(target or "")}')
    print(f'  B. product (live state) : {prod_after.get(TARGET_PROMPT, {}).get("text")!r}  '
          f'OK={prod_after.get(TARGET_PROMPT, {}).get("correct")}')

    print('\n--- DIAGNOSIS ---')
    guard_passed_before = normalized(reset_before) == normalized(target or '')
    product_ok_before = prod_before.get(TARGET_PROMPT, {}).get('correct')
    product_ok_after = prod_after.get(TARGET_PROMPT, {}).get('correct')
    if product_ok_before and not product_ok_after and not guard_passed_before:
        print('  CONFIRMED: the product answered this prompt correctly BEFORE the lesson,')
        print('  but the guard scored it as failing (reset clone), so it was not in the')
        print('  protected set and the guard could not detect the regression.')
        print('  Fix requires changing WHAT the guard measures (state semantics).')
        print('  It cannot be fixed by any threshold.')
    elif guard_passed_before:
        print('  Guard DID have it passing; regression slipped through elsewhere.')
    else:
        print('  Not reproduced on this path; see numbers above.')

    rows = guard_report(app, '__x__')
    div = []
    for g in rows:
        live = prod_after.get(g['prompt'])
        if live is None:
            continue
        if bool(g['pass']) != bool(live['correct']):
            div.append((g['prompt'], g['pass'], live['correct']))
    print(f'\n  reset-vs-live disagreements after lesson: {len(div)}')
    for d in div:
        print(f'    {d[0]!r}: guard_pass={d[1]} product_ok={d[2]}')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
