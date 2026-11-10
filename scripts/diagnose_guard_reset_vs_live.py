"""Does the guard score a different thing than the product exposes?

Known from earlier measurement: _guard_scores() scores a RESET-STATE clone while
the product answers from continuous live state. If a pair passes in the clone
but fails live (or vice versa), the guard is protecting something the user never
sees. Quantify on the same copy, no changes.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, normalized  # noqa: E402
from scarx.evaluation.product_acceptance import copy_session  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

work = copy_session(ROOT / 'local' / 'generation1')
app = Generation(work, background=False)
try:
    pairs = app._guard_pairs('__x__')
    print(f'guard pairs: {len(pairs)}\n')
    print(f"{'prompt':38s} {'target':20s} {'reset':7s} {'live':7s} agree")
    diverge = []
    for prompt, target in pairs:
        clone = UnifiedCore.loads(app.model.dumps())
        clone.core.reset_state()
        clone.last_active[:] = clone.ticks
        reset_out = clone.generate(prompt)['text']
        live_out = app.model.generate(prompt)['text']
        r_ok = normalized(reset_out) == normalized(target)
        l_ok = normalized(live_out) == normalized(target)
        agree = r_ok == l_ok
        if not agree:
            diverge.append((prompt, target, reset_out, live_out, r_ok, l_ok))
        print(f'{prompt[:36]!r:38s} {target[:18]!r:20s} '
              f"{'PASS' if r_ok else 'fail':7s} {'PASS' if l_ok else 'fail':7s} {agree}")

    print(f'\npairs where reset and live DISAGREE: {len(diverge)}/{len(pairs)}')
    for d in diverge:
        print(f'  {d[0]!r}')
        print(f'    guard(reset) : {d[2]!r}  -> {"PASS" if d[4] else "fail"}')
        print(f'    product(live): {d[3]!r}  -> {"PASS" if d[5] else "fail"}')
    print()
    if diverge:
        print('CONCLUSION: the guard and the product disagree, so the guard can')
        print('accept a lesson that breaks what the user actually sees, and can')
        print('reject a lesson that would have fixed a live behaviour.')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
