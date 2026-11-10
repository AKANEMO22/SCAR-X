"""Audit the current guard: which pairs it really protects on the live session.

Step 1 of the user's plan: inspect before changing. This prints, for every guard
pair, the model output and both criteria, so any pair that can never pass (because
its stored target is shorter than the real answer) is visible.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, PRODUCT_GUARD_PROMPTS  # noqa: E402
from scarx.evaluation.product_acceptance import copy_session  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

LIVE = ROOT / 'local' / 'generation1'
# The live app holds the session lock; audit a copy instead.
work = copy_session(LIVE)
app = Generation(work, background=False)
try:
    pairs = app._guard_pairs('__audit__')
    print(f'guard pairs: {len(pairs)}\n')
    print(f"{'prompt':18s} {'stored target':26s} {'model output':26s} content complete")
    rows = []
    for prompt, target in pairs:
        clone = UnifiedCore.loads(app.model.dumps())
        clone.core.reset_state()
        clone.last_active[:] = clone.ticks
        out = clone.generate(prompt)['text']
        v = Generation._pair_verdict(out, target)
        rows.append((prompt, target, out, v))
        print(f'{prompt[:16]!r:18s} {target[:24]!r:26s} {out[:24]!r:26s} '
              f'{str(v["content_ok"]):7s} {v["complete_ok"]}')

    n_content = sum(r[3]['content_ok'] for r in rows)
    n_complete = sum(r[3]['complete_ok'] for r in rows)
    print(f'\ncontent_ok  : {n_content}/{len(rows)}')
    print(f'complete_ok : {n_complete}/{len(rows)}')

    print('\n=== pairs whose STORED TARGET is shorter than the real answer ===')
    suspicious = [r for r in rows if not r[3]['content_ok'] and
                  all(w.lower().strip('.,!?') in r[2].lower() for w in r[1].split())]
    for prompt, target, out, v in suspicious:
        print(f'  {prompt!r}')
        print(f'    stored target {target!r} vs output {out!r}')
        print(f'    -> target is a SUBSTRING of output, so exact compare fails.')
        print(f'       This pair can never pass and therefore protects nothing.')
    if not suspicious:
        print('  none')
finally:
    app.close()
    import shutil
    shutil.rmtree(work.parent, ignore_errors=True)
