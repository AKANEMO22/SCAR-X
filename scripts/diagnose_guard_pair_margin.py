"""Is the guard pair that blocks all teaching actually robust, or razor-thin?

Every teaching attempt at every step size loses the SAME pair. If that pair is
only barely correct, the guard is protecting a coincidence while blocking all
progress. Measure it on both states the product uses. Read-only.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np  # noqa: E402
from scarx.app.generation import Generation, normalized  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

PAIR = ('QA kiểm tra tab: 4 + 4 bằng mấy?', '8')

work = Path(tempfile.mkdtemp(prefix='scarx-margin-')) / 'session'
shutil.copytree(ROOT / 'local' / 'generation1', work,
                ignore=shutil.ignore_patterns('session.lock'))
app = Generation(work, background=False)
try:
    print('=== guard scores a RESET-STATE clone, product uses live state ===')
    clone = UnifiedCore.loads(app.model.dumps())
    clone.core.reset_state()
    clone.last_active[:] = clone.ticks
    reset_out = clone.generate(PAIR[0])['text']
    print(f'  reset clone (guard) : {reset_out!r}  match={normalized(reset_out) == normalized(PAIR[1])}')

    live_out = app.model.generate(PAIR[0])['text']
    print(f'  live state (product): {live_out!r}  match={normalized(live_out) == normalized(PAIR[1])}')

    print('\n=== confidence of the first token for the blocking pair ===')
    ctx = app.model.context(PAIR[0])
    app.model.boundary()
    p = app.model.word_step(1, ctx, 0, source=PAIR[0])
    order = np.argsort(-p)[:4]
    for i in order:
        print(f'  token={app.model.vocabulary[i]!r:16s} p={float(p[i]):.4f}')
    print(f'  margin top-second = {float(p[order[0]]) - float(p[order[1]]):.4f}')

    print('\n=== every guard pair under the reset-clone rule ===')
    pairs = app._guard_pairs('__probe__')
    passing = 0
    for prompt, target in pairs:
        c = UnifiedCore.loads(app.model.dumps())
        c.core.reset_state()
        c.last_active[:] = c.ticks
        out = c.generate(prompt)['text']
        ok = normalized(out) == normalized(target)
        passing += ok
        print(f"  {'PASS' if ok else 'fail'}  {prompt[:32]!r:36s} want={target[:16]!r:20s} got={out[:20]!r}")
    print(f'\npassing: {passing}/{len(pairs)}')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
