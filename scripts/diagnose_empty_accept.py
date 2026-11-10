"""An accepted lesson changes weights but the answer stays wrong. Why?

Observed: teaching 'bạn tên gì' -> 'Tôi là scar-x.' is ACCEPTED at 1,2,4 repeats,
weights change, no regression - yet the model still answers 'Tôi là'.

Three candidates:
  A. the loss is still high, i.e. the update is too small to flip the argmax
  B. the answer IS reachable but only right after training, before any other call
  C. vocabulary growth moved the target token out of the reachable slice

Measure the trained pair immediately inside the same call sequence, and read the
loss trajectory, so the cause is evidenced rather than assumed.
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
from scarx.app.generation import Generation  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

LIVE = ROOT / 'local' / 'generation1'
PROMPT, TARGET = 'bạn tên gì', 'Tôi là scar-x.'


def read(model, prompt):
    c = UnifiedCore.loads(model.dumps())
    c.core.reset_state()
    c.last_active[:] = c.ticks
    return c.generate(prompt)['text']


work = Path(tempfile.mkdtemp(prefix='scarx-why2-')) / 'session'
shutil.copytree(LIVE, work, ignore=shutil.ignore_patterns('session.lock'))
app = Generation(work, background=False)
try:
    m = app.model
    print(f'before          : {read(m, PROMPT)!r}')
    print(f'vocab size      : {len(m.vocabulary)}')

    need = [t for t in str(TARGET).split() if t.lower().strip('.,!?') not in
            {v.lower() for v in m.vocabulary}]
    print(f'target tokens missing from vocab: {need}')

    additions = m.expand_vocabulary(TARGET)
    print(f'expand_vocabulary -> {additions}')

    print('\nloss trajectory (16 steps):')
    for i in range(16):
        loss = float(m.train_pair(PROMPT, TARGET))
        if i in (0, 1, 3, 7, 15):
            print(f'  step {i + 1:2d}  loss={loss:.4f}  read={read(m, PROMPT)!r}')

    print(f'\nafter 16 steps  : {read(m, PROMPT)!r}')

    # Is the target token at least in the candidate set?
    ctx = m.context(PROMPT)
    m.boundary()
    prev = 1
    for pos in range(4):
        p = m.word_step(prev, ctx, pos, source=PROMPT)
        order = np.argsort(-p)[:4]
        top = [(m.vocabulary[int(i)], round(float(p[int(i)]), 3)) for i in order]
        print(f'  pos {pos}: {top}')
        prev = int(order[0])
        if prev == 0:
            break
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
