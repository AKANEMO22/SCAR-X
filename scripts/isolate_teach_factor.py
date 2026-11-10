"""Why did a bare train_pair fix the answer in round 11 but teach() does not?

Round 11 showed: model.train_pair(prompt, target) once, then a reset-clone read
gave 'tôi là scar-x .' (correct).
This round shows: app.teach(repeats=1) gives 'tôi là scar-x là' (wrong).

The two differ in call order. teach() runs generate() before training to capture
'before' text; that advances the live state, and expand_vocabulary() runs before
training too. This isolates each factor on the same copy:

  A. bare train_pair                                   (round 11 recipe)
  B. expand_vocabulary + train_pair
  C. generate before + expand + train_pair             (teach order)
  D. same as C but read from a reset clone vs live

Whichever step flips the result names the real cause.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np  # noqa: E402
from scarx.app.generation import Generation  # noqa: E402
from scarx.evaluation.product_acceptance import copy_session  # noqa: E402
from scarx.neural.language import tokenize  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

PROMPT, TARGET = 'bạn là ai', 'Tôi là SCAR-X.'


def reset_read(model):
    c = UnifiedCore.loads(model.dumps())
    c.core.reset_state()
    c.last_active[:] = c.ticks
    return c.generate(PROMPT)['text']


def live_read(model):
    return model.generate(PROMPT)['text']


def run(label, steps):
    w = copy_session(ROOT / 'local' / 'generation1')
    app = Generation(w, background=False)
    try:
        m = app.model
        for step in steps:
            if step == 'generate':
                m.generate(PROMPT)
            elif step == 'expand':
                m.expand_vocabulary(TARGET)
            elif step == 'train':
                m.train_pair(PROMPT, TARGET)
        print(f'  {label:34s} reset={reset_read(m)[:22]!r:24s} live={live_read(m)[:22]!r}')
    finally:
        app.close()
        shutil.rmtree(w.parent, ignore_errors=True)


print('=== isolate the factor ===')
run('A. train only', ['expand', 'train'])
run('B. expand + train', ['expand', 'train'])
run('C. generate, expand, train', ['generate', 'expand', 'train'])
run('D. train (no expand)', ['train'])

print('\n=== repeat the bare recipe to check it is not luck ===')
for i in range(3):
    run(f'A{i+1}. train only #{i+1}', ['expand', 'train'])

print('\n=== does a SECOND train_pair help or hurt? ===')
run('E. train x2', ['expand', 'train', 'train'])
run('F. train x4', ['expand', 'train', 'train', 'train', 'train'])
