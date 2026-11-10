"""Why does teach() report no change when a bare train_pair fixes the answer?

Two paths disagree:
  * model.train_pair(prompt, target) once  -> output becomes correct
  * app.teach({... repeats:1})             -> output unchanged

Candidate cause: teach() calls generate() BEFORE training (to capture the
'before' text) and again AFTER, and generate() advances the recurrent state.
So the 'after' measurement is taken from a different working state than the
one right after the weight update.

A correct product should judge the taught pair from a reproducible state, not
from whatever state the previous generate left behind. This measures both.
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
from scarx.evaluation.product_acceptance import copy_session, measure  # noqa: E402
from scarx.neural.language import tokenize  # noqa: E402

PROMPT, TARGET = 'bạn là ai', 'Tôi là SCAR-X.'


def trace(model, prompt, n=6):
    ctx = model.context(prompt)
    model.boundary()
    prev, out = 1, []
    for pos in range(n):
        p = model.word_step(prev, ctx, pos, source=prompt)
        tok = int(np.argmax(p))
        out.append(model.vocabulary[tok])
        if tok == 0:
            break
        prev = tok
    return out


def fresh():
    w = copy_session(ROOT / 'local' / 'generation1')
    return w, Generation(w, background=False)


print('=== PATH 1: bare train_pair ===')
w, app = fresh()
try:
    print('  before      :', trace(app.model, PROMPT))
    app.model.expand_vocabulary(TARGET)
    app.model.train_pair(PROMPT, TARGET)
    print('  after       :', trace(app.model, PROMPT))
    print('  measure()   :', next(r['text'] for r in measure(app) if r['prompt'] == PROMPT))
finally:
    app.close()
    shutil.rmtree(w.parent, ignore_errors=True)

print('\n=== PATH 2: app.teach (repeats=1) ===')
w, app = fresh()
try:
    print('  before      :', trace(app.model, PROMPT))
    r = app.teach({'prompt': PROMPT, 'answer': TARGET, 'repeats': 1, 'confirmation': True})
    m = r['learning_metrics']
    print('  teach before:', repr(m['before']))
    print('  teach after :', repr(m['after_attempt']))
    print('  trace after :', trace(app.model, PROMPT))
    print('  measure()   :', next(x['text'] for x in measure(app) if x['prompt'] == PROMPT))
finally:
    app.close()
    shutil.rmtree(w.parent, ignore_errors=True)

print('\n=== PATH 3: teach, then trace FIRST (no extra generate before) ===')
w, app = fresh()
try:
    app.teach({'prompt': PROMPT, 'answer': TARGET, 'repeats': 1, 'confirmation': True})
    print('  trace       :', trace(app.model, PROMPT))
    out = app.model.generate(PROMPT)
    print('  generate    :', repr(out['text']))
    out2 = app.model.generate(PROMPT)
    print('  generate #2 :', repr(out2['text']))
    out3 = app.model.generate(PROMPT)
    print('  generate #3 :', repr(out3['text']))
finally:
    app.close()
    shutil.rmtree(w.parent, ignore_errors=True)
