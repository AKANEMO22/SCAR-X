"""generate() never resets recurrent state, so repeated measurement drifts.

ADR0006 says the runtime never calls reset_state: the core keeps one continuous
recurrent state on purpose. That is fine for a conversation, but it means two
consecutive evaluate calls on the SAME weights can disagree, because the second
call starts from the state the first one left.

Consequences measured:
  * an acceptance suite that measures many prompts in a row is order-dependent
  * a just-taught answer can look correct on the first read and wrong on the next

This script quantifies the drift and checks whether scoring each prompt from a
fresh state (clone + reset, exactly what the guard already does) is stable.
No product behaviour is changed here; this is a measurement diagnostic.
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
from scarx.evaluation.product_acceptance import CASES, copy_session  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

PROMPT = 'bạn là ai'


def live_text(model, prompt):
    return model.generate(prompt)['text']


def reset_text(model, prompt):
    c = UnifiedCore.loads(model.dumps())
    c.core.reset_state()
    c.last_active[:] = c.ticks
    return c.generate(prompt)['text']


w = copy_session(ROOT / 'local' / 'generation1')
app = Generation(w, background=False)
try:
    print('=== live state, repeated calls on the same prompt ===')
    seq = [live_text(app.model, PROMPT) for _ in range(8)]
    for i, t in enumerate(seq, 1):
        print(f'  call {i}: {t!r}')
    print(f'  distinct outputs: {len(set(seq))}')

    print('\n=== reset state before each call ===')
    seq2 = [reset_text(app.model, PROMPT) for _ in range(8)]
    for i, t in enumerate(seq2, 1):
        print(f'  call {i}: {t!r}')
    print(f'  distinct outputs: {len(set(seq2))}')

    print('\n=== order dependence across prompts (live state) ===')
    for order_label, order in (('forward', list(CASES)), ('reversed', list(reversed(CASES)))):
        app.model.boundary()
        texts = {}
        for case in order:
            texts[case[0]] = live_text(app.model, case[0])
        print(f'  {order_label}:')
        for k in [c[0] for c in CASES]:
            print(f'    {k:14s} {texts[k]!r}')
    print('\nIf the two orders disagree for a prompt, the live-state measurement')
    print('is order-dependent and must not be reported as a fixed product property.')
finally:
    app.close()
    shutil.rmtree(w.parent, ignore_errors=True)
