"""Rollback restores weights exactly yet the answer changes.

Observed:
  after an ACCEPTED lesson : 'hi' -> 'Hello.'
  after a REJECTED lesson  : 'hi' -> 'Hello'    (updates unchanged, weights restored)

If weights are identical and only the recurrent state differs, then the readout is
state-sensitive enough that a failed attempt still changes behaviour. That means
the guard restores the model but not the user-visible behaviour, and a rejected
lesson can still degrade the product.

Verify by comparing weights, ticks, voltage and the per-prompt texts across the
reject boundary.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np  # noqa: E402
from scarx.app.generation import Generation, LearningGuardRejected  # noqa: E402
from scarx.evaluation.product_acceptance import weights_sha256  # noqa: E402

PROMPTS = ['hi', 'chào bạn', 'bạn là ai', 'tạm biệt']


def snap(app):
    return {
        'weights': weights_sha256(app.model),
        'updates': app.model.core.updates,
        'ticks': app.model.ticks,
        'voltage_sum': float(np.abs(app.model.core.voltage).sum()),
        'adaptation_sum': float(np.abs(app.model.core.adaptation).sum()),
        'texts': {p: app.model.generate(p)['text'] for p in PROMPTS},
    }


d = Path(tempfile.mkdtemp(prefix='scarx-rb-')) / 's'
app = Generation(d, background=False)
try:
    # An accepted lesson first, so the reject is not the very first operation.
    app.teach({'prompt': 'bạn khỏe không', 'answer': 'Tôi đang hoạt động.',
               'repeats': 16, 'confirmation': True})
    before = snap(app)
    print('BEFORE the rejected attempt')
    print(f"  updates={before['updates']} ticks={before['ticks']}")
    print(f"  weights={before['weights'][:16]}")
    for p, t in before['texts'].items():
        print(f"    {p!r:12s} -> {t!r}")

    rejected = False
    try:
        app.teach({'prompt': 'bạn làm được gì', 'answer': 'Tôi đang học trả lời.',
                   'repeats': 16, 'confirmation': True})
    except LearningGuardRejected:
        rejected = True

    after = snap(app)
    print(f"\nrejected={rejected}")
    print('AFTER the rejected attempt')
    print(f"  updates={after['updates']} ticks={after['ticks']}")
    print(f"  weights={after['weights'][:16]}")
    for p, t in after['texts'].items():
        print(f"    {p!r:12s} -> {t!r}")

    print('\nDIFF')
    print(f"  weights identical : {before['weights'] == after['weights']}")
    print(f"  updates identical : {before['updates'] == after['updates']}")
    print(f"  ticks identical   : {before['ticks'] == after['ticks']}")
    print(f"  voltage identical : {before['voltage_sum'] == after['voltage_sum']}")
    print(f"  texts identical   : {before['texts'] == after['texts']}")
    changed = {p: (before['texts'][p], after['texts'][p])
               for p in PROMPTS if before['texts'][p] != after['texts'][p]}
    if changed:
        print('\nCHANGED TEXTS DESPITE IDENTICAL WEIGHTS')
        for p, (b, a) in changed.items():
            print(f"    {p!r:12s} {b!r} -> {a!r}")
        print('\n=> A rejected attempt altered user-visible behaviour while the')
        print('   guard reported a clean rollback. Rollback restores weights but')
        print('   not the recurrent state that produced the answer.')
finally:
    app.close()
    shutil.rmtree(d.parent, ignore_errors=True)
