"""Is the accepted small-step lesson stable, and does it actually teach anything?

Observed inconsistency: 'bạn là ai' with repeats=1 was ACCEPTED in one sequence and
REJECTED when run standalone. Two possible causes:
  * order dependence: earlier calls moved the guard baseline
  * genuine flakiness in the before/after comparison

This replays the exact sequence and then the standalone case, and records what an
accepted lesson actually changed in the weights.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, LearningGuardRejected  # noqa: E402
from scarx.evaluation.product_acceptance import copy_session, measure, weights_sha256  # noqa: E402

LESSON = ('bạn là ai', 'Tôi là SCAR-X.')


def attempt(label, work, steps):
    app = Generation(work, background=False)
    try:
        sha0 = weights_sha256(app.model)
        try:
            r = app.teach({'prompt': LESSON[0], 'answer': LESSON[1],
                           'repeats': steps, 'confirmation': True})
            accepted, metrics = True, r['learning_metrics']
            err = None
        except LearningGuardRejected as exc:
            accepted, metrics = False, getattr(exc, 'metrics', {}) or {}
            err = 'guard'
        sha1 = weights_sha256(app.model)
        out = next(x for x in measure(app) if x['prompt'] == LESSON[0])
        print(f'  {label:22s} steps={steps} accepted={accepted}'
              + (f'  err={err}' if err else '')
              + f'  out={out["text"][:26]!r} correct={out["correct"]}')
        if accepted:
            print(f'      changed_synapses={metrics.get("changed_synapses")}')
            print(f'      weight_delta={ {k: round(v, 6) for k, v in (metrics.get("weight_delta") or {}).items()} }')
            print(f'      loss_mean={metrics.get("loss_mean"):.4f}  weights_changed={sha0 != sha1}')
        return accepted, out['correct']
    finally:
        app.close()


print('=== standalone, fresh copy each time ===')
for steps in (1, 2):
    for trial in range(1, 4):
        w = copy_session(ROOT / 'local' / 'generation1')
        try:
            attempt(f'standalone t{trial}', w, steps)
        finally:
            shutil.rmtree(w.parent, ignore_errors=True)

print('\n=== after a prior rejected attempt in the SAME session ===')
w = copy_session(ROOT / 'local' / 'generation1')
try:
    app = Generation(w, background=False)
    try:
        try:
            app.teach({'prompt': 'tạm biệt', 'answer': 'Tạm biệt bạn.',
                       'repeats': 1, 'confirmation': True})
            print('  prior lesson: ACCEPTED')
        except LearningGuardRejected:
            print('  prior lesson: rejected (expected)')
    finally:
        app.close()
    attempt('after prior reject', w, 1)
finally:
    shutil.rmtree(w.parent, ignore_errors=True)
