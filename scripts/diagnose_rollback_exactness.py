"""Does rollback restore the EXACT previous state, or only the weights?

The retention proof flagged 'rollback_incomplete' for every rejected lesson while
lost_correct and lost_guard were both empty. That means the post-rollback snapshot
differed from the pre-lesson snapshot in some way that is not a lost skill.

Two possibilities:
  A. real defect: rollback leaves the model in a different state than before
  B. expected: weights are restored exactly, but the recurrent working state and
     tick counter keep advancing because ADR0006 forbids a reset

Distinguish by comparing weights_sha256, the tick counter, and the measured sets
separately across a rejected attempt.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, LearningGuardRejected  # noqa: E402
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, copy_session, guard_report, measure, weights_sha256,
)


def snap(app):
    return {
        'weights': weights_sha256(app.model),
        'ticks': app.model.ticks,
        'updates': app.model.core.updates,
        'correct': sorted(r['prompt'] for r in measure(app, CASES, reset_state=True) if r['correct']),
        'guard': sorted(g['prompt'] for g in guard_report(app, '__rb__') if g['pass']),
        'vocab': len(app.model.vocabulary),
    }


work = copy_session(ROOT / 'local' / 'generation1')
app = Generation(work, background=False)
try:
    before = snap(app)
    print('BEFORE')
    for k, v in before.items():
        print(f'  {k:8s}: {v}')

    rejected = False
    try:
        app.teach({'prompt': 'chào bạn', 'answer': 'Chào bạn.',
                   'repeats': 1, 'confirmation': True})
        print('\nteach ACCEPTED (expected reject for this probe)')
    except LearningGuardRejected:
        rejected = True
        print('\nteach REJECTED as expected')

    after = snap(app)
    print('AFTER')
    for k, v in after.items():
        print(f'  {k:8s}: {v}')

    print('\nDIFF')
    for k in before:
        same = before[k] == after[k]
        print(f'  {k:8s}: {"identical" if same else "CHANGED"}')

    print()
    if rejected and before['weights'] == after['weights']:
        print('VERDICT: rollback restored weights exactly.')
        if before['ticks'] != after['ticks']:
            print('         ticks advanced during the attempt, as ADR0006 requires.')
        if before['correct'] == after['correct'] and before['guard'] == after['guard']:
            print('         measured skills are identical, so no skill was lost.')
            print('         The earlier flag compared snapshots that include tick drift;')
            print('         that is not a retention violation.')
    elif rejected:
        print('VERDICT: REAL DEFECT - weights differ after a rejected attempt.')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
