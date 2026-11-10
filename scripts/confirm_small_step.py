"""Confirm the small-step result is reproducible, and show the full before/after.

The sweep found 'bạn là ai' accepted at repeats=1,2 but rejected at 3+. That is a
sharp boundary, so it must be checked for reproducibility before being reported
as a finding. Runs the same trial several times and shows the resulting product
outputs, so the gain is visible rather than inferred.
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
    CASES, copy_session, measure, weights_sha256,
)

PROMPT, ANSWER = 'bạn là ai', 'Tôi là SCAR-X.'
TRIALS = 5


def once(steps):
    work = copy_session(ROOT / 'local' / 'generation1')
    app = Generation(work, background=False)
    try:
        before = {r['prompt']: r['text'] for r in measure(app, CASES)}
        sha0 = weights_sha256(app.model)
        try:
            app.teach({'prompt': PROMPT, 'answer': ANSWER,
                       'repeats': steps, 'confirmation': True})
            accepted, lost = True, None
        except LearningGuardRejected as exc:
            accepted, lost = False, getattr(exc, 'metrics', {}).get('guard_lost')
        after_rows = measure(app, CASES)
        after = {r['prompt']: r['text'] for r in after_rows}
        sha1 = weights_sha256(app.model)
        return {'steps': steps, 'accepted': accepted, 'guard_lost': lost,
                'before': before.get(PROMPT), 'after': after.get(PROMPT),
                'emits_target': any(r['prompt'] == PROMPT and r['correct'] for r in after_rows),
                'sha_changed': sha0 != sha1,
                'regressed': [p for p, r in ((r['prompt'], r) for r in after_rows)
                              if not r['correct'] and before.get(p, '').strip()
                              and p != PROMPT]}
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


print(f'reproducibility for {PROMPT!r} -> {ANSWER!r}')
print(f"{'steps':6s} {'trial':6s} {'accept':7s} {'before':22s} {'after':24s} emits")
for steps in (1, 2, 3):
    for trial in range(1, TRIALS + 1):
        r = once(steps)
        print(f"{steps:<6d} {trial:<6d} {'ACCEPT' if r['accepted'] else 'reject':7s} "
              f"{str(r['before'])[:20]!r:22s} {str(r['after'])[:22]!r:24s} {r['emits_target']}")
    print()
