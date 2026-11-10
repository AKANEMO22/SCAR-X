"""Can more teaching beat state noise? Measure from BOTH states.

Round 11 established two facts:
  * train_pair genuinely fixes the answer ('tôi là scar-x là scar-x' -> correct)
  * generate() keeps one continuous state, so the same weight can read differently
    depending on what ran before it

The open question is practical: does teaching the same pair MANY times reinforce
it enough to survive both measurements? That is ordinary interactive teaching
(explicitly authorised), uses no new rule, and needs no protocol.

Measured per repeat count:
  * answer scored from a reset clone  (reproducible)
  * answer scored from live state     (what a user sees next turn)
  * whether the guard accepted
  * what was lost, if anything

Read-only on copies.
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
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, copy_session, measure, weights_sha256,
)

LESSONS = [('bạn là ai', 'Tôi là SCAR-X.'), ('bạn tên gì', 'Tôi là SCAR-X.')]
REPEATS = (1, 4, 8, 16, 32)


def trial(prompt, answer, repeats):
    work = copy_session(ROOT / 'local' / 'generation1')
    app = Generation(work, background=False)
    try:
        sha0 = weights_sha256(app.model)
        before_reset = {r['prompt']: r for r in measure(app, CASES, reset_state=True)}
        before_live = {r['prompt']: r for r in measure(app, CASES, reset_state=False)}
        accepted, guard_lost = False, None
        try:
            app.teach({'prompt': prompt, 'answer': answer,
                       'repeats': repeats, 'confirmation': True})
            accepted = True
        except LearningGuardRejected as exc:
            guard_lost = getattr(exc, 'metrics', {}).get('guard_lost')
            return {'repeats': repeats, 'accepted': False, 'guard_lost': guard_lost,
                    'reset_correct': before_reset[prompt]['correct'],
                    'live_correct': before_live[prompt]['correct'],
                    'reset_text': before_reset[prompt]['text'],
                    'live_text': before_live[prompt]['text'],
                    'newly_lost': [], 'rolled_back': weights_sha256(app.model) == sha0}
        after_reset = {r['prompt']: r for r in measure(app, CASES, reset_state=True)}
        after_live = {r['prompt']: r for r in measure(app, CASES, reset_state=False)}
        lost = sorted({p for p, r in before_reset.items() if r['correct'] and not after_reset[p]['correct']}
                      | {p for p, r in before_live.items() if r['correct'] and not after_live[p]['correct']})
        return {'repeats': repeats, 'accepted': True, 'guard_lost': None,
                'reset_correct': after_reset[prompt]['correct'],
                'live_correct': after_live[prompt]['correct'],
                'reset_text': after_reset[prompt]['text'],
                'live_text': after_live[prompt]['text'],
                'newly_lost': lost, 'rolled_back': False}
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


rows = []
for prompt, answer in LESSONS:
    print(f'\n=== {prompt!r} -> {answer!r} ===')
    print(f"{'repeats':8s} {'accept':7s} {'reset':7s} {'live':6s} {'newly_lost':12s} reset_text / live_text")
    for repeats in REPEATS:
        r = trial(prompt, answer, repeats)
        rows.append({'prompt': prompt, **r})
        print(f"{repeats:<8d} {'ACCEPT' if r['accepted'] else 'reject':7s} "
              f"{str(r['reset_correct']):7s} {str(r['live_correct']):6s} "
              f"{str(r['newly_lost']):12s} {r['reset_text'][:16]!r} / {r['live_text'][:16]!r}")

acc = [r for r in rows if r['accepted']]
teachable = [r for r in acc if r['reset_correct'] and r['live_correct']]
print()
print(f'attempts                     : {len(rows)}')
print(f'accepted                     : {len(acc)}')
print(f'accepted AND correct in both : {len(teachable)}')
print(f'accepted with a regression   : {len([r for r in acc if r["newly_lost"]])}')

out = ROOT / 'local' / 'teaching-strength.json'
out.write_text(json.dumps({'rows': rows, 'scope': 'Explicit teaching only; no rule change.'},
                          ensure_ascii=False, indent=2, default=str), encoding='utf-8')
print(f'\nwrote {out}')
