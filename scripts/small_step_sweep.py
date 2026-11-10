"""Adaptive-step search at the SMALL end (1,2,3), which was never tested.

The user authorised "thử các mức cập nhật nhỏ hơn thay vì luôn lặp một số bước cố
định". An earlier sweep only tried 4/8/16 and concluded no step works. That
conclusion was drawn from an untested assumption: that smaller means smaller than
4. The idle path itself uses repeats=1, so 1..3 are the natural candidates and
were missing from the evidence.

No rule, topology, signal or guard threshold changes. Copies only.
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

LESSONS = [
    ('bạn là ai', 'Tôi là SCAR-X.'),
    ('bạn tên gì', 'Tôi là SCAR-X.'),
    ('chào bạn', 'Chào bạn.'),
    ('cảm ơn', 'Không có gì.'),
    ('tạm biệt', 'Tạm biệt bạn.'),
    ('bạn khỏe không', 'Tôi đang hoạt động.'),
]
STEPS = (1, 2, 3, 4, 6)


def attempt(prompt, answer, steps):
    work = copy_session(ROOT / 'local' / 'generation1')
    app = Generation(work, background=False)
    try:
        sha0 = weights_sha256(app.model)
        rows0 = measure(app, CASES)
        hits0 = {r['prompt'] for r in rows0 if r['correct']}
        try:
            app.teach({'prompt': prompt, 'answer': answer,
                       'repeats': steps, 'confirmation': True})
            accepted, lost = True, None
        except LearningGuardRejected as exc:
            accepted = False
            lost = getattr(exc, 'metrics', {}).get('guard_lost')
        except Exception as exc:
            return {'prompt': prompt, 'steps': steps, 'accepted': False,
                    'error': type(exc).__name__, 'emits_target': False,
                    'newly_lost': [], 'rolled_back': weights_sha256(app.model) == sha0}
        rows1 = measure(app, CASES)
        hits1 = {r['prompt'] for r in rows1 if r['correct']}
        return {'prompt': prompt, 'steps': steps, 'accepted': accepted,
                'guard_lost': lost,
                'emits_target': any(r['prompt'] == prompt and r['correct'] for r in rows1),
                'newly_lost': sorted(hits0 - hits1),
                'newly_gained': sorted(hits1 - hits0),
                'rolled_back': (not accepted) and weights_sha256(app.model) == sha0}
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


print(f"{'prompt':20s} {'steps':5s} {'accept':7s} {'gLost':5s} {'emits':6s} {'newly_lost':12s}")
rows = []
for prompt, answer in LESSONS:
    for steps in STEPS:
        r = attempt(prompt, answer, steps)
        rows.append(r)
        print(f"{prompt[:18]!r:20s} {steps:<5d} "
              f"{'ACCEPT' if r['accepted'] else 'reject':7s} "
              f"{str(r.get('guard_lost')):5s} {str(r['emits_target']):6s} "
              f"{str(r['newly_lost']):12s}")

acc = [r for r in rows if r['accepted']]
print(f'\nacceptance: {len(acc)}/{len(rows)} = {len(acc)/len(rows):.3f}')
print(f'accepted but lost something: {len([r for r in acc if r["newly_lost"]])}')
print()
print('smallest step that is accepted, per lesson:')
for prompt, _ in LESSONS:
    sub = sorted([r for r in rows if r['prompt'] == prompt], key=lambda r: (not r['accepted'], r['steps']))
    best = sub[0]
    print(f"  {prompt!r:20s} {'ACCEPT steps='+str(best['steps']) if best['accepted'] else 'blocked at every step'}")

out = ROOT / 'local' / 'small-step-sweep.json'
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
print(f'\nwrote {out}')
