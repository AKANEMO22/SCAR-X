"""Guard acceptance rate: which lessons CAN be learned without losing anything?

The user's priority #1 is "learn new without forgetting old". With the guard now
running on the manual teaching path, this measures the honest acceptance rate and
the guard's regression count, then checks whether an ADAPTIVE STEP (fewer repeats)
lets a blocked lesson through — which the user explicitly authorised
("Thử các mức cập nhật nhỏ hơn thay vì luôn lặp một số bước cố định").

No learning rule, topology, signal or threshold is changed. Read-only on copies.
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
    CASES, copy_session, measure, summarise, weights_sha256,
)

CANDIDATES = [
    ('bạn tên gì', 'Tôi là SCAR-X.'),
    ('cảm ơn', 'Không có gì.'),
    ('tạm biệt', 'Tạm biệt bạn.'),
    ('bạn là ai', 'Tôi là SCAR-X.'),
    ('chào bạn', 'Chào bạn.'),
    ('chào', 'Chào bạn.'),
    ('bạn khỏe không', 'Tôi đang hoạt động.'),
    ('xin chào', 'Chào bạn.'),
]
STEPS = (4, 8, 16)


def attempt(lesson, steps):
    work = copy_session(ROOT / 'local' / 'generation1')
    app = Generation(work, background=False)
    try:
        base_sha = weights_sha256(app.model)
        rows_before = measure(app, CASES)
        hits_before = {r['prompt'] for r in rows_before if r['correct']}
        try:
            out = app.teach({'prompt': lesson[0], 'answer': lesson[1],
                             'repeats': steps, 'confirmation': True})
            accepted = True
            metrics = out.get('learning_metrics', {})
        except LearningGuardRejected as exc:
            accepted = False
            metrics = getattr(exc, 'metrics', {}) or {}
        except Exception as exc:
            accepted = False
            metrics = {'error': f'{type(exc).__name__}: {exc}'[:120]}
        rows_after = measure(app, CASES)
        hits_after = {r['prompt'] for r in rows_after if r['correct']}
        sha_after = weights_sha256(app.model)
        return {
            'prompt': lesson[0], 'answer': lesson[1], 'steps': steps,
            'accepted': accepted,
            'guard_lost': metrics.get('guard_lost'),
            'loss_mean': metrics.get('loss_mean'),
            'error': metrics.get('error'),
            'newly_lost': sorted(hits_before - hits_after),
            'newly_gained': sorted(hits_after - hits_before),
            'correct_before': len(hits_before), 'correct_after': len(hits_after),
            'weights_rolled_back': (not accepted and sha_after == base_sha),
            'emits_target': any(r['prompt'] == lesson[0] and r['correct'] for r in rows_after),
        }
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


def main():
    print('=== guard acceptance per lesson x adaptive step ===')
    print(f"{'prompt':18s} {'steps':5s} {'accept':7s} {'gLost':5s} {'newLost':9s} {'emits':6s} rollback")
    results = []
    for lesson in CANDIDATES:
        for steps in STEPS:
            r = attempt(lesson, steps)
            results.append(r)
            print(f"{r['prompt'][:16]!r:18s} {steps:<5d} "
                  f"{'ACCEPT' if r['accepted'] else 'reject':7s} "
                  f"{str(r['guard_lost']):5s} {str(r['newly_lost']):9s} "
                  f"{str(r['emits_target']):6s} {r['weights_rolled_back']}")

    total = len(results)
    accepted = [r for r in results if r['accepted']]
    print()
    print(f'acceptance rate      : {len(accepted)}/{total} = {len(accepted)/total:.3f}')
    print(f'accepted w/ new loss : {len([r for r in accepted if r["newly_lost"]])}')
    print(f'rejected w/ rollback : {len([r for r in results if r["weights_rolled_back"]])}/'
          f'{len([r for r in results if not r["accepted"]])}')
    print()
    print('per-lesson best result (smallest step that is accepted):')
    for prompt, _ in CANDIDATES:
        rows = sorted([r for r in results if r['prompt'] == prompt],
                      key=lambda r: (not r['accepted'], r['steps']))
        best = rows[0]
        state = 'ACCEPTED' if best['accepted'] else 'blocked'
        print(f"  {prompt!r:20s} {state:9s} steps={best['steps']:<3d} "
              f"emits={best['emits_target']} newly_lost={best['newly_lost']}")

    out = ROOT / 'local' / 'guard-acceptance.json'
    out.write_text(json.dumps({'results': results,
                               'acceptance_rate': len(accepted) / total,
                               'scope': 'Guard-enabled acceptance on copies; '
                                        'no rule/threshold change.'},
                              ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
