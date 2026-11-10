"""Retention proof: teach repeatedly and verify no protected skill is ever lost.

Criterion #4 of the user's list is "không làm mất các kỹ năng được bảo vệ". Earlier
runs measured this incidentally. This measures it directly and exhaustively:

  * snapshot the full product set BEFORE any teaching
  * teach every candidate in sequence, at several step sizes
  * after EVERY attempt re-measure the full set
  * assert that no prompt which was correct before is wrong after

Also checks the reverse direction: a protected pair that PASSES in the guard must
still pass afterwards, so protection is not merely cosmetic.

Read-only on copies. No rule, topology, signal or threshold change.
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
    CASES, copy_session, guard_report, measure, weights_sha256,
)

CANDIDATES = [
    ('bạn là ai', 'Tôi là SCAR-X.', 1),
    ('bạn là ai', 'Tôi là SCAR-X.', 2),
    ('bạn tên gì', 'Tôi là SCAR-X.', 1),
    ('chào bạn', 'Chào bạn.', 1),
    ('cảm ơn', 'Không có gì.', 1),
    ('tạm biệt', 'Tạm biệt bạn.', 1),
    ('bạn khỏe không', 'Tôi đang hoạt động.', 1),
    ('hi', 'Hello.', 1),
    ('xin chào', 'Chào bạn.', 1),
]


def correct_set(app):
    return {r['prompt'] for r in measure(app, CASES) if r['correct']}


def guard_pass_set(app, changing_prompt='__retention__'):
    """Pass the SAME exclude prompt teach() will use, or the list differs.

    guard_report() drops the prompt being taught from the pair list, so measuring
    with a different sentinel produces a different set and the before/after
    comparison is meaningless. Use the actual lesson prompt.
    """
    return {g['prompt'] for g in guard_report(app, changing_prompt) if g['pass']}


work = copy_session(ROOT / 'local' / 'generation1')
app = Generation(work, background=False)
violations = []
records = []
try:
    base_correct = correct_set(app)
    base_guard = guard_pass_set(app)
    # Per-attempt comparison state, so an accepted lesson does not make the next
    # rejection look like a loss against a stale baseline.
    prev_correct, prev_guard = set(base_correct), set(base_guard)
    print(f'baseline correct : {sorted(base_correct)}')
    print(f'baseline guard   : {sorted(base_guard)}')
    print()

    for prompt, answer, steps in CANDIDATES:
        sha_before = weights_sha256(app.model)
        # Measure the guard set with THIS lesson's exclude prompt, matching teach().
        prev_guard = guard_pass_set(app, prompt)
        prev_correct = correct_set(app)
        accepted = False
        guard_lost = None
        try:
            app.teach({'prompt': prompt, 'answer': answer,
                       'repeats': steps, 'confirmation': True})
            accepted = True
        except LearningGuardRejected as exc:
            guard_lost = getattr(exc, 'metrics', {}).get('guard_lost')

        after_correct = correct_set(app)
        after_guard = guard_pass_set(app, prompt)
        sha_after = weights_sha256(app.model)

        lost_correct = sorted(prev_correct - after_correct)
        lost_guard = sorted(prev_guard - after_guard)
        rolled_back = (not accepted) and sha_after == sha_before

        # A rejected attempt that rolled back restores the previous revision, so it
        # cannot have lost anything. The earlier version of this script flagged
        # every rejection because it compared against the FIRST baseline instead of
        # the state immediately before this attempt, and because ticks advance by
        # design (ADR0006). A separate exactness check confirmed rollback restores
        # weights, updates, vocabulary and both measured sets identically.
        restore_ok = True
        if rolled_back:
            restore_ok = (after_correct == prev_correct) and (after_guard == prev_guard)
            if not restore_ok:
                violations.append({
                    'prompt': prompt, 'steps': steps, 'kind': 'rollback_incomplete',
                    'lost_correct': sorted(prev_correct - after_correct),
                    'lost_guard': sorted(prev_guard - after_guard)})
            lost_correct, lost_guard = [], []

        rec = {'prompt': prompt, 'answer': answer, 'steps': steps,
               'accepted': accepted, 'guard_lost': guard_lost,
               'lost_correct': lost_correct, 'lost_guard': lost_guard,
               'rolled_back': rolled_back}
        records.append(rec)
        # Advance the comparison baseline to the state that remains in effect.
        prev_correct, prev_guard = after_correct, after_guard

        state = 'ACCEPT' if accepted else 'reject'
        print(f"  {state} {prompt!r:20s} x{steps}  "
              f"lost_correct={lost_correct or '-'} lost_guard={lost_guard or '-'}"
              + ('  ROLLED_BACK' if rolled_back else ''))

        # A violation is: something correct before becomes incorrect, OR a passing
        # guard pair stops passing, on the same model revision after the call.
        if lost_correct or lost_guard:
            violations.append(rec)

    print()
    print(f'attempts              : {len(records)}')
    print(f'accepted              : {sum(r["accepted"] for r in records)}')
    print(f'rejected              : {sum(not r["accepted"] for r in records)}')
    print(f'rejections rolled back: {sum(r["rolled_back"] for r in records)}/'
          f'{sum(not r["accepted"] for r in records)}')
    print(f'RETENTION VIOLATIONS  : {len(violations)}')
    if violations:
        for v in violations:
            print(f"   VIOLATION {v['prompt']!r} lost_correct={v['lost_correct']}")
    else:
        print('  => no protected skill was lost in any attempt.')

    out = ROOT / 'local' / 'retention-proof.json'
    out.write_text(json.dumps({
        'baseline_correct': sorted(base_correct),
        'baseline_guard': sorted(base_guard),
        'records': records, 'violations': violations,
        'attempts': len(records),
        'scope': 'Direct retention check per attempt on a discarded copy.'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nwrote {out}')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
