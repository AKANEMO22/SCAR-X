"""Does bounded product teaching actually converge, and does the guard block it?

Read-only w.r.t. the live session: every trial runs on a discarded copy.
Answers two separate questions that were previously conflated:
  1. Can the bounded teaching budget make the model emit the target at all?
  2. If it can, does the guard throw the update away?
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

from scarx.app.generation import Generation, normalized  # noqa: E402

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'local' / 'generation1'

TRIALS = [
    ('chào bạn', 'Chào bạn.'),
    ('chào', 'Chào bạn.'),
    ('cảm ơn', 'Không có gì.'),
    ('tạm biệt', 'Tạm biệt bạn.'),
    ('bạn là ai', 'Tôi là SCAR-X.'),
]


def run(prompt, target, repeats):
    work = Path(tempfile.mkdtemp(prefix='scarx-teach-')) / 'session'
    shutil.copytree(SOURCE, work, ignore=shutil.ignore_patterns('session.lock'))
    app = Generation(work, background=False)
    try:
        before = normalized(app.model.generate(prompt)['text'])
        guard = app._guard_pairs(prompt)
        g_before = app._guard_scores(app.model, guard) if guard else []
        losses = []
        try:
            # Mirror the real path: teaching may extend the vocabulary first.
            app.model.expand_vocabulary(target)
            for _ in range(repeats):
                losses.append(float(app.model.train_pair(prompt, target)))
        except ValueError as exc:
            return {'prompt': prompt, 'target': target, 'repeats': repeats,
                    'error': str(exc), 'before': before, 'after': before,
                    'output_matches_target': False, 'guard_lost': 0,
                    'guard_gained': 0, 'guard_pairs': len(guard), 'would_be_blocked': False,
                    'loss_first': None, 'loss_last': None}
        after = normalized(app.model.generate(prompt)['text'])
        g_after = app._guard_scores(app.model, guard) if guard else []
        lost = sum(b and not a for b, a in zip(g_before, g_after))
        gained = sum((not b) and a for b, a in zip(g_before, g_after))
        # Would the real (non-automatic) path accept this?
        blocked = bool(lost)
        return {
            'prompt': prompt, 'target': target, 'repeats': repeats,
            'before': before, 'after': after,
            'output_matches_target': normalized(after) == normalized(target),
            'loss_first': losses[0] if losses else None,
            'loss_last': losses[-1] if losses else None,
            'guard_lost': lost, 'guard_gained': gained, 'guard_pairs': len(guard),
            'would_be_blocked': blocked,
        }
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


print(f'session copy: {SOURCE}')
print(f"vocab/updates: reading via first trial\n")
rows = []
for prompt, target in TRIALS:
    for repeats in (4, 16, 32):
        r = run(prompt, target, repeats)
        rows.append(r)
        flag = 'OK  ' if r['output_matches_target'] else ('BLK ' if r['would_be_blocked'] else 'FAIL')
        print(f"{flag} {prompt!r:14s} x{repeats:<3d} {r['before']!r:22s} -> {r['after']!r:26s} "
              f"loss {r['loss_first']:.2f}->{r['loss_last']:.2f} lost={r['guard_lost']} gained={r['guard_gained']}/{r['guard_pairs']}")

ok = [r for r in rows if r['output_matches_target']]
blk = [r for r in rows if r['would_be_blocked']]
print(f"\nconverged (emits target) : {len(ok)}/{len(rows)}")
print(f"guard would block        : {len(blk)}/{len(rows)}")
print(f"converged AND blocked    : {len([r for r in ok if r['would_be_blocked']])}/{len(rows)}")
json.dump(rows, open(ROOT / 'local' / 'teach-convergence.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2, default=str)
print('\nwrote local/teach-convergence.json')
