"""The guard has the pair in its list yet still accepts a lesson that breaks it.

Observed: all four shipped prompts are present in _guard_pairs(), yet teaching
'bạn khỏe không' was ACCEPTED while 'hi' and 'tạm biệt' regressed.

The scoring uses ONE state convention (reset clone) for both before and after.
If a pair fails BOTH before and after it is never counted as lost:
    guard_lost = sum(before and not after)
A pair that was ALREADY failing in the reset clone cannot be lost, so it cannot
block anything - even though the product answers it correctly in live state.

This prints the before/after verdicts per pair to prove it.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

EXPECT = {'hi': 'Hello.', 'chào bạn': 'Chào bạn.',
          'bạn là ai': 'Tôi là scar-x.', 'tạm biệt': 'Tạm biệt bạn.'}
LESSON = ('bạn khỏe không', 'Tôi đang hoạt động.')


def score(model, prompt, target):
    c = UnifiedCore.loads(model.dumps())
    c.core.reset_state()
    c.last_active[:] = c.ticks
    return c.generate(prompt)['text']


def live(model, prompt):
    return model.generate(prompt)['text']


d = Path(tempfile.mkdtemp(prefix='scarx-gap-')) / 's'
app = Generation(d, background=False)
try:
    pairs = app._guard_pairs(LESSON[0])
    resets_before = []
    print('BEFORE the lesson')
    print(f"{'prompt':14s} {'expected':22s} {'guard(reset)':24s} {'product(live)':24s} guard_pass")
    for prompt, target in pairs:
        r = score(app.model, prompt, target)
        l = live(app.model, prompt)
        passed = r.strip(' .!?').lower() == str(target).strip(' .!?').lower()
        resets_before.append(passed)
        if prompt in EXPECT:
            print(f"{prompt!r:14s} {target[:20]!r:22s} {r[:22]!r:24s} {l[:22]!r:24s} {passed}")

    print('\n=> a pair that is False here can never be counted as "lost",')
    print('   because the guard needs before=True and after=False.')

    try:
        app.teach({'prompt': LESSON[0], 'answer': LESSON[1],
                   'repeats': 16, 'confirmation': True})
        print(f'\nteach ACCEPTED')
    except Exception as exc:
        print(f'\nteach {type(exc).__name__}')

    print('\nAFTER the lesson')
    for prompt, target in pairs:
        if prompt not in EXPECT:
            continue
        r = score(app.model, prompt, target)
        l = live(app.model, prompt)
        passed = r.strip(' .!?').lower() == str(target).strip(' .!?').lower()
        print(f"{prompt!r:14s} guard(reset)={r[:20]!r:22s} pass={passed}  live={l[:20]!r}")
finally:
    app.close()
    shutil.rmtree(d.parent, ignore_errors=True)
