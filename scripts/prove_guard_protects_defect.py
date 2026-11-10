"""After restore the guard blocks all three remaining fixes. Why?

'restore' gives 5/8. The three remaining prompts are NEARLY right:
  'hi'         -> 'Hello'                   (missing period)
  'cảm ơn'     -> 'Rất vui được giúp bạn.'  (missing period)
  'bạn tên gì' -> 'Tôi là'                  (truncated)

All three lessons were rejected with guard_lost=1 or 2. Because normalized()
strips trailing punctuation, 'Hello' and 'Hello.' compare EQUAL, so the guard
already considers these pairs passing and protects the truncated form.

That means: the guard is not blocking a fix, it is preserving a defect. Verify by
printing the guard's own verdict for each of these prompts before and after.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, normalized  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

GOOD = 'b305479afb0e40c1ba5a04acd0139658'
LIVE = ROOT / 'local' / 'generation1'
WATCH = ['hi', 'cảm ơn', 'bạn tên gì']


def guard_view(model, prompt, target):
    c = UnifiedCore.loads(model.dumps())
    c.core.reset_state()
    c.last_active[:] = c.ticks
    out = c.generate(prompt)['text']
    same = normalized(out) == normalized(target)
    return out, same


work = Path(tempfile.mkdtemp(prefix='scarx-guard2-')) / 'session'
shutil.copytree(LIVE, work, ignore=shutil.ignore_patterns('session.lock'))
app = Generation(work, background=False)
try:
    app.restore(GOOD)
    pairs = {p: t for p, t in app._guard_pairs('__probe__')}
    print('After restore — does the guard consider these already correct?\n')
    print(f"{'prompt':14s} {'stored target':24s} {'model output':26s} guard says")
    for p in WATCH:
        t = pairs.get(p)
        out, same = guard_view(app.model, p, t) if t is not None else (None, None)
        print(f'{p!r:14s} {str(t)[:22]!r:24s} {str(out)[:24]!r:26s} '
              f'{"PASS(equal)" if same else "fail"}')

    print()
    print('normalized() strips trailing punctuation, so a truncated reply that')
    print('matches on words is treated as already correct. Two consequences:')
    print('  1. the guard protects the truncated form and rejects any fix')
    print('  2. the product can never improve these three while the guard is on')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
