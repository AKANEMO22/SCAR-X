"""Does the shipped weight set simply end at a better point, or is teaching itself
destructive?

The trace showed each accepted lesson moves the readout and can drop a shipped
answer (the trailing '.' on 'hi', then 'tạm biệt bạn' at step 3). If teaching is
destructive in general, then no amount of guard tuning fixes it: the guard only
prevents losses it can SEE in its probe list.

Test a pristine shipped session against the same sequence with the guard probe
list expanded to include the four shipped answers, so a loss cannot hide.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, LearningGuardRejected  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

WATCH = ['hi', 'chào bạn', 'bạn là ai', 'tạm biệt']
EXPECT = {'hi': 'Hello.', 'chào bạn': 'Chào bạn.',
          'bạn là ai': 'Tôi là scar-x.', 'tạm biệt': 'Tạm biệt bạn.'}
SEQ = [('bạn khỏe không', 'Tôi đang hoạt động.'),
       ('tôi muốn dạy bạn', 'Bạn có thể dạy tôi.'),
       ('chào', 'Chào bạn.'),
       ('cảm ơn', 'Rất vui được giúp bạn.')]


def ok(model):
    c = UnifiedCore.loads(model.dumps())
    c.core.reset_state()
    c.last_active[:] = c.ticks
    return {p: c.generate(p)['text'] == EXPECT[p] for p in WATCH}


d = Path(tempfile.mkdtemp(prefix='scarx-guard-')) / 's'
app = Generation(d, background=False)
try:
    # Force the guard to protect all four shipped answers by injecting them as
    # learned lessons is not appropriate; instead record what the guard sees.
    print('shipped correctness:', ok(app.model))
    pairs = app._guard_pairs('__probe__')
    print(f'guard pairs observable: {len(pairs)}')
    names = {p for p, _ in pairs}
    for p in WATCH:
        print(f"  {p!r:12s} in guard list: {p in names}")

    print()
    for prompt, answer in SEQ:
        try:
            app.teach({'prompt': prompt, 'answer': answer,
                       'repeats': 16, 'confirmation': True})
            outcome = 'ACCEPT'
        except LearningGuardRejected:
            outcome = 'reject'
        state = ok(app.model)
        lost = [p for p, good in state.items() if not good]
        print(f'{outcome:7s} {prompt!r:20s} shipped_answers_ok={state} lost={lost}')
finally:
    app.close()
    shutil.rmtree(d.parent, ignore_errors=True)
