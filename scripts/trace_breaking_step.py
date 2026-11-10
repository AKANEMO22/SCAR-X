"""Which training event actually broke the shipped answers?

Shipped weights (241123 updates) answer correctly; the live session (241363) does
not. Only 240 updates apart. This replays teaching from the shipped weights and
records, after every step, both the weights hash and the four shipped answers, so
the exact event that flips them is identified rather than guessed.

Measured from a FRESH state each time as well, to separate "weights got worse"
from "state drifted".
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

from scarx.app.generation import Generation, LearningGuardRejected  # noqa: E402
from scarx.evaluation.product_acceptance import weights_sha256  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

WATCH = ['hi', 'chào bạn', 'bạn là ai', 'tạm biệt']
SEQ = [
    ('bạn khỏe không', 'Tôi đang hoạt động.'),
    ('bạn làm được gì', 'Tôi đang học trả lời.'),
    ('tôi muốn dạy bạn', 'Bạn có thể dạy tôi.'),
    ('chào', 'Chào bạn.'),
    ('cảm ơn', 'Rất vui được giúp bạn.'),
    ('7 + 8', '15'),
]


def read_live(model):
    return {p: model.generate(p)['text'] for p in WATCH}


def read_reset(model):
    c = UnifiedCore.loads(model.dumps())
    c.core.reset_state()
    c.last_active[:] = c.ticks
    return {p: c.generate(p)['text'] for p in WATCH}


d = Path(tempfile.mkdtemp(prefix='scarx-seq-')) / 's'
app = Generation(d, background=False)
try:
    print(f"shipped updates={app.model.core.updates} weights={weights_sha256(app.model)[:16]}")
    print('  live :', read_live(app.model))
    print('  reset:', read_reset(app.model))
    print()

    for i, (prompt, answer) in enumerate(SEQ, 1):
        before_updates = app.model.core.updates
        try:
            app.teach({'prompt': prompt, 'answer': answer,
                       'repeats': 16, 'confirmation': True})
            outcome = 'ACCEPT'
        except LearningGuardRejected:
            outcome = 'reject'
        except Exception as exc:
            outcome = f'error:{type(exc).__name__}'
        live = read_live(app.model)
        reset = read_reset(app.model)
        print(f'step {i} {outcome:7s} {prompt!r} -> {answer!r}  '
              f'updates {before_updates}->{app.model.core.updates}')
        print(f'   live : {live}')
        print(f'   reset: {reset}')
        print()

    print('final weights:', weights_sha256(app.model)[:16])
    print()
    print('CONCLUSION criterion: the first step where a shipped answer changes in')
    print('BOTH live and reset reads has altered the weights, not just the state.')
finally:
    app.close()
    shutil.rmtree(d.parent, ignore_errors=True)
