"""Does the good checkpoint stay good when replayed in the live topology?

The best checkpoint (b305479afb0e) is hidden=64, active=48. The live session is
hidden=128, active=48. If loading the good checkpoint and letting the app grow it
to 128 preserves the 4/4 answers, then restoring it gives the user a working
product immediately, and the growth path is proven safe end to end.

Also checks whether the live session's extra hidden rows alone can explain the
regression, or whether the weights themselves were damaged by later teaching.
Read-only: opens checkpoints, never writes the session.
"""
import io
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.neural.unified import UnifiedCore  # noqa: E402

EXPECT = {'hi': 'Hello.', 'chào bạn': 'Chào bạn.',
          'bạn là ai': 'Tôi là scar-x.', 'tạm biệt': 'Tạm biệt bạn.'}
GOOD = 'b305479afb0e'
DB = ROOT / 'local' / 'generation1' / 'generation1.sqlite'


def norm(t):
    return ' '.join(t.split()).strip(' .!?').lower()


def answers(model, reset=True):
    if reset:
        model.core.reset_state()
        model.last_active[:] = model.ticks
    return {p: model.generate(p)['text'] for p in EXPECT}


def score(texts):
    return sum(norm(texts[p]) == norm(EXPECT[p]) for p in EXPECT)


db = sqlite3.connect(f'file:{DB.as_posix()}?mode=ro', uri=True)
row = db.execute('SELECT id, created, payload FROM checkpoints WHERE id LIKE ?',
                 (GOOD + '%',)).fetchone()
cid, created, payload = row
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    data = z.read('model.zip')

model = UnifiedCore.loads(data)
print(f'checkpoint {cid[:12]} created {created}')
print(f'  hidden={model.core.config.hidden} active={model.active_count}')
base = answers(model)
print(f'  as-saved  : {score(base)}/4  {base}')

for target in (64, 96, 128):
    if model.core.config.hidden >= target:
        continue
    model.select_capacity(target)
    texts = answers(model)
    print(f'  grown {target:3d}: {score(texts)}/4  hidden={model.core.config.hidden} '
          f'active={model.active_count}')

print()
print('=== can the live session be explained by topology alone? ===')
live_model = UnifiedCore.loads(
    zipfile.ZipFile(io.BytesIO(
        db.execute('SELECT payload FROM checkpoints ORDER BY created DESC LIMIT 1')
          .fetchone()[0])).read('model.zip'))
live_texts = answers(live_model)
print(f'  live checkpoint : {score(live_texts)}/4  hidden={live_model.core.config.hidden} '
      f'updates={live_model.core.updates}')

# Does shrinking the live model to 64 recover anything?
if live_model.core.config.hidden > 64:
    try:
        live_model.select_capacity(64)
        shrunk = answers(live_model)
        print(f'  shrunk to 64    : {score(shrunk)}/4  {shrunk}')
        print()
        if score(shrunk) > score(live_texts):
            print('  => topology contributed; shrinking helped.')
        else:
            print('  => shrinking does NOT recover: the weights themselves are damaged.')
            print('     Restoring the good checkpoint is the correct fix, not resizing.')
    except Exception as exc:
        print(f'  shrink failed: {type(exc).__name__}: {exc}')
