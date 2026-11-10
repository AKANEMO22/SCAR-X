"""Did capacity growth (48 -> 128) break the shipped weights?

Shipped weights: updates 241123, hidden 48  -> answers correct
Live session   : updates 241363, hidden 128 -> answers broken

Only 240 updates separate them, so either the extra training hurt, or growing the
network from 48 to 128 hidden neurons corrupted the readout.

Test by taking the SHIPPED weights and growing them to 128 WITHOUT any training,
then measuring. If the answers break, growth itself is the cause and no amount of
teaching can fix it; the fix belongs in the expansion routine, not in learning.
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
from scarx.evaluation.product_acceptance import weights_sha256  # noqa: E402

PROMPTS = ['hi', 'chào bạn', 'bạn là ai', 'tạm biệt']


def fresh():
    d = Path(tempfile.mkdtemp(prefix='scarx-grow-')) / 's'
    return d, Generation(d, background=False)


def show(label, app):
    print(f'  {label}')
    for p in PROMPTS:
        print(f"      {p!r:12s} -> {app.model.generate(p)['text']!r}")
    print(f"      hidden={app.model.core.config.hidden} "
          f"active={app.model.active_count} weights={weights_sha256(app.model)[:16]}")


print('=== baseline: fresh shipped weights (48) ===')
d, app = fresh()
try:
    show('shipped', app)
finally:
    app.close()
    shutil.rmtree(d.parent, ignore_errors=True)

for target in (64, 96, 128):
    print(f'\n=== grow shipped weights to {target}, NO training ===')
    d, app = fresh()
    try:
        before = weights_sha256(app.model)
        try:
            app.model.select_capacity(target)
            grown = weights_sha256(app.model)
            show(f'grown to {target}', app)
            print(f'      weights changed by growth: {before != grown}')
        except Exception as exc:
            print(f'      growth failed: {type(exc).__name__}: {exc}')
    finally:
        app.close()
        shutil.rmtree(d.parent, ignore_errors=True)

print('\n=== does shrinking back to 48 recover? ===')
d, app = fresh()
try:
    app.model.select_capacity(128)
    print('  after grow to 128:')
    for p in PROMPTS:
        print(f"      {p!r:12s} -> {app.model.generate(p)['text']!r}")
    app.model.select_capacity(48)
    print('  after shrink back to 48:')
    for p in PROMPTS:
        print(f"      {p!r:12s} -> {app.model.generate(p)['text']!r}")
finally:
    app.close()
    shutil.rmtree(d.parent, ignore_errors=True)
