"""Why does a FRESH session answer correctly while the long-lived one does not?

Observed:
  * live session (local/generation1, 241363 updates, idle running for hours)
        'chào bạn'  -> 'Chào'
        'bạn là ai' -> 'Tôi là scar-x là'   (wrong)
  * fresh session (verify-guard-live, no idle)
        'chào bạn'  -> 'Chào bạn.'          (correct)
        'bạn là ai' -> 'Tôi là scar-x.'     (correct)

Two candidate causes:
  A. accumulated recurrent state over hours of idle ticks
  B. the trained weights themselves drifted

Discriminate by loading the LIVE weights into a fresh session directory and
measuring with and without idle. Weights are copied, not retrained.
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

from scarx.app.generation import Generation  # noqa: E402
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, copy_session, measure, summarise, weights_sha256,
)

PROMPTS = ['hi', 'chào bạn', 'bạn là ai', 'tạm biệt', 'cảm ơn']

print('=== A. live session copy, idle forced OFF ===')
w = copy_session(ROOT / 'local' / 'generation1')
try:
    app = Generation(w, background=False)
    try:
        print('  updates:', app.model.core.updates, ' hidden:', app.model.core.config.hidden)
        for p in PROMPTS:
            print(f"    {p!r:14s} -> {app.model.generate(p)['text']!r}")
        print('  weights:', weights_sha256(app.model)[:24])
    finally:
        app.close()
finally:
    shutil.rmtree(w.parent, ignore_errors=True)

print('\n=== B. fresh empty session (shipped weights, no history) ===')
w2 = Path(tempfile.mkdtemp(prefix='scarx-fresh-')) / 's'
try:
    app = Generation(w2, background=False)
    try:
        print('  updates:', app.model.core.updates, ' hidden:', app.model.core.config.hidden)
        for p in PROMPTS:
            print(f"    {p!r:14s} -> {app.model.generate(p)['text']!r}")
        print('  weights:', weights_sha256(app.model)[:24])
    finally:
        app.close()
finally:
    shutil.rmtree(w2.parent, ignore_errors=True)

print('\n=== C. live weights, but state reset before every read ===')
w3 = copy_session(ROOT / 'local' / 'generation1')
try:
    app = Generation(w3, background=False)
    try:
        rows = measure(app, CASES, reset_state=True)
        for r in rows:
            print(f"    {r['prompt']:14s} -> {r['text']!r:26s} correct={r['correct']}")
        print('  summary:', {k: round(v, 3) for k, v in summarise(rows).items()})
    finally:
        app.close()
finally:
    shutil.rmtree(w3.parent, ignore_errors=True)
