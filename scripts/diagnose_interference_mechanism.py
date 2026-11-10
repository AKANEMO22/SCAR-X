"""Where exactly does interference come from? Readout vs hidden, per step.

The user's priority 4 asks specifically about readout interference, hidden
contribution, and the effect of each update step. Earlier work measured the
symptom (a prompt starts looping); this measures the MECHANISM:

  1. snapshot weights before teaching a lesson
  2. teach one lesson, step by step
  3. after EACH step, record:
       - loss of the taught pair
       - which guard pairs are still content-ok
       - how much w_out changed vs w_in/w_rec
  4. repeat with hidden plasticity frozen (readout-only) to see whether the
     damage comes from the readout alone

No learning rule, topology or signal is changed. Copies only.
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

import numpy as np  # noqa: E402
from scarx.app.generation import Generation  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

LIVE = ROOT / 'local' / 'generation1'
LESSON = ('hi', 'Hello.')
WATCH = ['chào bạn', 'bạn là ai', 'cảm ơn', 'tạm biệt', 'hello']


def verdict(model, prompt, target):
    c = UnifiedCore.loads(model.dumps())
    c.core.reset_state()
    c.last_active[:] = c.ticks
    out = c.generate(prompt)['text']
    return out, Generation._pair_verdict(out, target)


def run(steps, freeze_hidden):
    work = Path(tempfile.mkdtemp(prefix='scarx-int-')) / 'session'
    shutil.copytree(LIVE, work, ignore=shutil.ignore_patterns('session.lock'))
    app = Generation(work, background=False)
    try:
        m = app.model
        pairs = {p: t for p, t in app._guard_pairs(LESSON[0])}
        base_w_out = m.core.w_out.copy()
        base_w_in = m.core.w_in.copy()
        base_w_rec = m.core.w_rec.copy()
        m.expand_vocabulary(LESSON[1])

        rows = []
        for step in range(1, steps + 1):
            if freeze_hidden:
                m.train_pair(LESSON[0], LESSON[1], mode='readout')
            else:
                m.train_pair(LESSON[0], LESSON[1])
            d_out = float(np.linalg.norm(m.core.w_out - base_w_out))
            d_in = float(np.linalg.norm(m.core.w_in - base_w_in))
            d_rec = float(np.linalg.norm(m.core.w_rec - base_w_rec))
            lost = []
            for p in WATCH:
                if p not in pairs:
                    continue
                _, v = verdict(m, p, pairs[p])
                if not v['content_ok']:
                    lost.append(p)
            rows.append({'step': step, 'd_w_out': round(d_out, 6),
                         'd_w_in': round(d_in, 6), 'd_w_rec': round(d_rec, 6),
                         'lost': lost})
        return rows
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


print('=== FULL LOCAL (hidden + readout both learn) ===')
full = run(8, freeze_hidden=False)
print(f"{'step':5s} {'|Δw_out|':10s} {'|Δw_in|':10s} {'|Δw_rec|':10s} lost")
for r in full:
    print(f"{r['step']:<5d} {r['d_w_out']:<10.6f} {r['d_w_in']:<10.6f} "
          f"{r['d_w_rec']:<10.6f} {r['lost']}")

print('\n=== READOUT ONLY (hidden frozen) ===')
ro = run(8, freeze_hidden=True)
print(f"{'step':5s} {'|Δw_out|':10s} {'|Δw_in|':10s} {'|Δw_rec|':10s} lost")
for r in ro:
    print(f"{r['step']:<5d} {r['d_w_out']:<10.6f} {r['d_w_in']:<10.6f} "
          f"{r['d_w_rec']:<10.6f} {r['lost']}")

first_full = next((r['step'] for r in full if r['lost']), None)
first_ro = next((r['step'] for r in ro if r['lost']), None)
print(f'\nfirst step with any loss — full local : {first_full}')
print(f'first step with any loss — readout only: {first_ro}')
print()
if first_ro is not None and first_full is not None:
    print('CONCLUSION: readout-only also loses pairs, so the interference is')
    print('            primarily READOUT overwrite, not hidden representation.')
elif first_full is not None and first_ro is None:
    print('CONCLUSION: hidden plasticity is what destroys the other pairs;')
    print('            readout-only is safe. Hidden learning is the culprit.')

out = ROOT / 'local' / 'interference-mechanism.json'
out.write_text(json.dumps({'full_local': full, 'readout_only': ro,
                           'lesson': LESSON, 'watch': WATCH},
                          ensure_ascii=False, indent=2), encoding='utf-8')
print(f'\nwrote {out}')
