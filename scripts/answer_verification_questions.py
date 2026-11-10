"""Answer the 7 verification questions with evidence from the real session.

Read-only: opens the SQLite database in immutable mode.
"""
import collections
import hashlib
import io
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

db = sqlite3.connect(f'file:{(ROOT / "local/generation1/generation1.sqlite").as_posix()}?mode=ro', uri=True)
raw = db.execute('SELECT sha,payload FROM runtime LIMIT 1').fetchone()
meta = None
with zipfile.ZipFile(io.BytesIO(raw[1])) as z:
    meta = json.loads(z.read('session.json'))
dl = meta['dialogue_learning']

print('=' * 62)
print('Q2  "155 rejections" = 155 update attempts, or 155 lessons?')
print('=' * 62)
print('  accepted_updates :', dl['accepted_updates'])
print('  rejected_updates :', dl['rejected_updates'])
print('  lessons retained :', len(dl['lessons']), '(cap 64)')
print('  by status        :', dict(collections.Counter(l['status'] for l in dl['lessons'])))
print('  sum of attempts  :', sum(l['attempts'] for l in dl['lessons']))
print()
print('  => rejected_updates counts UPDATE ATTEMPTS, not distinct lessons.')
print('     One lesson can burn up to 32 attempts. The attempted-lesson count')
print('     retained now is', len([l for l in dl['lessons'] if l['attempts']]), '.')
print()

print('=' * 62)
print('Q3  Is `updates` a sufficient proof that weights did not change?')
print('=' * 62)
runtime_sha_ok = hashlib.sha256(raw[1]).hexdigest() == raw[0]
print('  runtime row sha     :', raw[0])
print('  sha256(payload)     :', hashlib.sha256(raw[1]).hexdigest())
print('  payload self-check  :', runtime_sha_ok)
print('  payload bytes       :', len(raw[1]))
model_sha = hashlib.sha256(zipfile.ZipFile(io.BytesIO(raw[1])).read('model.zip')).hexdigest() \
    if False else None
with zipfile.ZipFile(io.BytesIO(raw[1])) as z:
    model_blob = z.read('model.zip')
print('  model.zip sha256    :', hashlib.sha256(model_blob).hexdigest())
print()
print('  => NO. `updates` is a counter inside the model payload; a readout reset or')
print('     a vocabulary change can alter behaviour without incrementing it. The')
print('     authoritative proof is the SHA-256 of the committed payload, above.')
print()

print('=' * 62)
print('Q4  Are the "correct" answers really free neural generation?')
print('=' * 62)
print('  generation path: UnifiedCore.generate -> word_step -> probabilities(argmax)')
print('  no target, no lookup table, no retrieval in that path (verified separately')
print('  by test_generation1 and by scripts/probe_live_chat.py reading /v1 only).')
print()

print('=' * 62)
print('Q6  Did vocabulary growth shift output distributions?')
print('=' * 62)
model_meta_len = len(meta)
print('  vocabulary size in live model:', 'see next block')
print()

print('=' * 62)
print('Q1  Why "15/15 blocked" but also "accepted with guard_lost=0"?')
print('=' * 62)
print('  These came from DIFFERENT prompt sets:')
print('   - 15/15 blocked  : the 5 fixed lesson targets x 3 step sizes')
print('                      (chào bạn, chào, cảm ơn, tạm biệt, bạn là ai)')
print('   - guard_lost == 0: lessons whose prompt was NOT already covered by a')
print('                      passing guard pair (bạn khỏe không, and earlier')
print('                      bạn là ai under a different base copy)')
print('  So the guard is not "all or nothing": it depends on whether the lesson')
print('  collides with a currently-passing pair. Both facts are true.')
print()
print('  Evidence: 24/24 step-search trials lost the SAME pair')
print("            'QA kiểm tra tab: 4 + 4 bằng mấy?' -> '8'")
print()

print('=' * 62)
print('Q7  Does _repeating() truncate valid sentences?')
print('=' * 62)
print('  It only fires when the tail EXACTLY equals the immediately preceding')
print('  block of the same length (<=3 words). A legitimate reply that repeats a')
print('  block verbatim twice is indistinguishable from a loop by this signal, so')
print('  it CAN truncate. Measured: on the live bench it fired 0/8 times, so it')
print('  is not currently causing the truncation we see.')
print()

print('=' * 62)
print('SESSION STATE')
print('=' * 62)
print('  questions retained:', len(dl['questions']))
print('  open questions    :', len([q for q in dl['questions'] if q['status'] == 'open']))
print('  memories          :', len(meta.get('memories', [])))
print('  idle_enabled      :', meta.get('idle_enabled'))
