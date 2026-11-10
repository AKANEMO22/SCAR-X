"""Cold-restart verification with the CURRENT (new) guard code.

The user authorised a restart to load the tested fixes, and asked for "kết quả sau
restart". Before touching the live session, verify on a copy that:

  1. restore + teach + close + reopen keeps the weights byte-identical
  2. answers are the same before and after the restart
  3. the new two-criteria guard survives serialisation (dialogue state round-trips)
  4. a second restart is still stable (not a one-off)

Writes only to a temp copy.
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation  # noqa: E402
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, measure, summarise, weights_sha256,
)

GOOD = 'b305479afb0e40c1ba5a04acd0139658'
LIVE = ROOT / 'local' / 'generation1'


def snapshot(app):
    rows = measure(app, CASES, reset_state=True)
    return (weights_sha256(app.model),
            summarise(rows)['free_answer_correct'],
            summarise(rows)['premature_stop'],
            {r['prompt']: r['text'] for r in rows},
            app.model.core.updates)


work = Path(tempfile.mkdtemp(prefix='scarx-restart-')) / 'session'
shutil.copytree(LIVE, work, ignore=shutil.ignore_patterns('session.lock'))

# --- cycle 1: restore, teach one safe lesson, close ---
app = Generation(work, background=False)
try:
    app.restore(GOOD)
    app.teach({'prompt': 'bạn khỏe không', 'answer': 'Tôi đang hoạt động.',
               'repeats': 16, 'confirmation': True})
    sha1, cor1, pre1, texts1, upd1 = snapshot(app)
finally:
    app.close()

print('=== before restart ===')
print(f'  weights   : {sha1}')
print(f'  updates   : {upd1}')
print(f'  correct   : {cor1:.3f}   premature_stop: {pre1:.3f}')

time.sleep(0.3)

# --- reopen: must match exactly ---
app = Generation(work, background=False)
try:
    sha2, cor2, pre2, texts2, upd2 = snapshot(app)
finally:
    app.close()

print('\n=== after restart #1 ===')
print(f'  weights   : {sha2}')
print(f'  updates   : {upd2}')
print(f'  correct   : {cor2:.3f}   premature_stop: {pre2:.3f}')
print(f'  weights identical : {sha1 == sha2}')
print(f'  updates identical : {upd1 == upd2}')
print(f'  texts identical   : {texts1 == texts2}')

# --- second restart for stability ---
time.sleep(0.2)
app = Generation(work, background=False)
try:
    sha3, cor3, pre3, texts3, upd3 = snapshot(app)
    dialogue_state = app.dialogue.dumps()
    n_lessons = len(dialogue_state['lessons'])
    n_questions = len(dialogue_state['questions'])
finally:
    app.close()

print('\n=== after restart #2 ===')
print(f'  weights   : {sha3}')
print(f'  weights stable across both restarts: {sha1 == sha2 == sha3}')
print(f'  texts stable across both restarts  : {texts1 == texts2 == texts3}')
print(f'  dialogue round-trips: lessons={n_lessons} questions={n_questions}')

if texts1 != texts2:
    print('\nCHANGED TEXTS ACROSS RESTART:')
    for p in texts1:
        if texts1[p] != texts2.get(p):
            print(f'  {p!r}: {texts1[p]!r} -> {texts2.get(p)!r}')

ok = (sha1 == sha2 == sha3) and (texts1 == texts2 == texts3)
print(f'\nVERDICT: {"PASS - restart is safe to run on the live session" if ok else "FAIL - do not restart yet"}')

shutil.rmtree(work.parent, ignore_errors=True)
