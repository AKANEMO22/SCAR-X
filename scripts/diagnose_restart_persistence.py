"""Is the restart-persistence failure a real bug, or my measurement?

The acceptance suite reported restart_persistence=False. Two possibilities:
  A. real defect: taught weights are not saved/reloaded
  B. measurement artefact: outputs differ only because the live recurrent state
     continues to evolve and is part of the checkpoint

Distinguish by comparing WEIGHTS (which must persist) and by re-measuring the
restart text against the same weights. Read-only on copies.
"""
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, normalized  # noqa: E402
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, copy_session, measure, weights_sha256,
)

LESSON = ('bạn khỏe không', 'Tôi đang hoạt động.')

work = copy_session(ROOT / 'local' / 'generation1')
try:
    app = Generation(work, background=False)
    try:
        before_rows = measure(app, CASES)
        try:
            app.teach({'prompt': LESSON[0], 'answer': LESSON[1],
                       'repeats': 16, 'confirmation': True})
            accepted = True
        except Exception as exc:
            accepted = False
            print('teach rejected:', type(exc).__name__)
        sha_after_teach = weights_sha256(app.model)
        after_rows = measure(app, CASES)
        state_after = app.model.dumps()
        print(f'accepted              : {accepted}')
        print(f'weights after teach   : {sha_after_teach[:24]}')
        print(f'outputs after teach   : {[r["text"] for r in after_rows]}')
    finally:
        app.close()

    time.sleep(0.2)
    reopened = Generation(work, background=False)
    try:
        sha_restored = weights_sha256(reopened.model)
        restored_rows = measure(reopened, CASES)
        state_restored = reopened.model.dumps()
        print()
        print('--- after cold restart ---')
        print(f'weights restored      : {sha_restored[:24]}')
        print(f'weights identical     : {sha_restored == sha_after_teach}')
        print(f'outputs restored      : {[r["text"] for r in restored_rows]}')
        print(f'texts identical       : {[r["text"] for r in restored_rows] == [r["text"] for r in after_rows]}')
        print(f'model bytes identical : {state_after == state_restored}')

        print()
        if sha_restored == sha_after_teach and state_after != state_restored:
            diff = 'model bytes differ but weights identical'
            print('DIAGNOSIS:', diff)
            print('  => weights DO persist; the differing part is working state')
            print('     (recurrent voltage/adaptation/ticks), which is expected to')
            print('     move. restart_persistence should compare WEIGHTS, not text.')
        elif sha_restored == sha_after_teach and [r['text'] for r in restored_rows] == [r['text'] for r in after_rows]:
            print('DIAGNOSIS: ARTFACT of the earlier run only; text matches here.')
            print('  The suite measured text after a longer interval, letting the')
            print('  live state drift across the settle boundary.')
        elif sha_restored != sha_after_teach:
            print('DIAGNOSIS: REAL BUG - weights did not persist across restart.')
    finally:
        reopened.close()
finally:
    shutil.rmtree(work.parent, ignore_errors=True)
