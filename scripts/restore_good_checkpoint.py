"""Restore the good checkpoint safely: back up first, verify, then restore.

This is the only script that writes to a session, so it is deliberately careful:

  1. refuses to run while another process holds the session lock
  2. copies the whole data dir to a timestamped backup BEFORE touching anything
  3. restores the chosen checkpoint
  4. re-measures and prints before/after
  5. tells you exactly how to roll back using the backup

Default mode is --dry-run: it prints what it WOULD do and touches nothing.
Pass --apply to actually restore.

Usage:
  python scripts/restore_good_checkpoint.py                 # dry run
  python scripts/restore_good_checkpoint.py --apply         # do it
  python scripts/restore_good_checkpoint.py --apply --session local/other
"""
import argparse
import datetime
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation  # noqa: E402
from scarx.evaluation.product_acceptance import CASES, measure, summarise, weights_sha256  # noqa: E402

TARGET = 'b305479afb0e40c1ba5a04acd0139658'
DEFAULT_SESSION = ROOT / 'local' / 'generation1'


def measure_session(session):
    app = Generation(session, background=False)
    try:
        rows = measure(app, CASES, reset_state=True)
        return summarise(rows), {r['prompt']: r['text'] for r in rows}, weights_sha256(app.model)
    finally:
        app.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default=str(DEFAULT_SESSION))
    ap.add_argument('--checkpoint', default=TARGET)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    session = Path(args.session)
    db = session / 'generation1.sqlite'
    if not db.is_file():
        print(f'ERROR: no session database at {db}')
        return 1

    lock = session / 'session.lock'
    if lock.exists():
        try:
            with open(lock, 'r+b') as handle:
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            print('ERROR: the session is in use by a running SCAR-X process.')
            print('       Stop it first (Ctrl+C in its window), then run this again.')
            return 2

    con = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
    row = con.execute('SELECT id, created, action FROM checkpoints WHERE id = ?',
                      (args.checkpoint,)).fetchone()
    con.close()
    if not row:
        print(f'ERROR: checkpoint {args.checkpoint} not found in this session.')
        return 3
    cid, created, action = row
    print(f'target checkpoint : {cid}')
    print(f'  created         : {created}')
    print(f'  action          : {action}')

    print('\ncurrent state:')
    before_sum, before_texts, before_sha = measure_session(session)
    print(f'  correct={before_sum["free_answer_correct"]:.3f} '
          f'premature_stop={before_sum["premature_stop"]:.3f} sha={before_sha[:16]}')

    if not args.apply:
        print('\nDRY RUN — nothing changed.')
        print('Add --apply to restore. A full backup is made first.')
        print('\nWhat you would gain (measured earlier on a copy): 0.000 -> 0.625 correct')
        return 0

    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    backup = session.parent / f'backup-{stamp}'
    print(f'\nbacking up -> {backup}')
    shutil.copytree(session, backup, ignore=shutil.ignore_patterns('session.lock'))
    size = sum(f.stat().st_size for f in backup.rglob('*') if f.is_file()) / 1e6
    print(f'  backup size {size:.1f} MB')

    app = Generation(session, background=False)
    try:
        app.restore(cid)
    finally:
        app.close()

    print('\nafter restore:')
    after_sum, after_texts, after_sha = measure_session(session)
    print(f'  correct={after_sum["free_answer_correct"]:.3f} '
          f'premature_stop={after_sum["premature_stop"]:.3f} sha={after_sha[:16]}')
    for prompt, text in after_texts.items():
        mark = 'ok  ' if after_sum else '    '
        print(f'    {prompt:14s} -> {text!r}')

    print(f"\ncorrect: {before_sum['free_answer_correct']:.3f} -> {after_sum['free_answer_correct']:.3f}")

    if after_sum['free_answer_correct'] < before_sum['free_answer_correct']:
        print('\nWARNING: result is worse than before.')
        print(f'Roll back with:  copy "{backup}" over "{session}"')
    else:
        print(f'\nRoll back any time by copying this back over the session:')
        print(f'  {backup}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
