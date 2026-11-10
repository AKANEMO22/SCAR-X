"""Prove a restore actually fixes the product, on a copy.

Round 16/17 established:
  * the good checkpoint (b305479afb0e) answers 4/4 even at the live topology (128)
  * the live checkpoint answers 1/4, and shrinking it does not recover

So restoring the good checkpoint is the correct fix. This proves it end to end on
a discarded copy of the real session: restore, then run the full product suite.

Writes nothing to the live session.
"""
import json
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation  # noqa: E402
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, measure, summarise, weights_sha256,
)

GOOD = 'b305479afb0e'
LIVE = ROOT / 'local' / 'generation1'


def copy_session():
    work = Path(tempfile.mkdtemp(prefix='scarx-restore-')) / 'session'
    shutil.copytree(LIVE, work, ignore=shutil.ignore_patterns('session.lock'))
    return work


def report(label, app):
    rows = measure(app, CASES, reset_state=True)
    s = summarise(rows)
    print(f'  {label:22s} correct={s["free_answer_correct"]:.3f} '
          f'premature={s["premature_stop"]:.3f} weights={weights_sha256(app.model)[:12]}')
    for r in rows:
        print(f"      {r['prompt']:14s} -> {r['text']!r:28s} correct={r['correct']}")
    return s


print('=== BEFORE restore (copy of live) ===')
work = copy_session()
try:
    app = Generation(work, background=False)
    try:
        before = report('live as-is', app)
        # Find the restore entry point the product exposes.
        candidates = [n for n in dir(app) if 'restore' in n.lower()]
        print(f'  restore API: {candidates}')

        cid = None
        con = sqlite3.connect(f'file:{(work / "generation1.sqlite").as_posix()}?mode=ro', uri=True)
        row = con.execute('SELECT id FROM checkpoints WHERE id LIKE ?', (GOOD + '%',)).fetchone()
        cid = row[0] if row else None
        print(f'  target checkpoint id: {cid}')

        if cid and candidates:
            fn = getattr(app, candidates[0])
            try:
                out = fn(cid)
                print(f'  {candidates[0]}(...) -> {type(out).__name__}')
            except TypeError:
                out = fn(checkpoint=cid)
                print(f'  {candidates[0]}(checkpoint=...) -> {type(out).__name__}')
            except Exception as exc:
                print(f'  restore error: {type(exc).__name__}: {exc}')
            print()
            print('=== AFTER restore ===')
            after = report('restored', app)
            print()
            print(f"  correct: {before['free_answer_correct']:.3f} -> {after['free_answer_correct']:.3f}")
            print(f"  premature_stop: {before['premature_stop']:.3f} -> {after['premature_stop']:.3f}")
    finally:
        app.close()
finally:
    shutil.rmtree(work.parent, ignore_errors=True)
