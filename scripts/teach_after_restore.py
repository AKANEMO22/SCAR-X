"""After restoring the good checkpoint, can ordinary teaching fix the remaining 3?

Restore gives 5/8. The three failures are:
  'hi'         -> 'Hello'                  (missing final period)
  'bạn tên gì' -> 'Tôi là'                 (truncated)
  'cảm ơn'     -> 'Rất vui được giúp bạn.' (missing final period)

Ordinary interactive teaching is explicitly authorised by AGENTS.md. This runs it
on a discarded copy that has first been restored, and reports before/after with
the strict acceptance metric plus a retention check.

Writes nothing to the live session.
"""
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, LearningGuardRejected  # noqa: E402
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, measure, summarise, weights_sha256,
)

GOOD = 'b305479afb0e40c1ba5a04acd0139658'
LIVE = ROOT / 'local' / 'generation1'
LESSONS = [
    ('hi', 'Hello.'),
    ('bạn tên gì', 'Tôi là SCAR-X.'),
    ('cảm ơn', 'Không có gì.'),
]


def fresh_restored():
    work = Path(tempfile.mkdtemp(prefix='scarx-fix-')) / 'session'
    shutil.copytree(LIVE, work, ignore=shutil.ignore_patterns('session.lock'))
    app = Generation(work, background=False)
    app.restore(GOOD)
    return work, app


def snapshot(app):
    rows = measure(app, CASES, reset_state=True)
    return summarise(rows), {r['prompt']: r['text'] for r in rows}, weights_sha256(app.model)


work, app = fresh_restored()
try:
    base_sum, base_texts, base_sha = snapshot(app)
    print(f'after restore: correct={base_sum["free_answer_correct"]:.3f} '
          f'premature={base_sum["premature_stop"]:.3f}')
    for p, t in base_texts.items():
        print(f'   {p:14s} -> {t!r}')

    print()
    prev_texts = dict(base_texts)
    for prompt, answer in LESSONS:
        before = {p: t for p, t in prev_texts.items()}
        try:
            app.teach({'prompt': prompt, 'answer': answer,
                       'repeats': 16, 'confirmation': True})
            outcome = 'ACCEPT'
        except LearningGuardRejected as exc:
            outcome = f'reject(guard_lost={getattr(exc, "metrics", {}).get("guard_lost")})'
        except Exception as exc:
            outcome = f'error:{type(exc).__name__}'
        s, texts, sha = snapshot(app)
        lost = [p for p in before if p in texts and before[p] != texts[p]]
        print(f'{outcome:24s} {prompt!r:14s} -> {answer!r}')
        print(f'   correct={s["free_answer_correct"]:.3f} premature={s["premature_stop"]:.3f} '
              f'changed={lost}')
        for p, t in texts.items():
            print(f'      {p:14s} -> {t!r}')
        prev_texts = texts
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
