"""Product acceptance: 12 metrics, one command, weights proven by SHA-256.

Usage:
  python scripts/acceptance_14metrics.py --baseline            (read-only baseline)
  python scripts/acceptance_14metrics.py --with-lesson PROMPT --answer ANSWER
  python scripts/acceptance_14metrics.py --json-out result.json

Never writes the live session: everything runs on a copy, and the live checkpoint
SHA-256 is captured before and after to prove it was untouched.
"""
import argparse
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
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, HELDOUT, copy_session, guard_report, measure, resources, summarise, weights_sha256,
)

LIVE = ROOT / 'local' / 'generation1'


def live_checkpoint_sha():
    db = sqlite3.connect(f'file:{(LIVE / "generation1.sqlite").as_posix()}?mode=ro', uri=True)
    row = db.execute('SELECT sha FROM runtime LIMIT 1').fetchone()
    return row[0] if row else None


def run_on_copy(lessons=(), apply_readout_reset=False):
    work = copy_session(LIVE)
    app = Generation(work, background=False)
    try:
        before = {'weights_sha256': weights_sha256(app.model), 'rows': measure(app),
                  'guard': guard_report(app), 'resources': resources(app)}
        before['summary'] = summarise(before['rows'])
        before['guard_passing'] = sum(g['pass'] for g in before['guard'])

        acquisition = []
        for prompt, answer in lessons:
            record = {'prompt': prompt, 'answer': answer, 'accepted': False}
            try:
                out = app.teach({'prompt': prompt, 'answer': answer,
                                 'repeats': 16, 'confirmation': True})
                record['accepted'] = True
                record['loss_mean'] = out.get('learning_metrics', {}).get('loss_mean')
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            record['emits_target'] = prompt in [
                r['prompt'] for r in measure(app)
            ] and any(r['prompt'] == prompt and r['correct'] for r in measure(app))
            acquisition.append(record)

        after = {'weights_sha256': weights_sha256(app.model), 'rows': measure(app),
                 'guard': guard_report(app), 'resources': resources(app)}
        after['summary'] = summarise(after['rows'])
        after['guard_passing'] = sum(g['pass'] for g in after['guard'])

        heldout_rows = []
        for prompt, expect in HELDOUT:
            text = app.model.generate(prompt)['text']
            heldout_rows.append({'prompt': prompt, 'expect': expect, 'output': text,
                                 'correct': expect in ' '.join(text.lower().split())})

        before_ids = {(r['prompt']) for r in before['rows'] if r['correct']}
        after_ids = {(r['prompt']) for r in after['rows'] if r['correct']}
        newly_lost = sorted(before_ids - after_ids)
        newly_gained = sorted(after_ids - before_ids)

        return {
            'weights_sha256_before': before['weights_sha256'],
            'weights_sha256_after': after['weights_sha256'],
            'weights_changed': before['weights_sha256'] != after['weights_sha256'],
            'before': before, 'after': after,
            'acquisition': acquisition,
            'heldout_accuracy': sum(h['correct'] for h in heldout_rows) / max(1, len(heldout_rows)),
            'heldout': heldout_rows,
            'newly_lost': newly_lost, 'newly_gained': newly_gained,
            'already_failed_before': sorted({r['prompt'] for r in before['rows'] if not r['correct']}),
        }
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


def print_report(title, r):
    b, a = r['before'], r['after']
    print(f'\n=== {title} ===')
    print(f"  weights sha256 before : {r['weights_sha256_before'][:34]}")
    print(f"  weights sha256 after  : {r['weights_sha256_after'][:34]}")
    print(f"  weights changed       : {r['weights_changed']}")
    print()
    print(f"  {'prompt':14s} {'before':22s} -> {'after':24s}")
    for br, ar in zip(b['rows'], a['rows']):
        mark = '  ' if br['correct'] == ar['correct'] else ('UP' if ar['correct'] else 'DN')
        print(f"  {mark} {br['prompt']:12s} {br['text'][:20]!r:24s} -> {ar['text'][:22]!r}")
    print()
    for key in ('free_answer_correct', 'usable', 'premature_stop', 'repetition'):
        print(f"  {key:22s}: {b['summary'][key]:.3f} -> {a['summary'][key]:.3f}")
    print(f"  {'guard_passing':22s}: {b['guard_passing']}/{len(b['guard'])} -> "
          f"{a['guard_passing']}/{len(a['guard'])}")
    print(f"  {'heldout_accuracy':22s}: {r['heldout_accuracy']:.3f}")
    print(f"  {'newly_lost':22s}: {r['newly_lost']}")
    print(f"  {'newly_gained':22s}: {r['newly_gained']}")
    print(f"  {'rss_bytes':22s}: {b['resources']['rss_bytes']} "
          f"(under 2GB: {b['resources']['under_limit']})")
    if r['acquisition']:
        print('\n  acquisition:')
        for ac in r['acquisition']:
            state = 'ACCEPTED' if ac['accepted'] else 'REJECTED'
            print(f"    {state} {ac['prompt']!r} -> {ac['answer']!r}"
                  + ('' if ac['accepted'] else f"  {ac.get('error','')[:58]}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-lesson', action='append', nargs=2, metavar=('PROMPT', 'ANSWER'),
                    default=[])
    ap.add_argument('--json-out')
    args = ap.parse_args()

    live_sha_before = live_checkpoint_sha()
    print(f'live checkpoint sha (before): {live_sha_before}')

    lessons = [tuple(x) for x in args.with_lesson]
    result = run_on_copy(lessons)
    print_report('BASELINE (no lesson)' if not lessons else 'WITH LESSON', result)

    live_sha_after = live_checkpoint_sha()
    print(f'\nlive checkpoint sha (after) : {live_sha_after}')
    print(f'live session untouched      : {live_sha_before == live_sha_after}')

    result['live_checkpoint_sha_before'] = live_sha_before
    result['live_checkpoint_sha_after'] = live_sha_after
    result['live_untouched'] = live_sha_before == live_sha_after
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                       encoding='utf-8')
        print(f'wrote {args.json_out}')


if __name__ == '__main__':
    main()
