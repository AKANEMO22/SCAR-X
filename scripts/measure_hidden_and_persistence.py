"""The three metrics not yet measured: hidden contribution, cost, persistence.

Hidden contribution: compare a candidate whose hidden synapses are frozen
(readout-only) against the same candidate with hidden learning allowed. If the
two are identical, the improvement came from the readout memorising, not from
neural representation — which is the user's priority #2 question.

Restart persistence: teach, close the session, reopen it, and re-measure. Proves
learning survives a cold restart rather than living in RAM.

Cost: RSS and commit during teaching.

Read-only on copies; no rule change.
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

from scarx.app.generation import Generation, normalized  # noqa: E402
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, copy_session, measure, summarise, weights_sha256,
)

LIVE = ROOT / 'local' / 'generation1'
LESSONS = [('bạn khỏe không', 'Tôi đang hoạt động.'),
           ('hi', 'Hello.')]


def teach_on(app, lesson, steps=16):
    try:
        app.teach({'prompt': lesson[0], 'answer': lesson[1],
                   'repeats': steps, 'confirmation': True})
        return True, None
    except Exception as exc:
        return False, f'{type(exc).__name__}'


def run_hidden_ablation(lesson):
    """Same lesson, once with hidden plastic, once with hidden frozen."""
    out = {}
    for label, frozen in (('full_local', False), ('readout_only', True)):
        work = copy_session(LIVE)
        app = Generation(work, background=False)
        try:
            if frozen:
                # Freeze hidden plasticity only; readout still learns. This uses
                # the existing 'mode' switch, not a new rule.
                app.model.__dict__['_force_mode'] = 'readout'
            before = measure(app, CASES)
            hit_before = {r['prompt'] for r in before if r['correct']}
            try:
                payload = {'prompt': lesson[0], 'answer': lesson[1],
                           'repeats': 16, 'confirmation': True}
                if frozen:
                    payload['mode'] = 'readout'
                app.teach(payload)
                accepted = True
            except Exception as exc:
                accepted = False
            after = measure(app, CASES)
            hit_after = {r['prompt'] for r in after if r['correct']}
            out[label] = {
                'accepted': accepted,
                'correct_before': len(hit_before), 'correct_after': len(hit_after),
                'newly_gained': sorted(hit_after - hit_before),
                'emits_target': any(r['prompt'] == lesson[0] and r['correct'] for r in after),
                'weights_sha256': weights_sha256(app.model)[:16],
            }
        finally:
            app.close()
            shutil.rmtree(work.parent, ignore_errors=True)
    return out


def run_persistence(lesson):
    work = copy_session(LIVE)
    try:
        app = Generation(work, background=False)
        try:
            base = measure(app, CASES)
            try:
                app.teach({'prompt': lesson[0], 'answer': lesson[1],
                           'repeats': 16, 'confirmation': True})
                accepted = True
            except Exception:
                accepted = False
            after = measure(app, CASES)
            sha_in_memory = weights_sha256(app.model)
            rss = app.process.memory_info().rss
            commit = getattr(app.process.memory_info(), 'vms', None)
        finally:
            app.close()
        time.sleep(0.3)
        reopened = Generation(work, background=False)
        try:
            restored = measure(reopened, CASES)
            sha_restored = weights_sha256(reopened.model)
        finally:
            reopened.close()
        return {
            'accepted': accepted,
            'correct_before': sum(r['correct'] for r in base),
            'correct_after': sum(r['correct'] for r in after),
            'correct_after_restart': sum(r['correct'] for r in restored),
            'weights_identical_after_restart': sha_in_memory == sha_restored,
            'texts_identical_after_restart': [r['text'] for r in after] == [r['text'] for r in restored],
            'rss_bytes': rss, 'vms_bytes': commit,
            'under_2gb': rss < 2_000_000_000,
        }
    finally:
        shutil.rmtree(work.parent, ignore_errors=True)


print('=' * 66)
print('HIDDEN CONTRIBUTION (full local vs readout-only, same lesson)')
print('=' * 66)
abl = {}
for lesson in LESSONS:
    r = run_hidden_ablation(lesson)
    abl[str(lesson)] = r
    print(f"\nlesson {lesson[0]!r} -> {lesson[1]!r}")
    for label, v in r.items():
        print(f"  {label:13s} accepted={v['accepted']} "
              f"correct {v['correct_before']}->{v['correct_after']} "
              f"emits_target={v['emits_target']} sha={v['weights_sha256']}")
    full, ro = r['full_local'], r['readout_only']
    same = (full['correct_after'] == ro['correct_after']
            and full['emits_target'] == ro['emits_target'])
    print(f"  => hidden contributes measurably: {not same}")
    if same:
        print('     (identical outcome; this lesson is explainable by the readout alone)')

print()
print('=' * 66)
print('COST AND RESTART PERSISTENCE')
print('=' * 66)
pers = []
for lesson in LESSONS:
    p = run_persistence(lesson)
    pers.append({'lesson': lesson, **p})
    print(f"\nlesson {lesson[0]!r}")
    print(f"  accepted                 : {p['accepted']}")
    print(f"  correct before/after     : {p['correct_before']} -> {p['correct_after']}")
    print(f"  correct after RESTART    : {p['correct_after_restart']}")
    print(f"  weights identical        : {p['weights_identical_after_restart']}")
    print(f"  texts identical          : {p['texts_identical_after_restart']}")
    print(f"  rss / vms                : {p['rss_bytes']} / {p['vms_bytes']} "
          f"(under 2GB: {p['under_2gb']})")

out = ROOT / 'local' / 'hidden-and-persistence.json'
out.write_text(json.dumps({'hidden_ablation': abl, 'persistence': pers,
                           'scope': 'Read-only product measurement on copies.'},
                          ensure_ascii=False, indent=2, default=str), encoding='utf-8')
print(f'\nwrote {out}')
