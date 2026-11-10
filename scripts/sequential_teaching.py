"""Sequential teaching experiment: does learning accumulate without forgetting?

Directly addresses the user's priority #1 (learn new, keep old) and #2
(generalisation). Runs entirely on a copy of the live session.

For each lesson in sequence it records, on the SAME copy:
  acquisition  - does the taught prompt now emit the taught answer
  retention    - which previously-correct bench prompts became wrong (newly_lost)
  guard        - how many protected pairs still pass
  heldout      - accuracy on phrasings never used for teaching
  decoder      - premature stop and repetition rates
  weights      - SHA-256 before/after, so weight change is proven not assumed

No learning rule, topology, signal, guard threshold or architecture is changed.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation  # noqa: E402
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, HELDOUT, copy_session, guard_report, measure, summarise, weights_sha256,
)

LIVE = ROOT / 'local' / 'generation1'

SEQUENCE = [
    ('bạn tên gì', 'Tôi là SCAR-X.'),
    ('cảm ơn', 'Không có gì.'),
    ('tạm biệt', 'Tạm biệt bạn.'),
    ('bạn là ai', 'Tôi là SCAR-X.'),
]


def heldout_acc(app):
    ok = 0
    rows = []
    for prompt, expect in HELDOUT:
        text = app.model.generate(prompt)['text']
        hit = expect in ' '.join(text.lower().split())
        ok += hit
        rows.append({'prompt': prompt, 'expect': expect, 'output': text, 'correct': hit})
    return ok / max(1, len(rows)), rows


def snapshot(app):
    rows = measure(app, CASES)
    guard = guard_report(app, '__seq_probe__')
    acc, held = heldout_acc(app)
    return {
        'weights_sha256': weights_sha256(app.model),
        'summary': summarise(rows), 'rows': rows,
        'guard_passing': sum(g['pass'] for g in guard), 'guard_total': len(guard),
        'guard_rows': guard,
        'heldout_acc': acc, 'heldout': held,
        'correct_prompts': sorted(r['prompt'] for r in rows if r['correct']),
    }


def main():
    work = copy_session(LIVE)
    app = Generation(work, background=False)
    try:
        states = [('start', snapshot(app))]
        print(f"start: correct={states[0][1]['summary']['free_answer_correct']:.3f} "
              f"guard={states[0][1]['guard_passing']}/{states[0][1]['guard_total']} "
              f"heldout={states[0][1]['heldout_acc']:.3f} "
              f"premature={states[0][1]['summary']['premature_stop']:.3f}")

        for prompt, answer in SEQUENCE:
            outcome = {'prompt': prompt, 'answer': answer}
            try:
                out = app.teach({'prompt': prompt, 'answer': answer,
                                 'repeats': 16, 'confirmation': True})
                outcome['accepted'] = True
                outcome['loss_mean'] = out.get('learning_metrics', {}).get('loss_mean')
                outcome['guard_lost'] = out.get('learning_metrics', {}).get('guard_lost')
            except Exception as exc:
                outcome['accepted'] = False
                outcome['error'] = f'{type(exc).__name__}: {exc}'[:200]
            snap = snapshot(app)
            states.append((f'after {prompt!r}', snap))
            prev = states[-2][1]
            newly_lost = sorted(set(prev['correct_prompts']) - set(snap['correct_prompts']))
            newly_gained = sorted(set(snap['correct_prompts']) - set(prev['correct_prompts']))
            print(f"\n{outcome['accepted'] and 'ACCEPTED' or 'REJECTED'} {prompt!r} -> {answer!r}")
            if not outcome['accepted']:
                print(f"   reason: {outcome['error'][:70]}")
            print(f"   correct  : {prev['summary']['free_answer_correct']:.3f} -> "
                  f"{snap['summary']['free_answer_correct']:.3f}")
            print(f"   guard    : {prev['guard_passing']}/{prev['guard_total']} -> "
                  f"{snap['guard_passing']}/{snap['guard_total']}")
            print(f"   heldout  : {prev['heldout_acc']:.3f} -> {snap['heldout_acc']:.3f}")
            print(f"   premature: {prev['summary']['premature_stop']:.3f} -> "
                  f"{snap['summary']['premature_stop']:.3f}")
            print(f"   newly_lost={newly_lost} newly_gained={newly_gained}")
            outcome['newly_lost'] = newly_lost
            outcome['newly_gained'] = newly_gained
            states[-1][1]['outcome'] = outcome

        first = states[0][1]
        last = states[-1][1]
        print('\n' + '=' * 60)
        print('TRAJECTORY')
        print('=' * 60)
        print(f"  correct   : {first['summary']['free_answer_correct']:.3f} -> "
              f"{last['summary']['free_answer_correct']:.3f}")
        print(f"  guard     : {first['guard_passing']}/{first['guard_total']} -> "
              f"{last['guard_passing']}/{last['guard_total']}")
        print(f"  heldout   : {first['heldout_acc']:.3f} -> {last['heldout_acc']:.3f}")
        print(f"  premature : {first['summary']['premature_stop']:.3f} -> "
              f"{last['summary']['premature_stop']:.3f}")
        print(f"  weights   : {first['weights_sha256'][:20]} -> {last['weights_sha256'][:20]}")
        lost_all = sorted(set(first['correct_prompts']) - set(last['correct_prompts']))
        print(f"  NET NEWLY LOST across whole sequence: {lost_all}")

        out = {'states': [{'label': label, **snap} for label, snap in states],
               'sequence': SEQUENCE,
               'net_newly_lost': lost_all,
               'scope': 'Sequential product teaching on a discarded copy. '
                        'No rule/topology/guard change.'}
        p = ROOT / 'local' / 'sequential-teaching.json'
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        print(f'\nwrote {p}')
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


if __name__ == '__main__':
    main()
