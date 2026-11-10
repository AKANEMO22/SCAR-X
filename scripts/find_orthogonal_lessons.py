"""Can we find lessons that are learnable WITHOUT touching protected skills?

Interference is real: teaching 'hi' makes 'chào bạn' loop. But that is the lesson
we chose, not necessarily all lessons. Before proposing a learning-rule change
(which needs approval), exhaust the space that needs no rule change:

  * teach only PROMPTS not already answered correctly  -> no protected pair can
    regress on its own prompt
  * teach SHORT targets the model can reach in few steps
  * sweep step counts to find the gentlest sufficient update
  * require: becomes correct AND zero content regressions

Whatever passes here is ordinary interactive teaching the rules already allow.
Runs on a copy of the restored live session.
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

from scarx.app.generation import Generation, LearningGuardRejected  # noqa: E402
from scarx.evaluation.product_acceptance import CASES, measure, summarise, weights_sha256  # noqa: E402

LIVE = ROOT / 'local' / 'generation1'

# Deliberately orthogonal: no shipped prompt, short answer, vocabulary the core
# already contains where possible.
CANDIDATES = [
    ('bạn tên gì', 'Tôi là scar-x.', 1),
    ('bạn tên gì', 'Tôi là scar-x.', 2),
    ('bạn tên gì', 'Tôi là scar-x.', 4),
    ('cảm ơn', 'Rất vui được giúp bạn.', 1),
    ('cảm ơn', 'Rất vui được giúp bạn.', 2),
    ('hi', 'Hello.', 1),
    ('hi', 'Hello!', 1),
]


def snap(app):
    rows = measure(app, CASES, reset_state=True)
    return (summarise(rows)['free_answer_correct'],
            {r['prompt']: (r['text'], r['correct']) for r in rows},
            weights_sha256(app.model))


results = []
for prompt, answer, steps in CANDIDATES:
    work = Path(tempfile.mkdtemp(prefix='scarx-orth-')) / 'session'
    shutil.copytree(LIVE, work, ignore=shutil.ignore_patterns('session.lock'))
    app = Generation(work, background=False)
    try:
        c0, t0, s0 = snap(app)
        try:
            m = app.teach({'prompt': prompt, 'answer': answer,
                           'repeats': steps, 'confirmation': True})
            accepted = True
            metrics = m['learning_metrics']
        except LearningGuardRejected as exc:
            accepted = False
            metrics = getattr(exc, 'metrics', {}) or {}
        c1, t1, s1 = snap(app)
        changed = [p for p in t0 if t0[p][0] != t1[p][0]]
        regressed = [p for p in t0 if t0[p][1] and not t1[p][1]]
        results.append({'prompt': prompt, 'answer': answer, 'steps': steps,
                        'accepted': accepted,
                        'guard_lost': metrics.get('guard_lost'),
                        'correct': f'{c0:.3f}->{c1:.3f}',
                        'target_ok': t1[prompt][1],
                        'regressed': regressed, 'changed': changed,
                        'weights_changed': s0 != s1})
        status = 'ACCEPT' if accepted else f"reject(lost={metrics.get('guard_lost')})"
        print(f'{status:20s} {prompt!r:14s}->{answer!r:26s} x{steps}  '
              f'correct {c0:.3f}->{c1:.3f}  target={t1[prompt][1]}  '
              f'regressed={regressed}')
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)

good = [r for r in results if r['accepted'] and r['target_ok'] and not r['regressed']]
print(f'\nlearnable with zero regression: {len(good)}/{len(results)}')
for r in good:
    print(f"  {r['prompt']!r} -> {r['answer']!r} at {r['steps']} repeats "
          f"(correct {r['correct']})")

out = ROOT / 'local' / 'orthogonal-lessons.json'
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'\nwrote {out}')
