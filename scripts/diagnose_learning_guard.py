"""Diagnose the learning guard on the live session, read-only.

Answers: which guard pairs exist, which ones the current weights already pass,
and whether a candidate lesson provably loses a previously-passing pair.

No new training is committed by default: every attempt runs on a throwaway copy
of the live session and is discarded, so the live checkpoint is untouched.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Only one process may write a data directory, and the live app holds its own.
# Diagnose against an explicit copy so the real session is never touched.
SESSION = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'local' / 'probe-copy'

from scarx.app.generation import Generation, LearningGuardRejected, normalized  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

app = Generation(SESSION, background=False)
try:
    print('=== live core ===')
    print('updates      =', app.model.core.updates)
    print('vocabulary   =', len(app.model.vocabulary))
    print('active       =', app.model.active_count, '/', app.model.core.config.hidden)
    print('accepted     =', app.dialogue.accepted_updates)
    print('rejected     =', app.dialogue.rejected_updates)

    lessons = app.dialogue.lessons
    print('\n=== lessons ===')
    for l in lessons:
        print(f"  {l['status']:12s} attempts={l['attempts']:3d} remaining={l.get('remaining')} "
              f"prompt={l['prompt'][:26]!r} target={l['target'][:24]!r}")

    print('\n=== guard pairs the live weights must keep ===')
    pairs = app._guard_pairs('__probe__')
    scores = app._guard_scores(app.model, pairs)
    for (p, t), ok in zip(pairs, scores):
        icon = 'PASS' if ok else 'FAIL'
        out = normalized(app.model.generate(p)['text'])
        print(f'  {icon}  prompt={p[:34]!r:38s} target={t[:22]!r:26s} now={out[:26]!r}')

    passing = [pair for pair, ok in zip(pairs, scores) if ok]
    print(f'\npassing guard pairs: {len(passing)}/{len(pairs)}')
    print('NOTE: guard only blocks when a passage currently PASSES and then breaks.')

    print('\n=== candidate lessons, dry-run on a discarded copy ===')
    for prompt, target in [('chÃ o báº¡n', 'chÃ o báº¡n mÃ¬nh lÃ  scar-x'),
                           ('chÃ o', 'ChÃ o báº¡n.'),
                           ('cáº£m Æ¡n', 'KhÃ´ng cÃ³ gÃ¬.')]:
        probe = Generation(SESSION, background=False)
        try:
            guard = probe._guard_pairs(prompt)
            before = probe._guard_scores(probe.model, guard) if guard else []
            losses = []
            for _ in range(4):
                losses.append(float(probe.model.train_pair(prompt, target)))
            after = probe._guard_scores(probe.model, guard) if guard else []
            lost = sum(b and not a for b, a in zip(before, after))
            out = normalized(probe.model.generate(prompt)['text'])
            verdict = 'WOULD_BE_BLOCKED' if lost else 'WOULD_PASS'
            print(f'  {prompt!r} -> {target!r}')
            print(f'     loss_mean={sum(losses)/len(losses):.4f} guard_lost={lost}/{len(guard)} '
                  f'output_after={out[:40]!r} verdict={verdict}')
        finally:
            probe.close()
finally:
    app.close()

