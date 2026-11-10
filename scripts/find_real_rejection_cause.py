"""guard_lost is 0 in my replay but teach() still raises. Find the real condition.

My external replay shows 0 lost pairs, yet teach() raises LearningGuardRejected.
So the guard sees a lost pair that my replay does not. Differences between the two:

  * teach() scores BEFORE anything, then expands the vocabulary, then trains,
    then scores AFTER - all on the SAME model object
  * my replay scores before, then calls teach(), then scores after - the
    vocabulary has already grown by the time I measure "after"

If vocabulary growth itself breaks a pair, the guard catches it between
expand_vocabulary() and train_pair(), a window my replay never observes.

Test by reproducing teach()'s exact order with explicit checkpoints.
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

from scarx.app.generation import Generation, normalized  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

GOOD = 'b305479afb0e40c1ba5a04acd0139658'
LIVE = ROOT / 'local' / 'generation1'


def guard_view(model, pairs):
    c = UnifiedCore.loads(model.dumps())
    out = []
    for prompt, target in pairs:
        c.core.reset_state()
        c.last_active[:] = c.ticks
        text = c.generate(prompt)['text']
        out.append(normalized(text) == normalized(target))
    return out


work = Path(tempfile.mkdtemp(prefix='scarx-win-')) / 'session'
shutil.copytree(LIVE, work, ignore=shutil.ignore_patterns('session.lock'))
app = Generation(work, background=False)
try:
    app.restore(GOOD)
    m = app.model

    for lesson_prompt, lesson_answer in [('hi', 'Hello.'), ('cảm ơn', 'Không có gì.')]:
        print(f'\n=== {lesson_prompt!r} -> {lesson_answer!r} ===')
        pairs = app._guard_pairs(lesson_prompt)
        print(f'  pairs: {len(pairs)}')

        step0 = guard_view(m, pairs)
        print(f'  step 0 baseline            passing={sum(step0)}')

        # teach() order: generate() on clone (now), then expand, then train
        before_vocab = len(m.vocabulary)
        m.expand_vocabulary(lesson_answer)
        after_vocab = len(m.vocabulary)
        step1 = guard_view(m, pairs)
        print(f'  step 1 after expand_vocab  passing={sum(step1)}  '
              f'vocab {before_vocab}->{after_vocab}')

        losses_at_expand = [pairs[i][0] for i in range(len(pairs)) if step0[i] and not step1[i]]
        print(f'         lost at expansion   : {losses_at_expand}')

        for _ in range(16):
            m.train_pair(lesson_prompt, lesson_answer)
        step2 = guard_view(m, pairs)
        print(f'  step 2 after train_pair    passing={sum(step2)}')
        lost_at_train = [pairs[i][0] for i in range(len(pairs)) if step1[i] and not step2[i]]
        print(f'         lost at training    : {lost_at_train}')

        print(f'  VERDICT: rejection caused by '
              f'{"VOCABULARY EXPANSION" if losses_at_expand else "TRAINING" if lost_at_train else "something else"}')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
