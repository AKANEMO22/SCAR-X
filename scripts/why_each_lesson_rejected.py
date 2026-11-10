"""Why EXACTLY was each of the three lessons rejected? Per-rule evidence.

The user explicitly warned: do not assume normalized() explains everything.
This decomposes every rejection into concrete causes, per guard pair:

  1. which guard pairs were passing BEFORE
  2. which of those fail AFTER
  3. for each lost pair: was it lost because of
       (a) trailing punctuation only  -> normalized() masks it
       (b) repetition loop
       (c) truncated content
       (d) content actually missing
  4. and whether the SAME pair was also broken in live state (state effect)
     or only in the reset clone (measurement effect)

Run after restoring the good checkpoint, on a copy.
"""
import json
import shutil
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, normalized  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

GOOD = 'b305479afb0e40c1ba5a04acd0139658'
LIVE = ROOT / 'local' / 'generation1'
LESSONS = [('hi', 'Hello.'), ('bạn tên gì', 'Tôi là SCAR-X.'), ('cảm ơn', 'Không có gì.')]


def cycles(words):
    for size in range(1, max(1, len(words) // 2) + 1):
        for start in range(len(words) - 2 * size + 1):
            if words[start:start + size] == words[start + size:start + 2 * size]:
                return size
    return 0


def classify(output, target):
    """Which defect does this output have relative to target?"""
    out_words = output.split()
    tgt = normalized(target)
    out = normalized(output)
    kinds = []
    if out != tgt:
        if out.rstrip(' .!?') == tgt.rstrip(' .!?') or out.replace(' ', '') == tgt.replace(' ', ''):
            kinds.append('punctuation_only')
        elif cycles(out_words):
            kinds.append('repetition')
        elif tgt.startswith(out) and len(out) < len(tgt):
            kinds.append('truncated')
        else:
            kinds.append('content_differs')
    return kinds


def score_both(model, prompt, target):
    clone = UnifiedCore.loads(model.dumps())
    clone.core.reset_state()
    clone.last_active[:] = clone.ticks
    reset_out = clone.generate(prompt)['text']
    live_out = model.generate(prompt)['text']
    return {
        'reset_out': reset_out, 'reset_pass': normalized(reset_out) == normalized(target),
        'live_out': live_out, 'live_pass': normalized(live_out) == normalized(target),
    }


work = Path(tempfile.mkdtemp(prefix='scarx-why-')) / 'session'
shutil.copytree(LIVE, work, ignore=shutil.ignore_patterns('session.lock'))
app = Generation(work, background=False)
report = []
try:
    app.restore(GOOD)
    print(f'after restore, weights={normalized(str(app.model.core.updates))}')

    for lesson_prompt, lesson_answer in LESSONS:
        print(f'\n{"=" * 66}')
        print(f'LESSON {lesson_prompt!r} -> {lesson_answer!r}')
        pairs = app._guard_pairs(lesson_prompt)
        before = {p: score_both(app.model, p, t) for p, t in pairs}
        passing = [p for p in before if before[p]['reset_pass']]

        try:
            app.teach({'prompt': lesson_prompt, 'answer': lesson_answer,
                       'repeats': 16, 'confirmation': True})
            print('  outcome: ACCEPTED')
        except Exception as exc:
            print(f'  outcome: {type(exc).__name__}')

        after = {p: score_both(app.model, p, t) for p, t in pairs}
        lost = [p for p in passing if not after[p]['reset_pass']]
        print(f'  guard pairs passing before: {len(passing)}')
        print(f'  guard pairs lost          : {len(lost)}')

        for p in lost:
            b, a = before[p], after[p]
            kinds = classify(a['reset_out'], [t for pp, t in pairs if pp == p][0])
            also_live = not a['live_pass']
            print(f"    LOST {p!r}")
            print(f"      before reset={b['reset_out']!r} live={b['live_out']!r}")
            print(f"      after  reset={a['reset_out']!r} live={a['live_out']!r}")
            print(f"      defect kinds    : {kinds}")
            print(f"      broken in live  : {also_live}")

        report.append({
            'lesson': lesson_prompt,
            'guard_pairs': len(pairs),
            'passing_before': len(passing),
            'lost': [{'prompt': p,
                      'before_reset': before[p]['reset_out'],
                      'after_reset': after[p]['reset_out'],
                      'after_live': after[p]['live_out'],
                      'kinds': classify(after[p]['reset_out'],
                                        [t for pp, t in pairs if pp == p][0]),
                      'broken_in_live': not after[p]['live_pass']}
                     for p in lost],
        })
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)

print(f'\n{"=" * 66}')
print('SUMMARY: cause tally across all rejections')
tally = {}
for entry in report:
    for lost in entry['lost']:
        for kind in lost['kinds']:
            tally[kind] = tally.get(kind, 0) + 1
print(' ', tally if tally else '(no pair lost at all)')

out = ROOT / 'local' / 'rejection-causes.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'\nwrote {out}')
