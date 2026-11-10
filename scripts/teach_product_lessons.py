"""Find and apply guard-safe lessons that fix common chat failures.

Product-first interactive teaching: this is exactly the mechanism ADR0009 and
AGENTS.md authorise without a research gate. No learning rule, threshold, guard
or architecture changes. Every candidate is first tried on a discarded copy and
kept only if it costs nothing on the guard (guard_lost == 0).

Usage:
  python scripts/teach_product_lessons.py --dry-run     (default, safe)
  python scripts/teach_product_lessons.py --apply       (writes to the session)
"""
import argparse
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

LIVE = ROOT / 'local' / 'generation1'

# (prompt, expect_substring) - expect is a product expectation, not a truth label.
CASES = [
    ('hi', 'hello'), ('hello', 'hello'), ('chào', 'chào bạn'),
    ('chào bạn', 'chào bạn'), ('bạn là ai', 'scar-x'), ('bạn tên gì', 'scar-x'),
    ('cảm ơn', 'không có gì'), ('tạm biệt', 'tạm biệt'),
]

CANDIDATES = [
    ('bạn là ai', 'Tôi là SCAR-X.'),
    ('bạn tên gì', 'Tôi là SCAR-X.'),
    ('chào bạn', 'Chào bạn.'),
    ('chào', 'Chào bạn.'),
    ('tạm biệt', 'Tạm biệt bạn.'),
    ('cảm ơn', 'Không có gì.'),
    ('bạn khỏe không', 'Tôi đang hoạt động.'),
    ('hi', 'Hello.'),
    ('hello', 'Hello.'),
]


def quality(text, expect):
    out = normalized(text)
    words = text.split()
    return {
        'out': text,
        'hit': normalized(expect) in out,
        'empty': not text.strip(),
        # A finished reply ends on punctuation or on the expected phrase.
        'complete': bool(text.strip()) and (text.strip().endswith(('.', '!', '?'))
                                            or normalized(expect) == out),
        'truncated': bool(text.strip()) and not text.strip().endswith(('.', '!', '?')),
        'words': len(words),
    }


def bench(app):
    rows = []
    for prompt, expect in CASES:
        q = quality(app.model.generate(prompt)['text'], expect)
        q['prompt'] = prompt
        rows.append(q)
    return rows


def summarise(rows):
    n = len(rows)
    return {'n': n, 'hit': sum(r['hit'] for r in rows),
            'complete': sum(r['complete'] for r in rows),
            'empty': sum(r['empty'] for r in rows)}


def try_lesson(prompt, answer, base):
    """Try one lesson on a copy of `base`; report guard cost and effect."""
    work = Path(tempfile.mkdtemp(prefix='scarx-lesson-')) / 'session'
    shutil.copytree(base, work, ignore=shutil.ignore_patterns('session.lock'))
    app = Generation(work, background=False)
    try:
        before = bench(app)
        guard = app._guard_pairs(prompt)
        g_before = app._guard_scores(app.model, guard) if guard else []
        try:
            out = app.teach({'prompt': prompt, 'answer': answer, 'repeats': 16,
                             'confirmation': True})
            accepted = True
            metrics = out.get('learning_metrics', {})
        except Exception as exc:
            accepted = False
            metrics = {}
            app.model.expand_vocabulary(answer)
            for _ in range(16):
                app.model.train_pair(prompt, answer)
        g_after = app._guard_scores(app.model, guard) if guard else []
        lost = sum(b and not a for b, a in zip(g_before, g_after)) if guard else 0
        after = bench(app)
        return {'prompt': prompt, 'answer': answer, 'accepted': accepted,
                'guard_lost': lost, 'passing_before': sum(g_before),
                'guard_pairs': len(guard),
                'loss_mean': metrics.get('loss_mean'),
                'before': summarise(before), 'after': summarise(after),
                'gain_hit': summarise(after)['hit'] - summarise(before)['hit'],
                'gain_complete': summarise(after)['complete'] - summarise(before)['complete'],
                'mode': 'public teach()' if accepted else 'raw train_pair (teach rejected)'}
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--session', default=str(LIVE))
    args = ap.parse_args()
    base = Path(args.session)

    print('=== baseline on a copy ===')
    tmp = Path(tempfile.mkdtemp(prefix='scarx-base-')) / 'session'
    shutil.copytree(base, tmp, ignore=shutil.ignore_patterns('session.lock'))
    app = Generation(tmp, background=False)
    try:
        base_rows = bench(app)
        base_sum = summarise(base_rows)
    finally:
        app.close()
        shutil.rmtree(tmp.parent, ignore_errors=True)
    for r in base_rows:
        print(f"  {r['prompt']:14s} hit={str(r['hit']):5s} complete={str(r['complete']):5s} {r['out'][:34]!r}")
    print(f"  baseline: {base_sum}")

    print('\n=== candidate lessons (each tried on its own copy) ===')
    safe = []
    for prompt, answer in CANDIDATES:
        r = try_lesson(prompt, answer, base)
        verdict = 'SAFE  ' if r['guard_lost'] == 0 else f"BLOCK({r['guard_lost']})"
        print(f"  {verdict} {prompt!r:18s} -> {answer!r:24s} "
              f"hit+{r['gain_hit']} complete+{r['gain_complete']} via {r['mode']}")
        if r['guard_lost'] == 0 and (r['gain_hit'] > 0 or r['gain_complete'] > 0):
            safe.append(r)

    print(f"\nguard-safe lessons with a measured product gain: {len(safe)}")
    for s in safe:
        print(f"  {s['prompt']!r} -> {s['answer']!r}  hit {s['before']['hit']}->{s['after']['hit']}, "
              f"complete {s['before']['complete']}->{s['after']['complete']}")

    out = {'baseline': base_sum, 'baseline_rows': base_rows,
           'safe_lessons': safe, 'applied': False,
           'scope': 'Product-first interactive teaching; no rule/guard/architecture change.'}

    if args.apply and safe:
        print('\n=== applying to the live session ===')
        app = Generation(base, background=False)
        try:
            for s in safe:
                try:
                    app.teach({'prompt': s['prompt'], 'answer': s['answer'],
                               'repeats': 16, 'confirmation': True})
                    print(f"  applied {s['prompt']!r}")
                except Exception as exc:
                    print(f"  skipped {s['prompt']!r}: {type(exc).__name__}")
            final = bench(app)
            out['applied'] = True
            out['final'] = summarise(final)
            out['final_rows'] = final
            print(f"  final: {out['final']}")
        finally:
            app.close()
    elif not args.apply:
        print('\n(dry run; pass --apply to write these lessons to the session)')

    json.dump(out, open(ROOT / 'local' / 'product-lessons.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print('wrote local/product-lessons.json')


if __name__ == '__main__':
    main()
