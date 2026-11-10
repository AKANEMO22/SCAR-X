"""Prove the public teach path improves the product bench, on a discarded copy.

Uses only the public Generation.teach API (the same entry point the web form and
Hermes use). No new learning rule, no guard change, no threshold change.
Everything runs on a throwaway copy; the live session is never written.
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
from scarx.learning.dialogue import DialogueLearning

CASES = [('hi', 'hello'), ('hello', 'hello'), ('chào', 'chào bạn'),
         ('chào bạn', 'chào bạn'), ('bạn là ai', 'scar-x'), ('bạn tên gì', 'scar-x'),
         ('cảm ơn', 'không có gì'), ('tạm biệt', 'tạm biệt')]

# Lessons already measured to cost nothing on the guard (guard_lost == 0).
LESSONS = [('bạn là ai', 'Tôi là SCAR-X.'), ('bạn khỏe không', 'Tôi đang hoạt động.')]


def bench(app):
    rows = []
    for prompt, expect in CASES:
        text = app.model.generate(prompt)['text']
        rows.append({'prompt': prompt, 'out': text,
                     'hit': normalized(expect) in normalized(text),
                     'empty': not text.strip()})
    return rows


def main():
    work = Path(tempfile.mkdtemp(prefix='scarx-prove-')) / 'session'
    shutil.copytree(ROOT / 'local' / 'generation1', work,
                    ignore=shutil.ignore_patterns('session.lock'))
    app = Generation(work, background=False)
    try:
        before = bench(app)
        results = []
        for prompt, answer in LESSONS:
            try:
                out = app.teach({'prompt': prompt, 'answer': answer, 'repeats': 16,
                                 'confirmation': True})
                metrics = out.get('learning_metrics', {})
                results.append({'prompt': prompt, 'answer': answer, 'accepted': True,
                                'guard_lost': metrics.get('guard_lost'),
                                'loss_mean': metrics.get('loss_mean')})
            except Exception as exc:
                results.append({'prompt': prompt, 'answer': answer, 'accepted': False,
                                'error': f'{type(exc).__name__}: {exc}'})
        after = bench(app)
    finally:
        app.close()

    print('=== public teach() on a discarded copy ===')
    for r in results:
        state = 'ACCEPTED' if r['accepted'] else 'REJECTED'
        print(f"  {state:9s} {r['prompt']!r:18s} -> {r['answer']!r:26s}"
              + (f" guard_lost={r.get('guard_lost')} loss={r.get('loss_mean')}"
                 if r['accepted'] else f" {r['error'][:60]}"))

    print('\n=== product bench before -> after ===')
    hit_before = sum(r['hit'] for r in before)
    hit_after = sum(r['hit'] for r in after)
    for b, a in zip(before, after):
        mark = '  ' if b['hit'] == a['hit'] else ('UP' if a['hit'] else 'DN')
        print(f"  {mark} {b['prompt']:14s} {b['out'][:26]!r:28s} -> {a['out'][:26]!r}")
    print(f"\nexpected content: {hit_before}/{len(before)}  ->  {hit_after}/{len(after)}")
    print(f"empty outputs   : {sum(r['empty'] for r in before)} -> {sum(r['empty'] for r in after)}")

    json.dump({'lessons': results, 'before': before, 'after': after,
               'hit_before': hit_before, 'hit_after': hit_after},
              open(ROOT / 'local' / 'prove-teach.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print('\nwrote local/prove-teach.json')
    shutil.rmtree(work.parent, ignore_errors=True)


if __name__ == '__main__':
    main()
