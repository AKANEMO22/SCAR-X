"""Reproducible product chat benchmark on a copy of a session.

Product-first acceptance measurement, NOT a research benchmark: it has no
training, no seeds and no claim about generalisation. It exists so that every
chat/vision change can be compared against the same fixed prompts instead of
being judged by one lucky reply.

Runs entirely on a throwaway copy of the data directory, so the live session is
never opened for writing.
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, normalized  # noqa: E402

# Fixed product acceptance set. `expect` is the shortest acceptable answer the
# product should produce; it is NOT a truth oracle and is NOT used as a label.
CASES = [
    ('hi',                'hello'),
    ('hello',             'hello'),
    ('chào',              'chào bạn'),
    ('chào bạn',          'chào bạn'),
    ('bạn là ai',         'scAR-x'),
    ('bạn tên gì',        'scAR-x'),
    ('cảm ơn',            'không có gì'),
    ('tạm biệt',          'tạm biệt'),
]

VISION_CASES = [
    # (label, pixels built in code) - a vertical stroke is digit 1.
    ('1', lambda: [[0] * 8 for _ in range(8)]),
]


def cycles(words):
    for size in range(1, max(1, len(words) // 2) + 1):
        for start in range(len(words) - 2 * size + 1):
            if words[start:start + size] == words[start + size:start + 2 * size]:
                return size
    return 0


def score_case(prompt, expect, text):
    words = text.split()
    out = normalized(text)
    routed = text.startswith('[Hỏi để học]')
    return {
        'prompt': prompt,
        'expect': expect,
        'output': text,
        'routed_to_question': routed,
        'empty': not text.strip(),
        'cycle_length': cycles(words),
        'word_count': len(words),
        # A reply counts only if it is real neural output, non-empty, not a
        # repetition loop and not a controller question.
        'usable': bool(text.strip()) and not routed and cycles(words) == 0,
        'expected_present': normalized(expect) in out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default=str(ROOT / 'local' / 'generation1'))
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    source = Path(args.session)
    work = Path(tempfile.mkdtemp(prefix='scarx-bench-')) / 'session'
    shutil.copytree(source, work, ignore=shutil.ignore_patterns('session.lock'))
    app = Generation(work, background=False)
    try:
        rows = []
        for prompt, expect in CASES:
            result = app.chat(prompt, record_experience=False)
            rows.append(score_case(prompt, expect, result['text']))

        total = len(rows)
        usable = sum(r['usable'] for r in rows)
        hit = sum(r['expected_present'] for r in rows)
        routed = sum(r['routed_to_question'] for r in rows)
        looping = sum(r['cycle_length'] > 0 for r in rows)

        print(f"=== PRODUCT CHAT BENCHMARK ({total} fixed prompts) ===")
        print(f"{'prompt':16s} {'usable':7s} {'hit':4s} {'cycle':6s}  output")
        for r in rows:
            print(f"{r['prompt']:16s} {str(r['usable']):7s} {str(r['expected_present']):4s} "
                  f"{r['cycle_length']:<6d}  {r['output'][:44]!r}")
        print()
        print(f"usable replies      : {usable}/{total}")
        print(f"expected content    : {hit}/{total}")
        print(f"routed to question  : {routed}/{total}")
        print(f"repetition loops    : {looping}/{total}")

        vision = {}
        try:
            import base64
            import io
            from PIL import Image

            img = Image.new('L', (8, 8), 0)
            for y in range(2, 6):
                img.putpixel((3, y), 255)
                img.putpixel((4, y), 255)
            buf = io.BytesIO()
            img.save(buf, 'PNG')
            uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
            out = app.vision({'image': uri, 'prompt': 'digit?'})
            vision = {'digit': out['digit'], 'confidence': float(max(out['probabilities'])),
                      'correct': out['digit'] == 1}
            print(f"\nvision stroke-as-1  : digit={vision['digit']} "
                  f"confidence={vision['confidence']:.3f} correct={vision['correct']}")
        except Exception as exc:  # vision is optional in this harness
            print(f'\nvision skipped: {type(exc).__name__}: {exc}')
            vision = {'error': f'{type(exc).__name__}: {exc}'}

        summary = {
            'cases': total, 'usable': usable, 'expected_present': hit,
            'routed': routed, 'looping': looping,
            'usable_rate': usable / total, 'hit_rate': hit / total,
            'vision': vision, 'rows': rows,
            'scope': 'Product acceptance only; fixed prompts, no training, no seeds, '
                     'no generalisation claim. Runs on a discarded session copy.',
        }
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'\nwrote {args.json_out}')
        return summary
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)


if __name__ == '__main__':
    main()
