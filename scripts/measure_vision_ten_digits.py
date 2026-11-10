"""Measure vision properly: accuracy across all ten digits, not one lucky sample.

The user asked for vision to be measured separately. Until now the only evidence
was a single synthetic stroke returning digit=1. One sample cannot distinguish a
working digit classifier from a constant-output function.

This builds deterministic synthetic digits (a 7-segment style rendering for each
of 0-9), runs them through the product vision path, and reports:
  * per-digit correctness
  * confusion pairs
  * mean confidence
  * whether the output varies at all (a constant predictor is the key failure)

Vision scope is handwritten-digit classification on 8x8 input, as documented.
This does not and cannot support any claim about general image understanding.

Read-only on a copy.
"""
import base64
import io
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image  # noqa: E402
from scarx.app.generation import Generation  # noqa: E402
from scarx.evaluation.product_acceptance import copy_session  # noqa: E402

# Segment sets (a,b,c,d,e,f,g) for a 7-segment display, 0-9.
SEGMENTS = {
    0: 'abcdef', 1: 'bc', 2: 'abdeg', 3: 'abcdg', 4: 'bcfg',
    5: 'acdfg', 6: 'acdefg', 7: 'abc', 8: 'abcdefg', 9: 'abcdfg',
}


def render(digit, size=8, thickness=1):
    """Deterministic 8x8 seven-segment rendering."""
    img = Image.new('L', (size, size), 0)
    px = img.load()
    on = SEGMENTS[digit]
    top, mid, bot = 1, size // 2, size - 2
    left, right = 1, size - 2

    def hline(y):
        for x in range(left, right + 1):
            for t in range(thickness):
                if 0 <= y + t < size:
                    px[x, y + t] = 255

    def vline(x, y0, y1):
        for y in range(y0, y1 + 1):
            for t in range(thickness):
                if 0 <= x + t < size:
                    px[x + t, y] = 255

    if 'a' in on:
        hline(top)
    if 'g' in on:
        hline(mid)
    if 'd' in on:
        hline(bot)
    if 'f' in on:
        vline(left, top, mid)
    if 'b' in on:
        vline(right, top, mid)
    if 'e' in on:
        vline(left, mid, bot)
    if 'c' in on:
        vline(right, mid, bot)
    return img


def as_uri(img):
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


work = copy_session(ROOT / 'local' / 'generation1')
app = Generation(work, background=False)
try:
    rows = []
    for digit in range(10):
        uri = as_uri(render(digit))
        out = app.vision({'image': uri, 'prompt': 'digit?'})
        probs = out['probabilities']
        rows.append({'expected': digit, 'predicted': out['digit'],
                     'confidence': float(max(probs)),
                     'correct': out['digit'] == digit,
                     'probabilities': [round(float(p), 5) for p in probs]})

    print(f"{'expected':9s} {'predicted':10s} {'conf':7s} correct")
    for r in rows:
        print(f"{r['expected']:<9d} {r['predicted']:<10d} {r['confidence']:.3f}   "
              f"{'YES' if r['correct'] else 'no'}")

    correct = sum(r['correct'] for r in rows)
    distinct = len({r['predicted'] for r in rows})
    mean_conf = sum(r['confidence'] for r in rows) / len(rows)
    confusions = [(r['expected'], r['predicted']) for r in rows if not r['correct']]

    print()
    print(f'accuracy          : {correct}/10 = {correct/10:.2f}')
    print(f'distinct outputs  : {distinct}/10  (1 would mean a constant predictor)')
    print(f'mean confidence   : {mean_conf:.3f}')
    print(f'confusions        : {confusions}')
    print()
    if distinct == 1:
        print('CONCLUSION: constant output; the readout is NOT classifying these inputs.')
    elif correct >= 8:
        print('CONCLUSION: synthetic seven-segment digits are recognised well.')
    else:
        print('CONCLUSION: partial recognition; vision is limited as documented.')
    print('SCOPE: 8x8 handwritten-digit channel only. Not screenshot, OCR or general'
          ' image understanding. Synthetic shapes are not handwritten samples.')

    out_path = ROOT / 'local' / 'vision-ten-digit.json'
    out_path.write_text(json.dumps({
        'rows': rows, 'accuracy': correct / 10, 'distinct_outputs': distinct,
        'mean_confidence': mean_conf,
        'scope': 'Synthetic seven-segment digits through the product vision path; '
                 'not handwriting, not general vision.'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nwrote {out_path}')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
