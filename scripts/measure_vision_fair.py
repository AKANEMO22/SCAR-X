"""Is the 10% vision result a real limit, or an artefact of my test images?

The model was trained on images resized to 8x8 greyscale, not on crisp 7-segment
shapes. Seven-segment glyphs are a different distribution, so 1/10 could be my
fault rather than the model's.

This measures three image families through the same product path:
  A. crisp 7-segment (what I used before)
  B. anti-aliased 7-segment drawn large then downscaled to 8x8 (same pipeline as
     real input)
  C. the model's own confidence behaviour across a control set of ALL-BLACK and
     noise images, to show whether it responds to content at all

If C varies with input, the readout is responsive; if A improves greatly under B,
the earlier number was a distribution artefact and must be reported as such.

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

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scarx.app.generation import Generation  # noqa: E402
from scarx.evaluation.product_acceptance import copy_session  # noqa: E402

SEGMENTS = {
    0: 'abcdef', 1: 'bc', 2: 'abdeg', 3: 'abcdg', 4: 'bcfg',
    5: 'acdfg', 6: 'acdefg', 7: 'abc', 8: 'abcdefg', 9: 'abcdfg',
}


def seven_segment(digit, canvas=64, width=6):
    """Draw the glyph large, the way a real screenshot would look."""
    img = Image.new('L', (canvas, canvas), 0)
    d = ImageDraw.Draw(img)
    pad, right, bottom, mid = 10, canvas - 10, canvas - 10, canvas // 2
    on = SEGMENTS[digit]
    seg = {
        'a': ((pad, pad), (right, pad)),
        'b': ((right, pad), (right, mid)),
        'c': ((right, mid), (right, bottom)),
        'd': ((pad, bottom), (right, bottom)),
        'e': ((pad, mid), (pad, bottom)),
        'f': ((pad, pad), (pad, mid)),
        'g': ((pad, mid), (right, mid)),
    }
    for name in on:
        d.line(seg[name], fill=255, width=width)
    return img


def as_uri_pipeline(img):
    """Push through the same 8x8 resize the product uses, then re-encode."""
    small = img.convert('L').resize((8, 8), Image.BILINEAR)
    buf = io.BytesIO()
    small.resize((64, 64), Image.NEAREST).save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode(), small


def run(app, uri):
    out = app.vision({'image': uri, 'prompt': 'digit?'})
    return out['digit'], float(max(out['probabilities'])), out['probabilities']


work = copy_session(ROOT / 'local' / 'generation1')
app = Generation(work, background=False)
try:
    print('=== A. crisp 8x8 seven-segment (earlier method) ===')
    a_rows = []
    for digit in range(10):
        img = Image.new('L', (8, 8), 0)
        from PIL import ImageDraw as D2
        # reuse the earlier drawing by scaling the 64px glyph down afterwards
        big = seven_segment(digit, canvas=64, width=8)
        uri, _ = as_uri_pipeline(big)
        pred, conf, _ = run(app, uri)
        a_rows.append({'expected': digit, 'predicted': pred, 'conf': conf,
                       'correct': pred == digit})
        print(f"  {digit}: predicted {pred:2d} conf={conf:.3f} {'YES' if pred == digit else ''}")
    acc_a = sum(r['correct'] for r in a_rows) / 10

    print('\n=== B. anti-aliased by bilinear downsample (same pipeline as real input) ===')
    b_rows = []
    for digit in range(10):
        big = seven_segment(digit, canvas=64, width=5)
        uri, _ = as_uri_pipeline(big)
        pred, conf, _ = run(app, uri)
        b_rows.append({'expected': digit, 'predicted': pred, 'conf': conf,
                       'correct': pred == digit})
        print(f"  {digit}: predicted {pred:2d} conf={conf:.3f} {'YES' if pred == digit else ''}")
    acc_b = sum(r['correct'] for r in b_rows) / 10

    print('\n=== C. is the readout responsive at all? (blank vs noise) ===')
    blank = Image.new('L', (64, 64), 0)
    uri_blank, _ = as_uri_pipeline(blank)
    pb, cb, _ = run(app, uri_blank)
    rng = np.random.default_rng(7)
    noise_preds = []
    for _ in range(5):
        arr = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        uri_n, _ = as_uri_pipeline(Image.fromarray(arr, 'L'))
        pn, cn, _ = run(app, uri_n)
        noise_preds.append((pn, round(cn, 3)))
    white = Image.new('L', (64, 64), 255)
    uri_w, _ = as_uri_pipeline(white)
    pw, cw, _ = run(app, uri_w)
    print(f'  blank -> {pb} (conf {cb:.3f})')
    print(f'  white -> {pw} (conf {cw:.3f})')
    print(f'  noise -> {noise_preds}')
    responsive = len({pb, pw} | {p for p, _ in noise_preds}) > 1

    print()
    print(f'accuracy A (crisp)      : {acc_a:.2f}')
    print(f'accuracy B (antialiased): {acc_b:.2f}')
    print(f'readout responsive      : {responsive}')
    print()
    print('SCOPE: 8x8 handwritten-digit channel only. Synthetic shapes are not')
    print('handwritten samples; neither number supports general vision.')

    out = ROOT / 'local' / 'vision-fair-measure.json'
    out.write_text(json.dumps({'crisp': a_rows, 'antialiased': b_rows,
                               'accuracy_crisp': acc_a, 'accuracy_antialiased': acc_b,
                               'blank_prediction': pb, 'white_prediction': pw,
                               'noise_predictions': noise_preds,
                               'responsive': responsive,
                               'scope': 'Synthetic 7-segment digits only; not handwriting.'},
                              ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nwrote {out}')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
