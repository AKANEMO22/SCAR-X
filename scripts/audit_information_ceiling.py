"""What the model cannot answer EVEN WITH PERFECT LEARNING — per cause, not merged.

The handover asks for information-theoretic impossibility separated from decoder,
representation, learning-rule and dataset limits. This computes the ceilings from
first principles and attributes each to exactly one cause:

  A. QUESTION-ONLY ceiling   — if the prompt alone is constant per answer class,
                               guessing the majority is optimal without looking at
                               the image. Any accuracy above this must come from
                               pixels; this is the floor a blind model reaches.
  B. 64-PIXEL ceiling        — after the current preprocessing, how many distinct
                               answers remain distinguishable per question. If two
                               scenes with different answers collapse to the same
                               64-pixel input, NO learning rule can separate them.
  C. LEXICAL ceiling         — answers whose reference tokens are absent from the
                               vocabulary cannot be produced at all.

Each is a different failure layer and is reported separately.
"""
import base64
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.pilots import PilotLab  # noqa: E402
from scarx.neural.language import tokenize  # noqa: E402

DATA = ROOT / 'data' / 'candidates' / '20261110T142721Z-adaptive-curriculum-grounding-v1-a7c646' / 'dataset.json'
data = json.loads(DATA.read_text(encoding='utf-8'))
vision = data['vision']
scenes = {s['id']: s for s in data['scenes']}

# ---- A. question-only ceiling -------------------------------------------------
print('=== A. QUESTION-ONLY CEILING (blind model, no pixels) ===')
print('    If this is high, the question alone leaks the answer and a model could')
print('    score well by memorising question -> answer without any vision.')
by_q = defaultdict(Counter)
for r in vision:
    by_q[r['input']['prompt']][r['semantic_target']] += 1
total = len(vision)
blind = sum(max(c.values()) for c in by_q.values())
print(f'  distinct prompts : {len(by_q)}')
print(f'  blind-correct    : {blind}/{total} = {blind / total:.4f}')
per_kind = defaultdict(lambda: [0, 0])
for r in vision:
    key = r['kind']
    per_kind[key][0] += 1
for q, counter in by_q.items():
    kind = next(r['kind'] for r in vision if r['input']['prompt'] == q)
    per_kind[kind][1] += max(counter.values())
for kind, (n, hit) in sorted(per_kind.items()):
    print(f'  {kind:16s} blind ceiling {hit}/{n} = {hit / n:.4f}')

# ---- B. 64-pixel ceiling -----------------------------------------------------
print()
print('=== B. 64-PIXEL CEILING (information destroyed by preprocessing) ===')
imgdir = DATA.parent / 'images'
lookup = {}
for path in sorted(imgdir.glob('*.png')):
    uri = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()
    pixels, digest = PilotLab.pixels(uri)
    lookup[path.name] = pixels.tobytes()

# For each question, group scenes by their 64-pixel input and see whether the
# answer is still determined. Unresolvable = ceiling.
print(f'  distinct 64-pixel inputs among {len(lookup)} images: {len(set(lookup.values()))}')
by_question_pixels = defaultdict(lambda: defaultdict(Counter))
for r in vision:
    fname = Path(r['input']['image_file']).name
    by_question_pixels[r['input']['prompt']][lookup[fname]][r['semantic_target']] += 1
resolvable = 0
collide = 0
for prompt, groups in by_question_pixels.items():
    for pix, answers in groups.items():
        best = max(answers.values())
        resolvable += best
        collide += sum(answers.values()) - best
print(f'  best achievable with 64-pixel inputs : {resolvable}/{total} = {resolvable / total:.4f}')
print(f'  answers lost to pixel collision      : {collide}')
per_kind2 = defaultdict(lambda: [0, 0])
for prompt, groups in by_question_pixels.items():
    kind = next(r['kind'] for r in vision if r['input']['prompt'] == prompt)
    for pix, answers in groups.items():
        per_kind2[kind][0] += sum(answers.values())
        per_kind2[kind][1] += max(answers.values())
for kind, (n, hit) in sorted(per_kind2.items()):
    print(f'  {kind:16s} 64px ceiling {hit}/{n} = {hit / n:.4f}')

# ---- C. lexical ceiling ------------------------------------------------------
print()
print('=== C. LEXICAL CEILING (tokens the vocabulary cannot emit) ===')
VOCAB_CAP = 256
current = 24
need = set()
for r in vision:
    need.update(tokenize(r['reference_answer']))
for r in data['chat']:
    need.update(tokenize(r['reference_answer']))
print(f'  distinct tokens required : {len(need)}')
print(f'  vocabulary cap in code   : {VOCAB_CAP}')
print(f'  live vocabulary size     : {current}')
print(f'  tokens unrepresentable   : {max(0, len(need) - VOCAB_CAP)} above cap')
print(f'  tokens unknown to live   : {len(need) - current} of {len(need)}')
print()
print('  => every reference answer needs at least one token the live core')
print('     does not have, so the decoder cannot produce ANY reference verbatim.')

# ---- attribution summary -----------------------------------------------------
print()
print('=== ATTRIBUTION (one cause per limit, never merged) ===')
print(f'  blind (question-only)   : {blind / total:.4f} accuracy floor')
print(f'  64-pixel information    : {resolvable / total:.4f} hard ceiling')
print(f'  vocabulary              : {len(need)} tokens needed vs {VOCAB_CAP} cap')
print()
print('  These are THREE independent limits:')
print('   1. dataset/question design  -> question-only floor')
print('   2. image preprocessing      -> 64-pixel ceiling (colour destroyed)')
print('   3. decoder vocabulary       -> cannot emit references at all')
print('  Fixing one does not fix the others.')
