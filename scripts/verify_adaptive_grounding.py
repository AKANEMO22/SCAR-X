"""Independent re-verification of the delivery numbers.

The handover states "388 output tokens needed, exceeding the 256 vocabulary cap",
while artifacts/adaptive-curriculum-grounding.json records
combined_public_release.distinct_reference_tokens = 386. One of them is wrong or
they count different things. This recomputes every headline number from the raw
dataset instead of trusting either, and prints the exact basis for each.

Read-only. No model, no training, no network.
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.neural.language import tokenize  # noqa: E402

ART = ROOT / 'artifacts' / 'adaptive-curriculum-grounding.json'
DATA = ROOT / 'data' / 'candidates' / '20261110T142721Z-adaptive-curriculum-grounding-v1-a7c646' / 'dataset.json'
REVIEWED = ROOT / 'artifacts' / 'adaptive-curriculum-reviewed.json'

art = json.loads(ART.read_text(encoding='utf-8'))
print('artifact decision :', art['decision'])
print('artifact sha256   :', art['sha256'][:20])
print()

# --- verify the dataset file matches the recorded digest ---
raw = DATA.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
print(f'dataset file      : {DATA.name}')
print(f'  recorded sha256 : {art["sha256"]}')
print(f'  actual sha256   : {digest}')
print(f'  MATCH           : {digest == art["sha256"]}')
print()

data = json.loads(raw)
print('dataset keys      :', sorted(data.keys()))
print()

# The dataset names these 'chat' and 'vision', not chat_references/vision_qa.
chat = [r for r in data.get('chat', [])]
vision = [r for r in data.get('vision', [])]
print(f'chat references   : {len(chat)}  (artifact says {art["chat_references"]})')
print(f'vision QA         : {len(vision)}  (artifact says {art["vision_qa"]})')
print(f'scenes            : {len(data.get("scenes", []))}  (artifact says {art["owned_images"]})')
print(f'automatic_learning_allowed : {data.get("automatic_learning_allowed")}')

# --- recompute distinct reference tokens, showing the exact basis ---
def tok(text):
    return tokenize(text)


chat_tokens = set()
for row in chat:
    chat_tokens.update(tok(row['reference_answer']))
vision_tokens = set()
for row in vision:
    vision_tokens.update(tok(row['reference_answer']))

chat_only = chat_tokens
vision_only = vision_tokens
combined = chat_tokens | vision_tokens

print('=== TOKEN RECOUNT (using scarx tokenizer) ===')
print(f'  distinct tokens in chat references   : {len(chat_only)}')
print(f'  distinct tokens in vision references : {len(vision_only)}')
print(f'  union (chat + vision)                : {len(combined)}')
print(f'  artifact combined_public_release     : {art["compatibility"]["combined_public_release"]["distinct_reference_tokens"]}')
print(f'  handover claim                       : 388')
print()

# The artifact reports per-set numbers too; compare each.
for key in ('chat_public_release', 'vision_public_release'):
    rec = art['compatibility'][key]['distinct_reference_tokens']
    mine = len(chat_only) if key.startswith('chat') else len(vision_only)
    print(f'  {key:24s} artifact={rec:4d}  mine={mine:4d}  match={rec == mine}')
print()

# --- length checks against the code limits ---
cap = art['compatibility']['chat_public_release']['vocabulary_cap_in_current_code']
limit = art['compatibility']['chat_public_release']['target_token_limit_in_current_code']
print('=== LENGTH vs CODE LIMITS ===')
print(f'  vocabulary cap in code : {cap}')
print(f'  target token limit     : {limit}')
over = [r['id'] for r in chat + vision if len(tok(r['reference_answer'])) > limit]
print(f'  references longer than {limit}: {len(over)}')
print(f'  exceeds vocabulary cap : {len(combined) > cap}  ({len(combined)} > {cap})')
print(f'  overflow over cap      : {len(combined) - cap} tokens')
print()

# --- does 388 come from adding the two sets instead of unioning? ---
print('=== RECONCILING 386 vs 388 ===')
print(f'  sum of set sizes (double counts overlap): {len(chat_only) + len(vision_only)}')
print(f'  union                                   : {len(combined)}')
print(f'  overlap (tokens in both)                : {len(chat_only & vision_only)}')
print(f'  union + ? to reach 388                  : {388 - len(combined)}')
overlap = len(chat_only & vision_only)
print(f'  sum - overlap                           : {len(chat_only) + len(vision_only) - overlap}')
print()

# --- collision analysis: do different colours collapse to identical 64-pixel input? ---
print('=== COLOUR COLLISION AT 64-PIXEL INPUT ===')
sys.path.insert(0, str(ROOT / 'src'))
import base64  # noqa: E402
from scarx.app.pilots import PilotLab  # noqa: E402

# PilotLab.pixels takes a data URI, not raw bytes.
def pixels_of(path):
    uri = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()
    return PilotLab.pixels(uri)[0]

groups = defaultdict(list)
imgdir = DATA.parent / 'images'
for png in sorted(imgdir.glob('*.png')):
    groups[pixels_of(png).tobytes()].append(png.name)

print(f'  images decoded : {sum(len(v) for v in groups.values())}')
collisions = {k: v for k, v in groups.items() if len(v) > 1}
print(f'  distinct 64-pixel inputs : {len(groups)}')
print(f'  collision groups         : {len(collisions)}')
print(f'  artifact says            : {art["legacy64_equal_color_groups"]} (legacy64_equal_color_groups)')
print()
sizes = Counter(len(v) for v in collisions.values())
print('  collision group sizes:', dict(sizes))
print('  (each group = different RGB scenes that become the SAME 64-pixel input)')
print()
for key, names in list(collisions.items())[:3]:
    print(f'  group of {len(names)}: {names}')
