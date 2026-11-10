"""Resolve fully_known_reference_count = 1 against "every reference is missing a token".

artifacts/adaptive-curriculum-grounding.json reports fully_known_reference_count = 1,
while docs 43/44 state that every reference needs at least one token the live core
lacks. Both cannot be true unless exactly one reference happens to be fully covered.

This finds which reference it is and why, WITHOUT publishing the vocabulary: it
only reports counts, per-reference booleans, and the language/kind of the row.
The token strings themselves are never printed.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.neural.language import tokenize  # noqa: E402

DATA = (ROOT / 'data' / 'candidates'
        / '20261110T142721Z-adaptive-curriculum-grounding-v1-a7c646' / 'dataset.json')
data = json.loads(DATA.read_text(encoding='utf-8'))

# Reconstruct the live vocabulary WITHOUT printing it: build it the same way the
# product does, then use it only for membership tests.
sys.path.insert(0, str(ROOT / 'src'))
from scarx.app.generation import Generation  # noqa: E402
from scarx.evaluation.product_acceptance import copy_session  # noqa: E402

work = copy_session(ROOT / 'local' / 'generation1')
app = Generation(work, background=False)
try:
    live = set(app.model.vocabulary)
finally:
    app.close()
    import shutil
    shutil.rmtree(work.parent, ignore_errors=True)

print(f'live vocabulary size (count only, contents not shown): {len(live)}')
print()

print('=== CHAT REFERENCE COVERAGE ===')
fully_known = []
for row in data['chat']:
    toks = set(tokenize(row['reference_answer']))
    missing = toks - live
    if not missing:
        fully_known.append(row)
chat_missing_counts = {}
for row in data['chat']:
    toks = set(tokenize(row['reference_answer']))
    n = len(toks - live)
    chat_missing_counts[n] = chat_missing_counts.get(n, 0) + 1
print(f'  references                : {len(data["chat"])}')
print(f'  fully known (0 missing)   : {len(fully_known)}')
print('  distribution of missing-token counts:')
for n in sorted(chat_missing_counts):
    print(f'    {n} missing : {chat_missing_counts[n]} reference(s)')

print()
print('=== VISION REFERENCE COVERAGE ===')
vision_fully = 0
vis_missing = {}
for row in data['vision']:
    toks = set(tokenize(row['reference_answer']))
    n = len(toks - live)
    vis_missing[n] = vis_missing.get(n, 0) + 1
    if n == 0:
        vision_fully += 1
print(f'  references                : {len(data["vision"])}')
print(f'  fully known (0 missing)   : {vision_fully}')
print('  distribution of missing-token counts:')
for n in sorted(vis_missing):
    print(f'    {n} missing : {vis_missing[n]} reference(s)')

print()
print('=== THE ONE FULLY-KNOWN CHAT REFERENCE (no vocabulary strings emitted) ===')
for row in fully_known:
    toks = tokenize(row['reference_answer'])
    print(f'  id           : {row["id"]}')
    print(f'  language     : {row["language"]}')
    print(f'  family       : {row["family"]}')
    print(f'  reference_kind: {row["reference_kind"]}')
    print(f'  token count  : {len(toks)}  (all present in live vocabulary)')
    print(f'  text length  : {len(row["reference_answer"])} characters')
    # Show a masked form: first/last character only.
    text = row['reference_answer']
    masked = text[:1] + '*' * max(0, len(text) - 2) + text[-1:]
    print(f'  masked text  : {masked!r}')

print()
print('=== RECONCILIATION ===')
print(f'  artifact fully_known_reference_count (chat)   : 1')
print(f'  my count of fully-known chat references       : {len(fully_known)}')
print(f'  artifact fully_known_reference_count (vision) : 0')
print(f'  my count of fully-known vision references     : {vision_fully}')
print()
if len(fully_known) == 1 and vision_fully == 0:
    print('  => CONSISTENT. Exactly one chat reference is fully covered by the live')
    print('     vocabulary; every OTHER reference (670 of 671) is missing at least')
    print('     one token. docs 43/44 wording "every reference" must be corrected to')
    print('     "every reference except one chat row".')
else:
    print('  => INCONSISTENT with the artifact. Recheck the counting basis.')
