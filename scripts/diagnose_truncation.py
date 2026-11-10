"""Why do replies truncate? Is EOS chosen too early, and is token-2 right?

Product diagnostic only. Four of eight bench prompts return a correct-looking
prefix that stops mid-sentence ('Chào', 'Rất', 'Tôi là'). If the model would have
continued sensibly at rank 2, the defect is in decoding, not in learning — and a
decode fix is allowed without touching any learning rule.

Read-only: runs on a discarded copy.
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

import numpy as np  # noqa: E402
from scarx.app.generation import Generation  # noqa: E402

CASES = [('chào', 'Chào bạn.'), ('chào bạn', 'Chào bạn.'),
         ('cảm ơn', 'Không có gì.'), ('bạn tên gì', 'Tôi là SCAR-X.'),
         ('bạn là ai', 'Tôi là SCAR-X.'), ('tạm biệt', 'Tạm biệt bạn.')]
MAXPOS = 8

work = Path(tempfile.mkdtemp(prefix='scarx-trunc-')) / 'session'
shutil.copytree(ROOT / 'local' / 'generation1', work,
                ignore=shutil.ignore_patterns('session.lock'))
app = Generation(work, background=False)
try:
    report = []
    for prompt, expect in CASES:
        ctx = app.model.context(prompt)
        app.model.boundary()
        prev = 1
        rows = []
        for pos in range(MAXPOS):
            p = app.model.word_step(prev, ctx, pos, source=prompt)
            order = np.argsort(-p)[:3]
            named = [(app.model.vocabulary[int(i)], float(p[int(i)])) for i in order]
            rows.append(named)
            tok = int(order[0])
            if tok == 0:
                break
            prev = tok
        report.append({'prompt': prompt, 'expect': expect, 'positions': rows})

    print('=== decode trace: top-3 tokens per position ===')
    for r in report:
        print(f"\n{r['prompt']!r}  (want {r['expect']!r})")
        for pos, named in enumerate(r['positions']):
            txt = '  '.join(f'{w!r}:{p:.3f}' for w, p in named)
            print(f'   p{pos}: {txt}')
        # Does the argmax chain ever pick EOS while a content token was 2nd?
        chain = [n[0][0] for n in r['positions']]
        if r['positions']:
            last = r['positions'][-1]
            stopped_on_eos = last[0][0] == '<eos>'
            print(f'   chain={chain}  stopped_on_eos={stopped_on_eos}')

    json.dump(report, open(ROOT / 'local' / 'truncation-trace.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print('\nwrote local/truncation-trace.json')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
