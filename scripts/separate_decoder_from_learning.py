"""Separate decoder failure from learning failure (objective item 4).

The corpus already contains full answers ('Chào bạn.', 'Tôi là SCAR-X.').
The product emits prefixes ('Chào', 'Tôi là'). Two competing explanations:

  L. LEARNING: the readout never assigned mass to the later tokens, so no
     decoder could recover them.
  D. DECODER: the readout DOES assign the correct next token, but generate()
     stops for a structural reason (mask, EOS handling, position flag).

Discriminate by inspecting the full probability vector at the exact position
the product stopped, on the same state the product used. If the correct token is
rank 1 and merely masked or suppressed, that is a decoder bug. If EOS genuinely
outranks it, that is a learning/objective issue.

Read-only, on a copy.
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
from scarx.evaluation.product_acceptance import copy_session  # noqa: E402

CORPUS = json.loads((ROOT / 'data/dialogue/generation1.json').read_text(encoding='utf-8'))

work = copy_session(ROOT / 'local' / 'generation1')
app = Generation(work, background=False)
try:
    probe = app.model
    vocab = probe.vocabulary
    rows = []
    for fam in CORPUS['families']:
        prompt, answer = fam['train'][0], fam['answer']
        ctx = probe.context(prompt)
        probe.boundary()
        prev = 1
        emitted = []
        trace = []
        for pos in range(10):
            p = probe.word_step(prev, ctx, pos, source=prompt)
            order = np.argsort(-p)[:3]
            top = int(order[0])
            masked = float(p[top])
            trace.append({
                'pos': pos,
                'top': vocab[top],
                'top_p': masked,
                'runner': vocab[int(order[1])],
                'runner_p': float(p[int(order[1])]),
                'eos_p': float(p[0]),
            })
            if top == 0:
                break
            emitted.append(vocab[top])
            prev = top

        produced = ' '.join(emitted)
        rows.append({'prompt': prompt, 'answer': answer, 'produced': produced,
                     'exact': produced.strip(' .!?').lower() == answer.strip(' .!?').lower(),
                     'trace': trace})

    print('=== where does generation stop, and what was available? ===')
    for r in rows:
        print(f"\nprompt {r['prompt']!r}")
        print(f"  wanted : {r['answer']!r}")
        print(f"  got    : {r['produced']!r}   exact={r['exact']}")
        for t in r['trace']:
            print(f"    p{t['pos']}: top={t['top']!r} ({t['top_p']:.3f})  "
                  f"runner={t['runner']!r} ({t['runner_p']:.3f})  eos={t['eos_p']:.3f}")

    # Classify each stop
    print('\n=== classification of the stopping step ===')
    summary = {'decoder_suspect': 0, 'learning_suspect': 0, 'ok': 0}
    for r in rows:
        if r['exact']:
            summary['ok'] += 1
            print(f"  OK        {r['prompt']!r}")
            continue
        last = r['trace'][-1]
        # If EOS is the top token and it beat the correct continuation, the
        # readout itself prefers stopping: learning/objective, not the mask.
        if last['top'] == '<eos>':
            summary['learning_suspect'] += 1
            print(f"  LEARNING  {r['prompt']!r}: stopped on EOS at p{last['pos']} "
                  f"(eos_p={last['eos_p']:.3f}); the readout chose to stop.")
        else:
            summary['decoder_suspect'] += 1
            print(f"  DECODER   {r['prompt']!r}: emitted {last['top']!r} but wanted more.")

    print('\nsummary:', summary)
    out = ROOT / 'local' / 'decoder-vs-learning.json'
    out.write_text(json.dumps({'rows': rows, 'summary': summary,
                               'scope': 'Read-only separation of decoder vs learning.'},
                              ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {out}')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
