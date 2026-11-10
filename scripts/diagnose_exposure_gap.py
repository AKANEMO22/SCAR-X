"""Why does the readout emit EOS mid-answer when the corpus answer continues?

The corpus contains 'Chào bạn.' but the model emits 'chào' then EOS at p=0.874.
Two candidate mechanisms:
  M1 POSITIONAL: the position flag (pos/24) tells the model "this is the end",
      because training sequences were short and position encodes length.
  M2 STATE:     the recurrent state after the first word no longer carries the
      prompt, so the readout falls back to a short generic continuation.

Discriminate by holding the prompt fixed and forcing the CORRECT first token,
then checking whether the next token is right. That is teacher forcing on the
same live weights: if the answer becomes correct, the model knows the answer and
only the free-running state is at fault (decoder/state), not the mapping.

Read-only, on a copy.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np  # noqa: E402
from scarx.app.generation import Generation  # noqa: E402
from scarx.evaluation.product_acceptance import copy_session  # noqa: E402
from scarx.neural.language import tokenize  # noqa: E402

CORPUS = json.loads((ROOT / 'data/dialogue/generation1.json').read_text(encoding='utf-8'))

work = copy_session(ROOT / 'local' / 'generation1')
app = Generation(work, background=False)
try:
    m = app.model
    vocab = m.vocabulary
    print(f'{"prompt":22s} {"free":26s} {"gold-prefix":30s} verdict')
    results = []
    for fam in CORPUS['families']:
        prompt, answer = fam['train'][0], fam['answer']
        gold = [vocab.index(t) for t in tokenize(answer) if t in m.ids]

        # Free run
        ctx = m.context(prompt); m.boundary(); prev = 1; free = []
        for pos in range(9):
            p = m.word_step(prev, ctx, pos, source=prompt)
            tok = int(np.argmax(p))
            if tok == 0:
                break
            free.append(vocab[tok]); prev = tok
        free_txt = ' '.join(free)

        # Gold prefix: same prompt, forced previous tokens, fresh settle.
        # Record TWO different things:
        #   img_ok    - the full gold sequence is reproduced while being forced
        #               (i.e. the mapping exists when the prefix is correct)
        #   agent_ok  - every gold token was the model's own argmax at that step
        #               (i.e. it would have produced the answer without help)
        # Conflating these two produced a wrong earlier conclusion.
        ctx2 = m.context(prompt); m.boundary(); prev2 = 1; forced = []
        all_argmax = True
        step_matches = []
        for pos, g in enumerate(gold):
            p = m.word_step(prev2, ctx2, pos, source=prompt)
            step_matches.append(int(np.argmax(p)) == g)
            all_argmax = all_argmax and step_matches[-1]
            forced.append(vocab[g])
            prev2 = g
            if pos >= len(gold) - 1:
                break
        forced_txt = ' '.join(forced)

        free_ok = free_txt.strip(' .!?').lower() == answer.strip(' .!?').lower()
        # img_ok uses the exact gold tokens we forced, so it only tells us the
        # gold sequence is well-formed, NOT that the model agrees.
        gold_ok = all_argmax
        verdict = ('both_ok' if free_ok and gold_ok else
                   'goldprefix_ok' if gold_ok else 'neither')
        print(f'{prompt[:20]!r:22s} {free_txt[:24]!r:26s} {forced_txt[:28]!r:30s} '
              f'{"+".join("1" if s else "0" for s in step_matches):12s} {verdict}')
        results.append({'prompt': prompt, 'answer': answer, 'free': free_txt,
                        'gold_prefix': forced_txt, 'free_ok': free_ok,
                        'gold_prefix_ok': gold_ok, 'all_gold_argmax': all_argmax,
                        'step_matches': step_matches,
                        'verdict': verdict})

    print()
    n = len(results)
    free_n = sum(r['free_ok'] for r in results)
    gold_n = sum(r['all_gold_argmax'] for r in results)
    print(f'free correct            : {free_n}/{n}')
    print(f'gold-prefix argmax      : {gold_n}/{n}')
    print()
    print('NOTE: the "gold-prefix" column shows the tokens that were FORCED, so its')
    print('text always equals the corpus answer. Only the +/0 step column and the')
    print('argmax count say whether the model itself agreed at each step.')
    print()
    if gold_n > free_n:
        print('CONCLUSION: with the prefix supplied the model agrees with far more')
        print('tokens than in free running. The prompt->answer mapping is present;')
        print('the loss is in free-running state/decoding (exposure gap).')
    elif gold_n == free_n:
        print('CONCLUSION: forcing the prefix does not improve agreement, so the')
        print('mapping itself is weak or absent at those steps.')

    out = ROOT / 'local' / 'exposure-gap.json'
    out.write_text(json.dumps({'results': results,
                               'free_ok': sum(r['free_ok'] for r in results),
                               'gold_prefix_ok': sum(r['all_gold_argmax'] for r in results),
                               'scope': 'Teacher-forced diagnostic on live weights; '
                                        'not free generation and not a training run.'},
                              ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nwrote {out}')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
