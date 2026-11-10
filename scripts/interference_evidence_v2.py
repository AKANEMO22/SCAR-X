"""Interference evidence, hardened per the user's review of round 24.

The round-24 result (|dw_out| >> |dw_in|) was treated as insufficient on its own:
norm ratio is not contribution, and one lesson ('hi') cannot be generalised.
This script fixes five specific weaknesses:

  1. distinguishes ALREADY-FAILING baseline pairs from NEWLY LOST ones
  2. measures w_out, bias, w_in, w_rec AND spike features AND logit margins
  3. controlled intervention: transplant ONLY readout, ONLY bias, ONLY hidden,
     keeping the rest fixed, to see which transplant reproduces the loss
  4. repeats over several lessons from copies of the current session. It does
     NOT yet replicate across independent seeds or multiple fixed checkpoints.
  5. never uses LR ratio or norm alone as the causal claim

Read-only except inside throwaway copies.
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
from scarx.neural.unified import UnifiedCore  # noqa: E402
from scarx.evaluation.parameter_interventions import transplant_arrays
from scarx.evaluation.product_acceptance import copy_session

LIVE = ROOT / 'local' / 'generation1'

# Several lessons, each a different pair, so nothing rests on 'hi'.
LESSONS = [
    ('hi', 'Hello.'),
    ('cảm ơn', 'Rất vui được giúp bạn.'),
    ('chào', 'Chào bạn.'),
    ('bạn khỏe không', 'Tôi đang hoạt động.'),
]
WATCH = ['hi', 'hello', 'chào', 'chào bạn', 'bạn là ai', 'cảm ơn', 'tạm biệt',
         'bạn khỏe không', 'tôi muốn dạy bạn']


def reset_read(model, prompt):
    c = UnifiedCore.loads(model.dumps())
    c.core.reset_state()
    c.last_active[:] = c.ticks
    out = c.generate(prompt)['text']
    return out


def margins(model, prompt):
    """Top-2 logit margin at the first decision — a continuous signal, not norm."""
    c = UnifiedCore.loads(model.dumps())
    c.core.reset_state()
    c.last_active[:] = c.ticks
    ctx = c.context(prompt)
    c.boundary()
    p = c.word_step(1, ctx, 0, source=prompt)
    order = np.argsort(-p)[:2]
    return float(p[order[0]] - p[order[1]]), c.vocabulary[int(order[0])]


def judge(model, prompt, target):
    out = reset_read(model, prompt)
    v = Generation._pair_verdict(out, target)
    return out, v


def snap(app):
    return {
        'w_out': app.model.core.w_out.copy(),
        'bias': app.model.core.bias.copy(),
        'w_in': app.model.core.w_in.copy(),
        'w_rec': app.model.core.w_rec.copy(),
        'features': app.model.core.features.copy(),
    }


def new_session():
    work = copy_session(LIVE)
    return work, Generation(work, background=False)


def transplant(dst_model, src_snap, parts):
    """Copy selected parts from src_snap into dst_model — controlled intervention."""
    return transplant_arrays(dst_model, src_snap, parts)


report = []
for lesson_prompt, lesson_answer in LESSONS:
    work, app = new_session()
    try:
        pairs = dict(app._guard_pairs(lesson_prompt))
        base = snap(app)
        base_verdict = {}
        base_margin = {}
        for p in WATCH:
            if p not in pairs:
                continue
            out, v = judge(app.model, p, pairs[p])
            base_verdict[p] = (out, v['content_ok'])
            base_margin[p] = margins(app.model, p)[0]

        # One lesson, enough steps to reproduce the loss seen in round 24.
        m = app.model
        m.expand_vocabulary(lesson_answer)
        for _ in range(6):
            m.train_pair(lesson_prompt, lesson_answer)
        after = snap(app)

        newly_lost, already_failing, still_ok = [], [], []
        after_verdict = {}
        for p in base_verdict:
            out, v = judge(m, p, pairs[p])
            after_verdict[p] = (out, v['content_ok'])
            was = base_verdict[p][1]
            if was and not v['content_ok']:
                newly_lost.append(p)
            elif not was and not v['content_ok']:
                already_failing.append(p)
            else:
                still_ok.append(p)

        # Controlled transplant: start from the BASE model, apply only parts.
        interventions = {}
        for name, parts in (('readout_only', ('w_out',)),
                            ('bias_only', ('bias',)),
                            ('hidden_only', ('w_in', 'w_rec')),
                            ('readout+bias', ('w_out', 'bias'))):
            w2, app2 = new_session()
            try:
                # rebuild base weights exactly, then transplant the trained parts
                transplant(app2.model, base, ('w_out', 'bias', 'w_in', 'w_rec'))
                transplant(app2.model, after, parts)
                lost = []
                for p in base_verdict:
                    if not base_verdict[p][1]:
                        continue
                    _, v2 = judge(app2.model, p, pairs[p])
                    if not v2['content_ok']:
                        lost.append(p)
                interventions[name] = lost
            finally:
                app2.close()
                shutil.rmtree(w2.parent, ignore_errors=True)

        report.append({
            'lesson': lesson_prompt, 'answer': lesson_answer,
            'baseline_failing': already_failing,
            'newly_lost': newly_lost,
            'still_ok': still_ok,
            'interventions': interventions,
            'margin_delta': {p: round(margins(app.model, p)[0] - base_margin[p], 4)
                             for p in base_margin if p in base_margin},
        })
        print(f'\nLESSON {lesson_prompt!r} -> {lesson_answer!r}')
        print(f'  baseline already failing : {already_failing}')
        print(f'  NEWLY LOST               : {newly_lost}')
        print(f'  still ok                 : {len(still_ok)}')
        for name, lost in interventions.items():
            print(f'    transplant {name:14s} reproduces loss: {lost}')
    finally:
        app.close()
        shutil.rmtree(work.parent, ignore_errors=True)

print('\n' + '=' * 66)
print('CAUSE TALLY across lessons (newly-lost pairs only)')
tally = {}
for entry in report:
    for name, lost in entry['interventions'].items():
        tally[name] = tally.get(name, 0) + len(lost)
for name, n in sorted(tally.items(), key=lambda x: -x[1]):
    print(f'  {name:16s} reproduced {n} newly-lost pair(s)')

out = ROOT / 'local' / 'interference-evidence-v2.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
print(f'\nwrote {out}')
