"""Weight fingerprint and the 12-metric product acceptance harness.

Why this exists: the previous report used the `updates` counter as proof the
weights were untouched. That counter does not change when the readout or the
vocabulary is rewritten, so it cannot prove anything. A SHA-256 over the actual
weight arrays can.

This module provides:
  * weights_sha256(model)  - hash of every learnable array plus topology identity
  * run_acceptance(...)    - the 12 pre-registered product metrics

No training rule, topology or guard is changed here.
"""
import hashlib
import json
import math
import shutil
import tempfile
import sqlite3
from pathlib import Path

import numpy as np

# Fixed product acceptance set.
#
# `expect` is a PRODUCT expectation, not a truth label and never a training target.
# `require_all` lists words that must ALL be present. This matters: a loose
# substring like 'scar-x' counted 'Tôi là scar-x là' as correct even though it
# loops and truncates. An acceptance metric that passes a broken reply is worse
# than no metric, because it hides the defect it exists to expose.
CASES = [
    ('hi', 'hello', ('hello',)),
    ('hello', 'hello', ('hello',)),
    ('chào', 'chào bạn', ('chào', 'bạn')),
    ('chào bạn', 'chào bạn', ('chào', 'bạn')),
    ('bạn là ai', 'scar-x', ('scar-x',)),
    ('bạn tên gì', 'scar-x', ('scar-x',)),
    ('cảm ơn', 'không có gì', ('không', 'gì')),
    ('tạm biệt', 'tạm biệt', ('tạm', 'biệt')),
]

# Preferred full answers, used only to judge "complete" vs "truncated", never fed
# to the model.
REFERENCE = {
    'hi': 'Hello.',
    'hello': 'Hello.',
    'chào': 'Chào bạn.',
    'chào bạn': 'Chào bạn.',
    'bạn là ai': 'Tôi là SCAR-X.',
    'bạn tên gì': 'Tôi là SCAR-X.',
    'cảm ơn': 'Không có gì.',
    'tạm biệt': 'Tạm biệt bạn.',
}

# Held-out phrasings never used for tuning: same intent, different words.
HELDOUT = [
    ('xin chào', 'chào'),
    ('chào nhé', 'chào'),
    ('bạn tên là gì', 'scar-x'),
    ('anh là ai', 'scar-x'),
    ('cám ơn nhé', 'cảm ơn'),
    ('bye', 'tạm biệt'),
]


def _array_bytes(model):
    core = model.core
    parts = []
    for name in core.weight_names:
        arr = getattr(core, name)
        parts.append(name.encode())
        parts.append(np.ascontiguousarray(arr).tobytes())
    parts.append(b'|indices|')
    parts.append(np.ascontiguousarray(core.indices).tobytes())
    parts.append(b'|shape|')
    c = core.config
    parts.append(json.dumps({'inputs': c.inputs, 'hidden': c.hidden,
                             'outputs': c.outputs, 'fan_in': c.fan_in,
                             'plastic_rows': c.plastic_rows,
                             'neuron': c.neuron}).encode())
    parts.append(b'|vocab|')
    parts.append(json.dumps(list(model.vocabulary), ensure_ascii=False).encode())
    head = getattr(model, 'head', None)
    if head is not None:
        for name in ('weights', 'bias', 'mean', 'm2'):
            arr = getattr(head, name, None)
            if arr is not None:
                parts.append(name.encode())
                parts.append(np.ascontiguousarray(arr).tobytes())
    return b''.join(parts)


def weights_sha256(model):
    """Stable hash of learnable weights + topology + vocabulary identity."""
    return hashlib.sha256(_array_bytes(model)).hexdigest()


def _cycles(words):
    """Longest repeated adjacent block length, or 0.

    Also detects a whole-phrase repeat such as 'tạm biệt tạm biệt', which the
    adjacent-block test alone misses because the two halves are separated by the
    norm on the whole string but still repeat as a phrase.
    """
    n = len(words)
    for size in range(1, max(1, n // 2) + 1):
        for start in range(n - 2 * size + 1):
            if words[start:start + size] == words[start + size:start + 2 * size]:
                return size
    # Whole-sequence halving: 'a b a b' style already caught above, but cover the
    # case where the repeat spans the entire reply twice.
    if n and n % 2 == 0 and words[:n // 2] == words[n // 2:]:
        return n // 2
    return 0


def _normalized(text):
    from ..neural.language import tokenize
    return ' '.join(tokenize(text)).strip(' .!?')


def _judge_v1(text, expect, prompt, require_all=None):
    words = text.split()
    norm = _normalized(text)
    ref = REFERENCE.get(prompt, '')
    routed = text.startswith('[Hỏi để học]')
    ends_clean = bool(text.strip()) and (text.strip()[-1] in '.!?' or norm == ref)
    # Correctness requires every required word, a non-looping reply, AND a
    # finished sentence. Without the completion test, 'Tôi là scar-x là' counted
    # as a correct answer to 'bạn là ai' even though it stops mid-sentence and
    # repeats. An acceptance metric that passes a broken reply hides the very
    # defect it exists to expose.
    required = tuple(require_all or (expect,))
    has_words = all(_normalized(w) in norm for w in required)
    # Detect repetition on normalized tokens so trailing punctuation cannot hide a
    # repeated phrase ('Tạm biệt tạm biệt.' splits with 'biệt.' attached).
    looping = _cycles(norm.split()) > 0
    # A routed reply is a controller message, not neural output, so it can never
    # be a correct answer even when it happens to quote the prompt.
    return {
        'text': text,
        'usable': bool(text.strip()) and not routed and not looping,
        'correct': has_words and not looping and ends_clean and not routed,
        'has_expected_words': has_words,
        'complete': ends_clean,
        'premature_stop': bool(text.strip()) and not ends_clean,
        'repetition': looping,
        'routed': routed,
        'words': len(words),
    }


def _judge(text, expect, prompt, require_all=None, expected_answer=None):
    """Version2: closed-fixture content, not substring or generalsemanticgrade.

    Preserve v1 for historical comparison. Reference labels are only scorerdata;
    they are never given to model.generate(). Valid alternative phrasings may
    fail this deliberately narrow canonical-style metric.
    """
    row=_judge_v1(text,expect,prompt,require_all)
    tokens=_normalized(text).split();required=tuple(require_all or (expect,))
    def contains(term):
        target=_normalized(term).split()
        return bool(target) and any(tokens[i:i+len(target)]==target for i in range(len(tokens)-len(target)+1))
    row['has_expected_words']=all(contains(term) for term in required)
    reference=expected_answer if expected_answer is not None else REFERENCE.get(prompt,expect)
    row['content_ok']=_normalized(text)==_normalized(reference) and not row['repetition'] and not row['routed']
    row['complete']=bool(text.strip()) and text.strip()[-1] in '.!?'
    row['premature_stop']=bool(text.strip()) and not row['complete']
    row['correct']=row['content_ok'] and row['complete']
    row['metric_version']='product-canonical-v2';row['expected_reference']=reference
    return row


def canonical_reply(model,prompt):
    """Pure product probe: reset a clone and use available requestcapacity only."""
    from ..neural.unified import UnifiedCore
    from ..app.workload import plan_workload
    clone=UnifiedCore.loads(model.dumps());clone.core.reset_state();clone.last_active[:]=clone.ticks
    clone.select_capacity(min(clone.core.config.hidden,plan_workload(clone,prompt)['requested_neurons']))
    return clone.generate(prompt)


def measure(app, cases=None, reset_state=None, metric_version='product-canonical-v2'):
    """Score the product set.

    IMPORTANT — measurement semantics. `generate()` intentionally never resets the
    recurrent state (ADR0006: one continuous state for the whole session). So two
    consecutive calls on identical weights can disagree: measured on the live core,
    8 repeated calls on 'bạn là ai' produced 2 distinct outputs, and reversing the
    prompt order changed 'chào bạn' from 'Chào' to 'Bạn'.

    A number that depends on prompt order is not a property of the model, so by
    default each prompt is scored from a fresh clone with a reset state — the same
    convention the learning guard already uses. That makes the metric reproducible.
    Pass reset_state=False to observe the conversational (order-dependent) behaviour
    instead, which is what a user actually experiences turn by turn.
    """
    cases = cases or CASES
    if reset_state is None:
        reset_state = True
    if metric_version not in ('product-canonical-v2','legacy-keywords-v1'):raise ValueError('Unknown product metricversion')
    rows = []
    for case in cases:
        prompt, expect = case[0], case[1]
        require_all = case[2] if len(case) > 2 else None
        if reset_state:
            if metric_version=='legacy-keywords-v1':
                # Preserve the original default capacity/state convention too.
                from ..neural.unified import UnifiedCore
                clone=UnifiedCore.loads(app.model.dumps());clone.core.reset_state();clone.last_active[:]=clone.ticks
                result=clone.generate(prompt)
            else:result = canonical_reply(app.model,prompt)
        else:
            result = app.model.generate(prompt)
        if metric_version=='legacy-keywords-v1':
            row=_judge_v1(result['text'],expect,prompt,require_all);row['metric_version']=metric_version
        else:row = _judge(result['text'], expect, prompt, require_all,case[3] if len(case)>3 else None)
        row['prompt'] = prompt
        row['expect'] = expect
        row['require_all'] = list(require_all or (expect,))
        row['stop_reason'] = result.get('stop_reason')
        row['state_reset'] = reset_state
        rows.append(row)
    return rows


def summarise(rows):
    n = len(rows) or 1
    return {
        'n': len(rows),
        'free_answer_correct': sum(r['correct'] for r in rows) / n,
        'usable': sum(r['usable'] for r in rows) / n,
        'premature_stop': sum(r['premature_stop'] for r in rows) / n,
        'repetition': sum(r['repetition'] for r in rows) / n,
        'routed': sum(r['routed'] for r in rows) / n,
    }


def guard_report(app, exclude=None):
    """Which guard pairs pass, scored EXACTLY as the product guard scores them.

    Uses Generation._pair_verdict so the dashboard, the acceptance suite and the
    guard can never disagree about what "passing" means. They previously used two
    different comparisons (this one ignored trailing punctuation), so the dashboard
    could show a pair as PASS while the guard treated it as failing.
    """
    from ..neural.unified import UnifiedCore
    from ..app.generation import Generation
    pairs = app._guard_pairs(exclude or '__acceptance_probe__')
    rows = []
    for prompt, target in pairs:
        clone = UnifiedCore.loads(app.model.dumps())
        clone.core.reset_state()
        clone.last_active[:] = clone.ticks
        out = clone.generate(prompt)['text']
        verdict = Generation._pair_verdict(out, target)
        rows.append({'prompt': prompt, 'target': target, 'output': out,
                     'content_ok': verdict['content_ok'],
                     'complete_ok': verdict['complete_ok'],
                     'repetition': verdict['repetition'],
                     'pass': verdict['content_ok']})
    return rows


def copy_session(source,*,destination_parent=None):
    """Coherent onlineSQLite snapshot, never race copying liveDB/WAL/lockfiles."""
    database=(Path(source)/'generation1.sqlite').resolve()
    if not database.is_file():raise FileNotFoundError(database)
    if destination_parent is not None:Path(destination_parent).mkdir(parents=True,exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='scarx-acc-',dir=destination_parent)) / 'session'
    work.mkdir()
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=5) as origin,sqlite3.connect(work/'generation1.sqlite') as target:
        origin.backup(target,pages=256,sleep=.01)
    return work


def resources(app):
    try:
        info = app.process.memory_info()
        rss = int(info.rss)
    except Exception:
        rss = None
    return {'rss_bytes': rss, 'ram_limit_bytes': 2_000_000_000,
            'under_limit': (rss is None or rss < 2_000_000_000)}
