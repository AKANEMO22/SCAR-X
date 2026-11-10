"""Is full neural-state rollback correct? Isolated test, no idle, no stray generate.

The user asked for an isolated experiment comparing every piece of state across a
rejected attempt, because earlier tests let idle ticks and generate() calls run in
between and could not separate "rollback is broken" from "the world moved".

This drives _online_microstep directly on a copy, with background=False so no idle
thread exists, and compares before/after:

  w_in, w_rec, w_out, bias, stable_*   weights
  voltage, adaptation, spikes          neuron state
  pre, post, features                  traces
  elig_in, elig_rec, rows              eligibility
  indices                              topology
  vocabulary                           vocabulary
  RNG states (sampler, calibration)    randomness
  ticks, updates, counters             clocks
  behaviour from an identical start    outputs

It forces a rejection by injecting a guard verdict, so the rollback path is
exercised deterministically.
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
from scarx.app.generation import Generation, LearningGuardRejected  # noqa: E402
from scarx.evaluation.product_acceptance import copy_session, weights_sha256  # noqa: E402
from scarx.neural.unified import UnifiedCore  # noqa: E402

ARRAYS = ('w_in', 'w_rec', 'w_out', 'bias', 'stable_in', 'stable_rec', 'stable_out', 'stable_bias',
          'voltage', 'adaptation', 'spikes', 'pre', 'post', 'features',
          'elig_in', 'elig_rec', 'rows', 'indices')
COUNTERS = ('updates',)
SLOTS = ('prepared', 'previous', 'position', 'responses_seen', 'active_count')

PROBE = ['hi', 'chào bạn', 'bạn là ai', 'tạm biệt']


def state_of(app):
    m = app.model
    core = m.core
    snap = {}
    for name in ARRAYS:
        arr = getattr(core, name, None)
        if arr is None:
            continue
        snap[name] = np.ascontiguousarray(arr).tobytes()
    for name in COUNTERS:
        snap[name] = getattr(core, name, None)
    for name in SLOTS:
        snap[name] = getattr(m, name, None)
    snap['vocabulary'] = tuple(m.vocabulary)
    snap['ticks'] = m.ticks
    snap['capacity_rng'] = json.dumps(m.capacity_rng.bit_generator.state, sort_keys=True)
    snap['app_rng'] = json.dumps(app.rng.bit_generator.state, sort_keys=True)
    snap['probe'] = json.dumps(app.probe, sort_keys=True)
    snap['replay_cursor'] = app.replay_cursor
    snap['sequence'] = app.sequence
    snap['memories'] = json.dumps(app.memories, sort_keys=True)
    snap['dialogue'] = json.dumps(app.dialogue.dumps(), sort_keys=True)
    return snap


def read_from_fresh(model, prompt):
    c = UnifiedCore.loads(model.dumps())
    c.core.reset_state()
    c.last_active[:] = c.ticks
    return c.generate(prompt)['text']


work = copy_session(ROOT / 'local' / 'generation1')
# background=False -> no idle thread can tick during the experiment
app = Generation(work, background=False)
try:
    before = state_of(app)
    sha_before = weights_sha256(app.model)
    outs_before = {p: read_from_fresh(app.model, p) for p in PROBE}

    # Queue a lesson and force a rejection with an injected guard verdict.
    app.dialogue.enqueue('probe', 'chào bạn', 'Chào bạn.', 'isolation test')
    lesson = app.dialogue.lessons[-1]
    lesson['remaining'] = 1
    import itertools
    verdicts = itertools.cycle([[True], [False]])
    original_pairs = app._guard_pairs
    app._guard_pairs = lambda _p: [('probe skill', 'target')]
    app._guard_scores = lambda *_a: next(verdicts)
    try:
        record = app._online_microstep('probe')
    finally:
        app._guard_pairs = original_pairs
        del app._guard_scores

    after = state_of(app)
    sha_after = weights_sha256(app.model)
    outs_after = {p: read_from_fresh(app.model, p) for p in PROBE}

    print(f"rejected      : {not record.get('accepted')}")
    print(f"rolled_back   : {record.get('rolled_back')}")
    print(f"guard_lost    : {record.get('guard_lost')}")
    print(f"weights sha   : {sha_before[:16]} -> {sha_after[:16]}  same={sha_before == sha_after}")
    print()

    print('=== per-component comparison ===')
    diffs = []
    for key in before:
        if key == 'dialogue':
            # Dialogue counters are expected to change: a rejected attempt still
            # records that an attempt happened.
            same = before[key] == after[key]
            print(f"  {'same' if same else 'CHANGED':8s} {key}   (counter bookkeeping)")
            continue
        same = before[key] == after[key]
        if not same:
            diffs.append(key)
        print(f"  {'same' if same else 'CHANGED':8s} {key}")

    print()
    print('=== behaviour from an identical start ===')
    for p in PROBE:
        same = outs_before[p] == outs_after[p]
        print(f"  {'same' if same else 'CHANGED':8s} {p!r:14s} {outs_before[p]!r} -> {outs_after[p]!r}")
        if not same:
            diffs.append(f'output:{p}')

    print()
    if diffs:
        print(f'ROLLBACK NOT EXACT. Differing: {diffs}')
    else:
        print('ROLLBACK EXACT for every weight, neuron state, trace, eligibility,')
        print('topology, vocabulary, RNG, clock and behaviour. No bug to fix.')

    out = ROOT / 'local' / 'rollback-isolation.json'
    out.write_text(json.dumps({'differing': diffs,
                               'sha_before': sha_before, 'sha_after': sha_after,
                               'record_keys': sorted(record.keys())},
                              ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nwrote {out}')
finally:
    app.close()
    shutil.rmtree(work.parent, ignore_errors=True)
