"""Round 5B — independent key/value generator.

WHY THIS ROUND EXISTS
---------------------
R4A used `values = (stored, 1 - stored)`, i.e. the two values were COMPLEMENTARY.
That reduces the whole task to one stored bit and makes the label `stored XOR query`.
The R4A numbers therefore say nothing about whether ALIF can hold two INDEPENDENT
facts, nor whether a readout can select between them.

This generator removes the constraint: vA and vB are drawn independently from
{0,1}, so all four memory states (00, 01, 10, 11) are legal and equally likely.

THE BASELINE CONSEQUENCE (this is the crux of the round)
--------------------------------------------------------
With independent values, 50% is NO LONGER the right null:

  * majority class          -> 0.50
  * query-only              -> 0.50
  * always answer key A     -> 0.75   (right when q=A, plus when vB=vA and q=B)
  * know both values, no query -> 0.75 (right when vA=vB, else a coin flip)

A learner at 75% may simply be answering "vA" every time. Every claim in this round
is therefore measured against the 0.75 shortcut ceiling, never against 0.50.

CHANNELS (width 8, 12 ticks, same as R4A for a fair comparison)
    ch0  key A marker
    ch1  key B marker
    ch2  value-0 marker
    ch3  value-1 marker
    ch4  query = A marker
    ch5  query = B marker
    ch6,7 background (unused by the oracle)
Writes at ticks 2 and 6 in random key order; query at tick 10.

The oracle reads ONLY the specification markers. It never sees a label, and neither
does any feature map.
"""
import hashlib

import numpy as np

PULSE = 1.5
WRITE_TICKS = (2, 6)
QUERY_TICK = 10
N_TICKS = 12
N_CHANNELS = 8


def independent_query_set_v1(seed, count):
    """Two independent values, uniform query, uniform 4-way value pattern.

    Returns
    -------
    x       : (count, 12, 8) float32
    y       : (count,) int32, the value of the queried key
    vA, vB  : (count,) int32, the two stored values (for analysis, never for features)
    q       : (count,) int32, 0 if key A was queried, 1 if key B
    pid     : (count,) int32, counterfactual pair id (-1 if unpaired)
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, N_TICKS, N_CHANNELS)).astype(np.float32)
    vA = rng.integers(0, 2, count).astype(np.int32)
    vB = rng.integers(0, 2, count).astype(np.int32)
    q = rng.integers(0, 2, count).astype(np.int32)

    for i in range(count):
        values = {0: int(vA[i]), 1: int(vB[i])}
        for slot, key in zip(WRITE_TICKS, rng.permutation(2)):
            k = int(key)
            x[i, slot, k] += PULSE                       # key marker, ch0 or ch1
            x[i, slot, 2 + values[k]] += PULSE           # value marker, ch2 or ch3
        x[i, QUERY_TICK, 4 + int(q[i])] += PULSE         # query marker, ch4 or ch5

    y = np.where(q == 0, vA, vB).astype(np.int32)
    pid = np.full(count, -1, np.int32)
    return x, y, vA, vB, q, pid


def independent_paired_set_v1(seed, n_pairs):
    """Counterfactual siblings: identical facts, noise and timing; ONLY the query differs.

    Because the values are independent, a pair falls into one of two cases:
      * vA != vB  -> the query swap MUST flip the answer   ("case A")
      * vA == vB  -> the query swap MUST NOT flip the answer ("case B")

    Both cases are returned and must be scored separately. Case B must never be
    scored as a paired-flip failure.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((2 * n_pairs, N_TICKS, N_CHANNELS), np.float32)
    y = np.zeros(2 * n_pairs, np.int32)
    vA = np.zeros(2 * n_pairs, np.int32)
    vB = np.zeros(2 * n_pairs, np.int32)
    q = np.zeros(2 * n_pairs, np.int32)
    pid = np.repeat(np.arange(n_pairs), 2).astype(np.int32)
    flips = np.zeros(n_pairs, bool)      # True when the pair is "case A"

    for p in range(n_pairs):
        a = int(rng.integers(2))
        b = int(rng.integers(2))
        bg = rng.normal(0, .08, (N_TICKS, N_CHANNELS)).astype(np.float32)
        order = rng.permutation(2)
        flips[p] = a != b
        for slot, query in ((0, 0), (1, 1)):
            idx = 2 * p + slot
            x[idx] = bg
            values = {0: a, 1: b}
            for place, key in zip(WRITE_TICKS, order):
                k = int(key)
                x[idx, place, k] += PULSE
                x[idx, place, 2 + values[k]] += PULSE
            x[idx, QUERY_TICK, 4 + query] += PULSE
            y[idx] = values[query]
            vA[idx], vB[idx], q[idx] = a, b, query
    return x, y, vA, vB, q, pid, flips


def oracle_answer(x_row):
    """Deterministic oracle: read the spec markers wherever they are.

    It LOCATES the markers rather than assuming fixed ticks, so it stays correct
    under a temporal shift. This matters: an oracle pinned to ticks 2/6/10 would
    report "malformed" for every shifted episode and a shifted experiment would look
    broken when it is only mis-instrumented.
    """
    x_row = np.asarray(x_row, np.float32)
    write_ticks = [t for t in range(x_row.shape[0])
                   if np.any(x_row[t, :2] > .5)]
    query_ticks = [t for t in range(x_row.shape[0])
                   if np.any(x_row[t, 4:6] > .5)]
    if len(write_ticks) != 2:
        raise ValueError(f"expected 2 writes, found {len(write_ticks)}")
    if len(query_ticks) != 1:
        raise ValueError(f"expected 1 query, found {len(query_ticks)}")
    vals = {}
    for t in write_ticks:
        keys = np.flatnonzero(x_row[t, :2] > .5)
        vs = np.flatnonzero(x_row[t, 2:4] > .5)
        if len(keys) != 1 or len(vs) != 1:
            raise ValueError(f"malformed write at tick {t}")
        vals[int(keys[0])] = int(vs[0])
    if len(vals) != 2:
        raise ValueError("both keys must be written")
    q = np.flatnonzero(x_row[query_ticks[0], 4:6] > .5)
    return vals[int(q[0])]


def locate_query_tick(x_row):
    """The tick where this episode's query marker sits (shift-safe)."""
    x_row = np.asarray(x_row, np.float32)
    ticks = np.flatnonzero(np.any(x_row[:, 4:6] > .5, axis=1))
    if len(ticks) != 1:
        raise ValueError(f"expected 1 query tick, found {len(ticks)}")
    return int(ticks[0])


def oracle_all(x):
    return np.array([oracle_answer(r) for r in x], np.int32)


def query_cue(x, tick=None):
    """The legal query cue taken from the INPUT, at the tick this data uses.

    `tick=None` locates the query marker per-episode, which is required under a
    temporal shift. Passing a fixed tick is allowed but must match the data.
    """
    x = np.asarray(x, np.float32)
    if tick is None:
        ticks = np.flatnonzero(np.any(x[:, :, 4:6] > .5, axis=2).any(axis=0))
        if len(ticks) != 1:
            raise ValueError(f"expected one query tick, found {list(ticks)}")
        tick = int(ticks[0])
    q = (x[:, tick, 5] > .5).astype(np.float32)
    return np.stack([1 - q, q], 1)


def dataset_sha256(x, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, np.float32).tobytes())
    h.update(np.ascontiguousarray(y, np.int32).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- baselines
def baseline_first_key(x):
    """Always answer the value bound to key A.

    Expected 0.75 with independent values: correct when q=A, and also when q=B and
    vB == vA. This is the shortcut a 75% score cannot rule out.

    Write ticks are LOCATED, not assumed, so the baseline stays valid on shifted
    data. A baseline that breaks under the very transform being tested would make
    the comparison meaningless.
    """
    x = np.asarray(x, np.float32)
    out = np.zeros(len(x), np.int32)
    for i, row in enumerate(x):
        write_ticks = [t for t in range(row.shape[0]) if np.any(row[t, :2] > .5)]
        vals = {}
        for t in write_ticks:
            k = int(np.argmax(row[t, :2]))
            v = int(np.argmax(row[t, 2:4]))
            vals[k] = v
        if 0 not in vals:
            raise ValueError("key A was never written")
        out[i] = vals[0]
    return out


def baseline_majority(n, value=0):
    return np.full(n, value, np.int32)


def baseline_query_only(x):
    """Predict from the query cue alone. Both classes are equally likely, so 0.50."""
    x = np.asarray(x, np.float32)
    return (x[:, QUERY_TICK, 5] > .5).astype(np.int32)


# ------------------------------------------------------------- transforms
def shifted(x, shift=1, seed=None):
    """Move every event `shift` ticks later. Labels are unchanged.

    IMPORTANT NEGATIVE RESULT SURFACED IN STAGE 4
    ---------------------------------------------
    The obvious implementation pads the vacated leading ticks with ZEROS. For a
    zero-initialised recurrent network that is a NO-OP: `v = 0.65*0 + 0 = 0`,
    `z = 0`, so the whole state stays at its initial value and every subsequent tick
    is bit-identical to the unshifted run. Verified: the post-query state under the
    zero-pad shift has `max |diff| = 0.0` against clean.

    A transpose that leaves the network's computation unchanged cannot test temporal
    robustness, so this function pads with FRESH NOISE instead. The background at the
    vacated ticks is redrawn from the same distribution; the event pulses and all
    other background samples are preserved exactly. That keeps the shift a genuine
    change of timing with a minimal, disclosed noise change confined to the vacated
    ticks. Callers that need a pure-noise condition should use `noisier()` instead.
    """
    x = np.asarray(x, np.float32)
    if shift <= 0:
        return x.copy()
    out = np.zeros_like(x)
    for t in range(N_TICKS):
        if t + shift < N_TICKS:
            out[:, t + shift, :] = x[:, t, :]
    rng = np.random.default_rng(0 if seed is None else seed + 31_337)
    pad = rng.normal(0, .08, (len(x), shift, N_CHANNELS)).astype(np.float32)
    out[:, :shift, :] = pad
    return out


def shifted_zero_pad(x, shift=1):
    """The naive zero-pad shift, kept ONLY to demonstrate that it is a no-op.

    Exposed so the audit can assert the no-op property rather than take it on trust.
    """
    x = np.asarray(x, np.float32)
    out = np.zeros_like(x)
    for t in range(N_TICKS):
        if t + shift < N_TICKS:
            out[:, t + shift, :] = x[:, t, :]
    return out


def reversed_writes(x):
    """Swap the two write ticks. Labels must be unchanged."""
    x = np.asarray(x, np.float32).copy()
    x[:, list(WRITE_TICKS)] = x[:, list(WRITE_TICKS)[::-1]]
    return x


def with_distractors(x, seed, n=2):
    """Extra pulses on background channels at unused ticks. Labels unchanged."""
    rng = np.random.default_rng(seed)
    out = np.asarray(x, np.float32).copy()
    used = set(WRITE_TICKS) | {QUERY_TICK}
    free = [t for t in range(N_TICKS) if t not in used]
    for i in range(len(out)):
        for t in rng.choice(free, min(n, len(free)), replace=False):
            out[i, int(t), int(rng.integers(6, 8))] += PULSE
    return out


def noisier(x, seed, sigma=.08):
    """Re-draw the background at a different sigma, keeping the event pulses."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float32)
    events = np.zeros_like(x)
    for place in WRITE_TICKS:
        events[:, place, :4] = np.where(x[:, place, :4] > .5, PULSE, 0)
    events[:, QUERY_TICK, 4:6] = np.where(x[:, QUERY_TICK, 4:6] > .5, PULSE, 0)
    return (rng.normal(0, sigma, x.shape).astype(np.float32) + events)
