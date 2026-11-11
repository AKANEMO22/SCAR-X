"""Round 4A canonical relational benchmark — copied from Grok R3B with provenance.

PROVENANCE (do not silently alter semantics)
--------------------------------------------
source branch : research/grok/r3-temporal-stress
source commit : 211380fecdcd05966ec78bb7686f54fa8179c11a
source file   : src/scarx/evaluation/temporal_stress.py
blob sha      : 874fb83507af445ea8ec5db73942fabef03563fa
module sha256 : 260ea4135c85721d5d12461de679948685a87568c18c9976d459b7912dcaabbd
query_set body sha256 : 01b50b4d3af1b763be368c6b93b10ca4ac67d7e5748934a557ba0094a69b8406
PULSE = 1.5

`query_set` below is reproduced verbatim. Any extension is a SEPARATE function
with a distinct name; `query_set` itself is never modified.

Structure of the canonical task (this is why it is hard)
--------------------------------------------------------
    pair = (stored, 1 - stored)      <- the two values are COMPLEMENTARY
    label = pair[query] = stored XOR query

So the stored information is exactly ONE bit, and the label is
`stored XOR query`: an order-3 interaction of (key marker, value marker, query).
A readout that sees only marginally-added features cannot express XOR of two
stored bits with a query, which is the mathematical reason a first-order
substrate is near chance here. That is a property of THIS generator, and it is
recorded so that a low score is not misread as a storage failure.
"""
import hashlib

import numpy as np

PULSE = 1.5
# The verbatim source is stored as a fixture and hashed there, not by
# introspecting the live function: inspect.getsource is formatting-sensitive and
# a docstring or line-wrap change would break a byte hash without changing
# semantics. Equivalence is therefore asserted BEHAVIOURALLY (same arrays for the
# same seeds) plus by the fixture digest.
QUERY_SET_FIXTURE = "data/fixtures/grok_r3b_query_set.py.txt"
QUERY_SET_FIXTURE_SHA256 = "0cac40729010b49619362ecb2302781963193d7f75d89d310aab20f98d691471"
QUERY_SET_SOURCE_BLOB_SHA = "874fb83507af445ea8ec5db73942fabef03563fa"
QUERY_SET_SOURCE_COMMIT = "211380fecdcd05966ec78bb7686f54fa8179c11a"
QUERY_SET_SOURCE_MODULE_SHA256 = "260ea4135c85721d5d12461de679948685a87568c18c9976d459b7912dcaabbd"


def query_set(seed, count):
    """VERBATIM copy of Grok R3B `query_set`. Returns (x, labels, first_key)."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.08, (count, 12, 8)).astype(np.float32)
    labels = np.zeros(count, np.int32)
    first_key = np.zeros(count, np.int32)
    for index in range(count):
        query = index % 2
        stored = int(rng.integers(2))
        pair = (stored, 1 - stored)
        first_key[index] = pair[0]
        labels[index] = pair[query]
        order = rng.permutation(2)
        for place, key in zip((2, 6), order):
            key = int(key)
            values[index, place, key] += PULSE
            values[index, place, 2 + pair[key]] += PULSE
        values[index, 10, 4 + query] += PULSE
    order = rng.permutation(count)
    return values[order], labels[order].astype(np.int32), first_key[order].astype(np.int32)


def independent_query_set(seed, count):
    """Independent re-implementation used only to cross-check `query_set`.

    Written from the specification, not from the code: two complementary values,
    two write slots at t=2 and t=6 in random key order, query at t=10, label is
    the value bound to the queried key.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = np.zeros(count, np.int32)
    fk = np.zeros(count, np.int32)
    for i in range(count):
        q = i % 2
        v0 = int(rng.integers(2))
        vals = np.array([v0, 1 - v0], np.int32)
        fk[i] = vals[0]
        y[i] = vals[q]
        for place, key in zip((2, 6), rng.permutation(2)):
            k = int(key)
            x[i, place, k] += PULSE
            x[i, place, 2 + int(vals[k])] += PULSE
        x[i, 10, 4 + q] += PULSE
    p = rng.permutation(count)
    return x[p], y[p], fk[p]


def dataset_sha256(x, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, np.float32).tobytes())
    h.update(np.ascontiguousarray(y, np.int32).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------ counterfactuals
def paired_counterfactual(seed, count):
    """Two episodes sharing stored facts, timing and noise; only the query differs.

    Because the values are complementary, swapping the query flips the answer,
    so these pairs are the strongest possible test of query use.

    Returns (x, y, pair_id, shares_facts) where pair_id links the two siblings.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((2 * count, 12, 8), np.float32)
    y = np.zeros(2 * count, np.int32)
    pid = np.repeat(np.arange(count), 2)
    for i in range(count):
        v0 = int(rng.integers(2))
        vals = np.array([v0, 1 - v0], np.int32)
        bg = rng.normal(0, .08, (12, 8)).astype(np.float32)
        order = rng.permutation(2)
        for slot, q in ((0, 0), (1, 1)):
            x[i * 2 + slot] = bg
            for place, key in zip((2, 6), order):
                k = int(key)
                x[i * 2 + slot, place, k] += PULSE
                x[i * 2 + slot, place, 2 + int(vals[k])] += PULSE
            x[i * 2 + slot, 10, 4 + q] += PULSE
            y[i * 2 + slot] = vals[q]
    return x, y, pid, True


def equal_value_pairs(seed, count):
    """Equal values: changing the query legitimately does NOT change the answer.

    These must not be scored as paired-flip failures.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((2 * count, 12, 8), np.float32)
    y = np.zeros(2 * count, np.int32)
    pid = np.repeat(np.arange(count), 2)
    for i in range(count):
        v = int(rng.integers(2))
        vals = np.array([v, v], np.int32)      # both keys hold the same value
        bg = rng.normal(0, .08, (12, 8)).astype(np.float32)
        order = rng.permutation(2)
        for slot, q in ((0, 0), (1, 1)):
            x[i * 2 + slot] = bg
            for place, key in zip((2, 6), order):
                k = int(key)
                x[i * 2 + slot, place, k] += PULSE
                x[i * 2 + slot, place, 2 + int(vals[k])] += PULSE
            x[i * 2 + slot, 10, 4 + q] += PULSE
            y[i * 2 + slot] = vals[q]
    return x, y, pid


def reversed_write_order(seed, count):
    """Same facts, write slots swapped. The correct labels are unchanged."""
    rng = np.random.default_rng(seed)
    x, y, fk = query_set(seed, count)
    out = x.copy()
    out[:, [2, 6]] = out[:, [6, 2]]
    return out, y, fk


def shuffled_association(seed, count):
    """Assign values to the WRONG keys; the oracle label must follow the rewrite."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = np.zeros(count, np.int32)
    for i in range(count):
        q = i % 2
        v0 = int(rng.integers(2))
        vals = np.array([v0, 1 - v0], np.int32)
        swapped = vals[::-1]                     # deliberately wrong binding
        order = rng.permutation(2)
        for place, key in zip((2, 6), order):
            k = int(key)
            x[i, place, k] += PULSE
            x[i, place, 2 + int(swapped[k])] += PULSE
        x[i, 10, 4 + q] += PULSE
        y[i] = swapped[q]
    return x, y


def oracle_answer(x_row):
    """Deterministic oracle reading the INPUT spec, never a model."""
    vals = {}
    for place in (2, 6):
        keys = np.where(x_row[place, :2] > .5)[0]
        vs = np.where(x_row[place, 2:4] > .5)[0]
        if len(keys) != 1 or len(vs) != 1:
            raise ValueError("malformed episode")
        vals[int(keys[0])] = int(vs[0])
    qs = np.where(x_row[10, 4:6] > .5)[0]
    if len(qs) != 1:
        raise ValueError("malformed query")
    return vals[int(qs[0])]
