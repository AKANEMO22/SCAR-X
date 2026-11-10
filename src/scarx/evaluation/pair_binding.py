"""Rank-2 binding task: a valid replacement for the order-3 `strict_binding` gate.

Why this exists
---------------
`binding_audit21.strict_binding` computes `label = values[query] XOR query`.
The target is an order-3 interaction (key x value x query). An additive point
neuron followed by a linear readout is order-1 by construction, so that task is
unreachable at ANY learning-rule quality. Measured with fixed-feature ridge
probes on the raw tape: order-1 0.52, order-2 0.54, order-3 1.00, label-shuffled
per 0.45. The repository's isolated BPTT control also stalled at 63.54%.

`pair_binding` keeps the property that makes binding meaningful -- a bag of all
writes cannot answer it, because every key and every value appears exactly once
and only their PAIRING determines the label -- while staying order-2, so an
additive neuron plus linear readout can in principle express it.

Construction (12 ticks x 8 channels, 2 classes -- identical shapes to
strict_binding so the harness is unchanged):

  positions p1 in {1,2}, p2 in {4,5,6}
  at each position: key marker on channel `key`, value marker on channel 2+value
  key 0 and key 1 each appear exactly once; values 0 and 1 each appear once
  label = value[key=0] XOR value[key=1]

The label is a function of the pairing only. Swapping the values while keeping
the key positions produces the opposite label with an identical bag of writes.
"""
import numpy as np


def pair_binding(seed, count):
    """Order-2 binding task with NO bag shortcut.

    Design constraint (learned the hard way -- see report section on the
    discarded first version): the multiset of writes must be IDENTICAL for both
    labels, otherwise an order-1 bag readout reads the label directly and the
    task is not a binding test at all.

    Enforced by making the two values always differ: values = {0: v, 1: 1-v},
    so the bag always contains exactly one `0` marker and one `1` marker no
    matter the label. Only the PAIRING of key to value carries the label.

      label = value[key=0]

    A first version drew both values independently, which leaked: when the two
    values coincided the multiset collapsed to {0,0} or {1,1}, and a linear bag
    readout scored 0.70-0.77. That version is recorded as a design defect, not
    used here.
    """
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 2
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    for i in range(count):
        v0 = int(y[i])
        values = {0: v0, 1: 1 - v0}          # always differ -> bag is constant
        order = rng.permutation(2)            # which position holds which key
        positions = (int(rng.integers(1, 3)), int(rng.integers(4, 7)))
        for position, key in zip(positions, order):
            key = int(key)
            x[i, position, key] += 1.5
            x[i, position, 2 + values[key]] += 1.5
    order = rng.permutation(count)
    return x[order], y[order].astype(np.int32)


def order3_binding(seed, count):
    """The existing strict_binding generator, duplicated for probe comparison.

    Not used for training. Kept so the order analysis is reproducible from this
    module alone without importing the audit module.
    """
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 2
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    for i in range(count):
        query = int(rng.integers(2))
        a = int(y[i]) ^ query
        values = (a, 1 - a)
        order = rng.permutation(2)
        positions = (int(rng.integers(1, 3)), int(rng.integers(4, 7)))
        for position, key in zip(positions, order):
            key = int(key)
            x[i, position, key] += 1.5
            x[i, position, 2 + values[key]] += 1.5
        x[i, 10, 4 + query] += 1.5
        x[i, 11, 7] += 1
    order = rng.permutation(count)
    return x[order], y[order].astype(np.int32)


def order2_features(x):
    """Same-position key*value products: the order-2 basis a neuron could supply."""
    n = len(x)
    cols = [x.reshape(n, -1)]
    for p in (1, 2, 3, 4, 5, 6):
        for k in (0, 1):
            for v in (0, 1):
                cols.append((x[:, p, k] * x[:, p, 2 + v])[:, None])
    return np.concatenate(cols, axis=1)


def order3_features(x):
    """Adds the query interaction: only strict_binding should need this."""
    n = len(x)
    cols = [order2_features(x)]
    for p in (1, 2, 3, 4, 5, 6):
        for k in (0, 1):
            for v in (0, 1):
                for q in (0, 1):
                    cols.append((x[:, p, k] * x[:, p, 2 + v] * x[:, 10, 4 + q])[:, None])
    return np.concatenate(cols, axis=1)
