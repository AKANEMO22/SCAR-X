"""Toy predictions for the unclipped delta, before any scored R5 screen."""
import numpy as np

from scarx.evaluation import r5b_probe as probe
from scarx.evaluation.r6_candidate import NoClip


class NoClip(probe.Head):
    def learn(self, f, y):
        p = self.probs(f)
        err = -p.copy()
        err[int(y)] += 1
        self.w += self.lr * err[:, None] * self.basis(f)[None, :]
        self.b += self.lr * err
        np.clip(self.w, -3, 3, out=self.w)
        np.clip(self.b, -3, 3, out=self.b)


def _fit(cls, rows, labels):
    head = cls(rows.shape[1], 5, lr=0.5, clip=0.025, zscore=False)
    for row in rows:
        head.observe(row)
    for row, label in zip(rows, labels):
        head.learn(row, int(label))
    return head


def test_unclipped_moves_farther_on_a_consistent_coordinate():
    rows = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (30, 1))
    labels = np.ones(30, np.int32)
    clipped = _fit(probe.Head, rows, labels)
    free = _fit(NoClip, rows, labels)
    assert abs(float(free.w[1, 0])) > abs(float(clipped.w[1, 0])) + 0.2


def test_unclipped_overshoots_a_huge_contradictory_coordinate():
    rows = np.tile(np.array([50.0, 0.0], np.float32), (10, 1))
    labels = np.array([0, 1] * 5, np.int32)
    clipped = _fit(probe.Head, rows, labels)
    free = _fit(NoClip, rows, labels)
    assert np.max(np.abs(free.w)) > np.max(np.abs(clipped.w)) + 1.0
