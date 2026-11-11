"""Unclipped R5 head. One change: the per-step clip is absent."""
import numpy as np

from . import r5b_probe as probe


class NoClip(probe.Head):
    def learn(self, f, y):
        probabilities = self.probs(f)
        error = -probabilities.copy()
        error[int(y)] += 1
        self.w += self.lr * error[:, None] * self.basis(f)[None, :]
        self.b += self.lr * error
        np.clip(self.w, -3, 3, out=self.w)
        np.clip(self.b, -3, 3, out=self.b)
