"""Deterministic synthetic task generator v1, created for this repository.

Test seeds are disjoint from training. Generalization is to new noise/delays,
not to novel semantic concepts or new language instructions.
"""
import numpy as np


def episodes(seed, count, *, classes=4, contexts=1, context=0, delay=4, reverse=False):
    rng = np.random.default_rng(seed)
    permutation = np.random.default_rng(700+context).permutation(classes)
    for i in range(count):
        cue = i % classes
        x = rng.normal(0,.025,(delay+3,classes+contexts)).astype(np.float32)
        x[:2,cue] += 1.
        x[:,classes+context] += .65
        target = int(permutation[cue]) if contexts > 1 else cue
        if reverse:
            target = (target+1) % classes
        yield x,target


def accuracy(model, samples):
    return np.array([int(np.argmax(model.forward(x))) == y for x,y in samples],dtype=np.float32)

