from collections import defaultdict
import os
from pathlib import Path
import tempfile
from itertools import islice
import numpy as np
from ..neural.core import SpikingCore


class LearningController:
    def __init__(self, model, checkpoint_dir):
        self.model = model
        self.directory = Path(checkpoint_dir)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.last_checkpoint = None

    def _score(self, guard):
        groups = defaultdict(list)
        for sequence, target, group in guard:
            groups[group].append(int(np.argmax(self.model.forward(sequence))) == target)
        return {k:float(np.mean(v)) for k,v in groups.items()}

    def learn(self, examples, *, guard, mode="full", replay=(), consolidate=True):
        """Guard data are development sentinels, never final test data.

        examples/replay are bounded materialized sequences. Any failure restores
        all weights, state, traces, reward baseline and update counter.
        """
        examples = list(islice(iter(examples),513))
        replay = list(islice(iter(replay),129))
        guard = list(islice(iter(guard),513))
        if len(examples)>512 or len(replay)>128 or len(guard)>512:
            raise ValueError("Batch exceeds bounded learning/replay/guard capacity")
        if not guard:
            raise ValueError("A development guard set is required for transactional learning")
        total = 0
        for record in examples+replay+guard:
            x,y = record[:2]
            if not isinstance(x,np.ndarray) or x.ndim!=2 or x.shape[1]!=self.model.config.inputs or not 1<=len(x)<=4096 or not np.isfinite(x).all():
                raise ValueError("Invalid learning episode")
            if not isinstance(y,(int,np.integer)) or not 0<=y<self.model.config.outputs:
                raise ValueError("Invalid learning target")
            total += x.nbytes
        if total > 64_000_000:
            raise MemoryError("Episode batch exceeds 64 MB working buffer")
        fd, name = tempfile.mkstemp(prefix="before-",suffix=".npz",dir=self.directory)
        os.close(fd)
        self.model.save(name)
        self.last_checkpoint = name
        try:
            before = self._score(guard)
            losses = []
            for sequence,target in list(examples)+list(replay):
                self.model.forward(sequence,track=True)
                losses.append(self.model.learn(target,mode=mode))
            if consolidate and mode != "frozen":
                self.model.consolidate()
            after = self._score(guard)
            drops = [before[k]-after[k] for k in before]
            rejected = bool(max(drops) > .05+1e-9 or np.mean(drops) > .03+1e-9)
            if rejected:
                self.rollback(name)
            return {"accepted":not rejected,"before":before,"after_candidate":after,
                    "mean_loss":float(np.mean(losses)) if losses else None,"rollback":name if rejected else None}
        except BaseException:
            self.rollback(name)
            raise

    def rollback(self, checkpoint=None):
        target = checkpoint or self.last_checkpoint
        if target is None:
            raise ValueError("No rollback checkpoint")
        self.model = type(self.model).load(target)
        return self.model

