"""Quick smoke: does head conditioning close the ridge gap? (Exploratory, not the run.)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation.conditioned_head import ConditionedCore, sim_xor
from scarx.evaluation.tasks21 import ridge_probe


def main():
    print("=" * 74)
    print("SMOKE: can a delta-rule head reach the offline ridge on a FROZEN hidden?")
    print("=" * 74)
    seed = 71001
    fx, fy = sim_xor(seed + 100_000, 128)
    trx, try_ = sim_xor(seed + 200_000, 400)
    tx, ty = sim_xor(seed + 300_000, 300)

    for head in ("raw", "zscore", "rms"):
        m = ConditionedCore(seed, head=head, signal="frozen")
        # baseline: ridge on the same frozen features
        rf = m.frozen_features(fx); rt = m.frozen_features(tx)
        ridge_acc = ridge_probe(rf, fy, rt, ty)
        # train the head only (hidden frozen), then measure
        for seq, lab in zip(trx, try_):
            m.run(seq, track=False)
            m.learn(int(lab))
        acc = m.head_accuracy(tx, ty)
        print(f"  head={head:8s}  ridge={ridge_acc:.4f}  head_after_training={acc:.4f}  "
              f"gap={acc-ridge_acc:+.4f}")


if __name__ == "__main__":
    main()
