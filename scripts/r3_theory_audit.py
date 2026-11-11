"""Pre-protocol theoretical audit of candidate D and E.

The brief allows changing the D/E choice based on a theory audit BEFORE the
protocol is locked. This measures whether the two candidates can in principle
separate the five tasks, using the offline ridge only as a linear diagnostic.

Nothing here is a capability claim. It decides which two candidates to register.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation.r3_substrate import (  # noqa: E402
    SUBSTRATES, TASKS, dims, substrate,
)
from scarx.evaluation.tasks21 import ridge_probe  # noqa: E402

SEEDS = (91001, 91007)
N_TRAIN, N_TEST = 400, 300


def main():
    print("=" * 84)
    print("LINEAR DIAGNOSTIC: can each substrate carry each task?  (ridge only)")
    print("This is NOT an information ceiling and NOT a capability result.")
    print("=" * 84)
    print(f"  {'substrate':18s} {'dims':>5} " +
          " ".join(f"{t:>14s}" for t in TASKS))
    table = {}
    for name in SUBSTRATES:
        cells = []
        for task in TASKS:
            accs = []
            for seed in SEEDS:
                trx, try_ = TASKS[task](seed + 100_000, N_TRAIN)
                tex, tey = TASKS[task](seed + 900_000, N_TEST)
                a = substrate(seed, trx, name)
                b = substrate(seed, tex, name)
                accs.append(ridge_probe(a, try_, b, tey))
            cells.append(float(np.mean(accs)))
        table[name] = dict(zip(TASKS, cells))
        print(f"  {name:18s} {dims(name):>5} " +
              " ".join(f"{c:14.4f}" for c in cells))

    print("\n" + "=" * 84)
    print("VERDICT INPUTS for the D/E choice")
    print("=" * 84)
    print("  best task-carrying substrate per task:")
    for task in TASKS:
        best = max(table, key=lambda s: table[s][task])
        print(f"    {task:16s} {best:18s} {table[best][task]:.4f}")

    # does any candidate beat both baselines on any task?
    print("\n  candidate vs the better of decay025 / spikesum (ridge):")
    for cand in ("count_plus_fast", "temporal_bins_3"):
        wins = []
        for task in TASKS:
            base = max(table["decay025"][task], table["spikesum"][task])
            delta = table[cand][task] - base
            if delta > .02:
                wins.append((task, delta))
        print(f"    {cand:18s} beats baseline on {len(wins)} task(s): "
              + (", ".join(f"{t}{d:+.3f}" for t, d in wins) if wins else "none"))

    print("\n  bank5 (the 5-timescale reference):")
    for task in TASKS:
        base = max(table["decay025"][task], table["spikesum"][task])
        print(f"    {task:16s} bank5-base {table['bank5'][task]-base:+.4f}")


if __name__ == "__main__":
    main()
