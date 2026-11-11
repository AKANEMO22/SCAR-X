"""Post-hoc EXPLORATORY out-of-design check (Amendment 1, not a protocol-v1 gate).

Does DC-ALIF's strict_binding result survive on a different binding task that the
mechanism was not designed against?  Uses deepseek's `pair_binding` generator
(order-2 pairing, constant bag) loaded read-only from its worktree by file path.
Hyperparameters are the locked protocol-v1 values; nothing is tuned on this task.
Fresh seeds 36011.. (not used by any registered run).  Writes one JSON under
artifacts/collab/<campaign>/antigravity/.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

import numpy as np

from scarx.evaluation.dc_alif_eligibility import DCALIFEligibility
from scarx.evaluation.phase21 import interval
from scarx.neural.core import Config

GEN = Path(r"C:\Users\hachimi\Downloads\model agent\scar-x-deepseek-round1\src\scarx\evaluation\pair_binding.py")
spec = importlib.util.spec_from_file_location("deepseek_pair_binding", GEN)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
pair_binding = mod.pair_binding

SEEDS = (36011, 36017, 36029, 36037, 36041, 36053, 36059, 36061)
METHODS = ("correct", "no_gate", "frozen", "shuffled")


def trial(seed, method):
    c = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0)
    m = DCALIFEligibility(c, method)
    cal, _ = pair_binding(seed + 1360000, 128)
    m.calibrate(cal)
    x, y = pair_binding(seed + 1370000, 1024)
    for s, t in zip(x, y):
        m.learn(s, int(t))
    tx, ty = pair_binding(seed + 1390000, 512)
    return float(np.mean([m.predict(s) == t for s, t in zip(tx, ty)]))


def main():
    began = time.perf_counter()
    rows = [{"seed": s, **{m: trial(s, m) for m in METHODS}} for s in SEEDS]
    out = {
        "scope": "post-hoc exploratory; pair_binding generator from research/deepseek (read-only); locked hyperparameters; NOT a protocol-v1 gate",
        "generator_file_sha256": __import__("hashlib").sha256(GEN.read_bytes()).hexdigest(),
        "seeds": list(SEEDS),
        "rows": rows,
        "mean": {m: float(np.mean([r[m] for r in rows])) for m in METHODS},
        "correct_minus_frozen": interval([r["correct"] - r["frozen"] for r in rows]),
        "correct_minus_no_gate": interval([r["correct"] - r["no_gate"] for r in rows]),
        "correct_minus_shuffled": interval([r["correct"] - r["shuffled"] for r in rows]),
        "seconds": time.perf_counter() - began,
    }
    dest = ROOT / "artifacts" / "collab" / "scarx-neural-round1-20261111" / "antigravity" / "explore-pair-binding.json"
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("mean", "correct_minus_frozen", "correct_minus_no_gate", "correct_minus_shuffled", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
