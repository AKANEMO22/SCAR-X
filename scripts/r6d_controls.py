"""Round 6D controls — fills the gates that R6A hard-coded.

G4 random labels, G5 shortcut baselines, computed on the SAME cells and seeds as the
matrix so the numbers are comparable. Nothing here is a literal True.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r6d_audit as A  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402


def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r6d/protocol-v1.md"
    psha = hashlib.sha256(proto.read_bytes()).hexdigest()
    phase = sys.argv[1] if len(sys.argv) > 1 else "replay"
    seeds = A.DEV_SEEDS if phase == "replay" else A.FRESH_SEEDS
    run = new_run(root, f"r6d-controls-{phase}-v1", {
        "protocol_sha256": psha, "seeds": list(seeds), "phase": phase})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in seeds:
            sc = A.fixed_calibration(seed, "r5b")
            d = A.build_cell(seed, 16384, sc, 1024, 256)
            rng = np.random.default_rng(seed + 404)

            # reference
            h = A.train_head(d["f_tr"], d["y_tr"], seed, rng.permutation(16384), 1)
            ref = A.evaluate(h, d)

            # G4: random training labels
            y_rand = rng.permutation(d["y_tr"])
            hr = A.train_head(d["f_tr"], y_rand, seed, rng.permutation(16384), 1)
            rand_direct = float(np.mean(hr.predict(d["f_te"]) == d["y_te"]))

            # G5: shortcut baselines on held-out
            fk = float(np.mean(G.baseline_first_key(d["tex"]) == d["y_te"]))
            qo = float(np.mean(G.baseline_query_only(d["tex"]) == d["y_te"]))

            rows.append({"seed": seed, "ref_both_A": ref["both_A"],
                         "ref_direct": ref["direct"],
                         "random_labels_direct": rand_direct,
                         "first_key": fk, "query_only": qo})
            mon.check()
            print(f"  seed{seed} ref both_A {ref['both_A']:.4f} rand {rand_direct:.4f} "
                  f"fk {fk:.4f} qo {qo:.4f}", flush=True)

    m = lambda k: float(np.mean([r[k] for r in rows]))       # noqa: E731
    summary = {
        "reference_both_A": m("ref_both_A"), "reference_direct": m("ref_direct"),
        "random_labels_direct": m("random_labels_direct"),
        "first_key": m("first_key"), "query_only": m("query_only"),
        # gates computed from the measurement, never hard-coded
        "G4_random_labels_chance": bool(m("random_labels_direct") <= .60),
        "G5_shortcuts_ok": bool(abs(m("first_key") - .75) <= .05
                                and abs(m("query_only") - .50) <= .05),
    }
    write_json(run / "result.json",
               {"protocol_sha256": psha, "trials": rows, "summary": summary,
                "resources": mon.report(), "wall_seconds": time.time() - t0})

    print("\n" + "=" * 70)
    print("CONTROLS")
    print("=" * 70)
    print(f"  reference both_A     {summary['reference_both_A']:.4f}")
    print(f"  random labels        {summary['random_labels_direct']:.4f} "
          f"(gate <=0.60: {summary['G4_random_labels_chance']})")
    print(f"  first-key shortcut   {summary['first_key']:.4f} (expect ~0.75)")
    print(f"  query-only           {summary['query_only']:.4f} (expect ~0.50)")
    print(f"  G5: {summary['G5_shortcuts_ok']}")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
