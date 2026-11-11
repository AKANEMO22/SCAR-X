"""Paired statistics for the Round 2A confirmation run."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scarx.evaluation.phase21 import interval  # noqa: E402


def main():
    root = Path(__file__).resolve().parents[1]
    # match on THIS round's protocol prefix only; older runs share the word
    # "confirmation" and must not be picked up
    conf = [d for d in (root / "runs").iterdir()
            if d.is_dir() and "r2-substrate-confirmation" in d.name][0]
    rows = json.loads((conf / "result.json").read_text(encoding="utf-8"))["trials"]
    print(f"run: {conf.name}")

    by = {(r["seed"], r["substrate"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})

    print("=" * 76)
    print(f"PAIRED CONTRASTS over {len(seeds)} confirmation seeds")
    print("=" * 76)

    out = {"n_seeds": len(seeds), "seeds": seeds, "contrasts": {}}

    pairs = [
        ("head: spikesum - decay025", "direct", "spikesum", "decay025"),
        ("head: bank5 - decay025", "direct", "bank5", "decay025"),
        ("oracle: spikesum - decay025", "oracle", "spikesum", "decay025"),
        ("oracle: bank5_proj48 - decay025", "oracle", "bank5_proj48", "decay025"),
    ]
    for name, key, a, b in pairs:
        d = [by[(s, a)][key] - by[(s, b)][key] for s in seeds]
        iv = interval(d)
        out["contrasts"][name] = iv
        print(f"  {name:34s} {iv['mean']*100:+7.2f}pp "
              f"[{iv['lower95']*100:+7.2f}, {iv['upper95']*100:+7.2f}]")

    print()
    print("  gap to oracle (head - oracle), per substrate")
    for sub in ("decay025", "spikesum", "bank5", "bank5_proj48"):
        d = [by[(s, sub)]["direct"] - by[(s, sub)]["oracle"] for s in seeds]
        iv = interval(d)
        out["contrasts"][f"gap:{sub}"] = iv
        verdict = "MATCHES ORACLE" if iv["lower95"] > -.05 else "lags"
        print(f"    {sub:14s} {iv['mean']*100:+7.2f}pp "
              f"[{iv['lower95']*100:+7.2f}, {iv['upper95']*100:+7.2f}]  {verdict}")

    print()
    print("  head accuracy range on spikesum (every seed must clear 0.65)")
    v = [by[(s, "spikesum")]["direct"] for s in seeds]
    print(f"    min {min(v):.4f}  max {max(v):.4f}  mean {np.mean(v):.4f}")
    out["spikesum_head_min"] = min(v)
    out["spikesum_head_max"] = max(v)

    print()
    print("  decay025 head never exceeds 0.65?")
    v2 = [by[(s, "decay025")]["direct"] for s in seeds]
    print(f"    min {min(v2):.4f}  max {max(v2):.4f}  (gate 0.65)")
    out["decay025_head_max"] = max(v2)

    p = root / "artifacts/deepseek-r2/confirmation-stats.json"
    p.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nwritten: {p}")


if __name__ == "__main__":
    main()
