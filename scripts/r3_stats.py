"""Round 3A paired statistics on the confirmation run."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scarx.evaluation.phase21 import interval  # noqa: E402


def main():
    root = Path(__file__).resolve().parents[1]
    conf = [d for d in (root / "runs").iterdir()
            if d.is_dir() and "r3-temporal-confirmation" in d.name][0]
    rows = json.loads((conf / "result.json").read_text(encoding="utf-8"))["trials"]
    by = {(r["seed"], r["task"], r["substrate"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    tasks = ("sim_xor", "xor", "order", "pair_binding", "strict_binding")
    print(f"run: {conf.name}\nseeds: {len(seeds)}")

    print("\n" + "=" * 78)
    print("PRIMARY CONTRASTS vs the better baseline, paired by seed (pp)")
    print("=" * 78)
    out = {"run": conf.name, "n_seeds": len(seeds), "contrasts": {}}
    for cand in ("count_plus_fast", "temporal_bins_3"):
        print(f"\n  {cand}")
        for task in tasks:
            d = []
            for s in seeds:
                base = max(by[(s, task, "decay025")]["head"],
                           by[(s, task, "spikesum")]["head"])
                d.append(by[(s, task, cand)]["head"] - base)
            iv = interval(d)
            key = f"{cand}|{task}"
            out["contrasts"][key] = iv
            mark = "WIN" if iv["lower95"] > 0 else ("loss" if iv["upper95"] < 0 else "n.s.")
            print(f"    {task:16s} {iv['mean']*100:+7.2f} "
                  f"[{iv['lower95']*100:+7.2f}, {iv['upper95']*100:+7.2f}]  {mark}")

    print("\n" + "=" * 78)
    print("SPIKE-SUM LOSES ORDER INFORMATION?  (vs temporal substrates)")
    print("=" * 78)
    for task in ("xor", "order"):
        d = [by[(s, task, "temporal_bins_3")]["head"]
             - by[(s, task, "spikesum")]["head"] for s in seeds]
        iv = interval(d)
        out["contrasts"][f"bins3_minus_spikesum|{task}"] = iv
        print(f"  bins3 - spikesum on {task:6s}: {iv['mean']*100:+7.2f}pp "
              f"[{iv['lower95']*100:+7.2f}, {iv['upper95']*100:+7.2f}]")

    print("\n  head on spikesum per task, mean over 8 seeds:")
    for task in tasks:
        v = [by[(s, task, "spikesum")]["head"] for s in seeds]
        print(f"    {task:16s} {np.mean(v):.4f}")

    print("\n" + "=" * 78)
    print("ORACLE vs HEAD: does the online head exploit what ridge can read?")
    print("=" * 78)
    print(f"  {'substrate':18s} " + " ".join(f"{t:>15s}" for t in tasks))
    for sub in ("decay025", "spikesum", "bank5", "count_plus_fast",
                "temporal_bins_3"):
        g = []
        for task in tasks:
            d = [by[(s, task, sub)]["head"] - by[(s, task, sub)]["oracle"]
                 for s in seeds]
            g.append(np.mean(d) * 100)
        print(f"  {sub:18s} " + " ".join(f"{x:+15.1f}" for x in g))

    p = root / "artifacts/deepseek-r3/confirmation-stats.json"
    p.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nwritten: {p}")


if __name__ == "__main__":
    main()
