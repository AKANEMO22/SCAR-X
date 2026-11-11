"""R7 round 43b — recompute the DECOLLE verdict from the run's own per-seed output.

The 43 run completed all 15 seeds and printed the aggregate table, then crashed in the
summary loop because `acc`/`probe` are per-task dicts and the paired-difference code
subtracted dicts. The artifact was never written, so the per-seed values below are
transcribed from the run's stdout (probe_mean per seed per arm), not re-simulated.

TRANSCRIBED probe_mean, in seed order (from the 43 run's printed lines):
  seed        frozen  scalar  perneuron  zero    shuffled
  620003..620157
The first five seeds scrolled off the captured tail; this script uses the TEN seeds that
were captured (620058..620157) and says so. n=10, paired.
"""
import json
from pathlib import Path

import numpy as np

SEEDS = [620058, 620069, 620080, 620091, 620102, 620113, 620124, 620135, 620146, 620157]
FROZEN = [0.882, 0.887, 0.903, 0.920, 0.898, 0.860, 0.895, 0.962, 0.913, 0.913]
SCALAR = [0.887, 0.887, 0.940, 0.945, 0.927, 0.895, 0.938, 0.965, 0.928, 0.938]
PERNEU = [0.953, 0.932, 0.908, 0.980, 0.947, 0.952, 0.930, 0.967, 0.925, 0.908]
ZERO = [0.882, 0.887, 0.903, 0.920, 0.898, 0.860, 0.895, 0.962, 0.913, 0.913]
SHUF = [0.870, 0.880, 0.912, 0.917, 0.892, 0.875, 0.857, 0.932, 0.958, 0.942]

ARMS = {"frozen": FROZEN, "scalar": SCALAR, "perneuron": PERNEU,
        "perneuron_zero": ZERO, "perneuron_shuffled": SHUF}


def ci(v, t=2.262):        # t(9) 95%
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - t * se), float(v.mean() + t * se)


def main():
    print("=" * 92)
    print("R7 ROUND 43b — DECOLLE verdict recomputed from the run's per-seed stdout")
    print("=" * 92)
    print(f"  n = {len(SEEDS)} seeds (the ten captured; five scrolled off the tail)")
    print("  paired, probe = ridge decodability of the frozen features (Q1)\n")

    print(f"  {'arm':20s} {'probe':>8} {'sd':>7}")
    for a, v in ARMS.items():
        v = np.array(v)
        print(f"  {a:20s} {v.mean():8.4f} {v.std(ddof=1):7.4f}")

    print("\n" + "=" * 92)
    print("PAIRED CONTROLS")
    print("=" * 92)
    rows = []
    for label, a, b in (
            ("perneuron - frozen      (any learning?)", "perneuron", "frozen"),
            ("perneuron - zero        (extra updates?)", "perneuron", "perneuron_zero"),
            ("perneuron - shuffled    (correct routing?)", "perneuron",
             "perneuron_shuffled"),
            ("perneuron - scalar      (per-neuron > scalar?)", "perneuron", "scalar"),
            ("scalar    - frozen      (production rule)", "scalar", "frozen")):
        d = np.array(ARMS[a]) - np.array(ARMS[b])
        m, lo, hi = ci(d)
        better = int((d > 0).sum())
        print(f"  {label:44s} {m*100:+6.2f}pp  CI [{lo*100:+6.2f}, {hi*100:+6.2f}]  "
              f"{better:2d}/10 better")
        rows.append({"label": label, "mean_pp": m * 100, "ci": [lo * 100, hi * 100],
                     "better": better})

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    d_pz = np.array(ARMS["perneuron"]) - np.array(ARMS["perneuron_zero"])
    d_ps = np.array(ARMS["perneuron"]) - np.array(ARMS["perneuron_shuffled"])
    d_pc = np.array(ARMS["perneuron"]) - np.array(ARMS["scalar"])
    m_pz, l_pz, _ = ci(d_pz)
    m_ps, l_ps, _ = ci(d_ps)
    m_pc, l_pc, h_pc = ci(d_pc)

    print(f"  registered: perneuron beats scalar by >= 2pp on the probe")
    print(f"  measured  : {m_pc*100:+.2f}pp  CI [{l_pc*100:+.2f}, {h_pc*100:+.2f}]")

    learned = m_pz > 0 and l_pz > 0
    specific = m_ps > 0 and l_ps > 0
    beats_scalar = m_pc > 0 and l_pc > 0

    print(f"\n  (a) does the hidden rule learn at all?      "
          f"perneuron - zero = {m_pz*100:+.2f}pp, CI lower {l_pz*100:+.2f}  "
          f"-> {'YES' if learned else 'NO'}")
    print(f"  (b) does the ROUTING matter?                "
          f"perneuron - shuffled = {m_ps*100:+.2f}pp, CI lower {l_ps*100:+.2f}  "
          f"-> {'YES' if specific else 'NO'}")
    print(f"  (c) does per-neuron beat the SCALAR?        "
          f"{m_pc*100:+.2f}pp, CI lower {l_pc*100:+.2f}  "
          f"-> {'YES' if beats_scalar else 'NO'}")

    if learned and specific and beats_scalar:
        print(f"\n  => CONFIRMED. Per-neuron error learns, the routing is specific, and it")
        print(f"     beats the production scalar.")
        verdict = "decolle_confirmed"
    elif learned and specific:
        print(f"\n  => PARTIAL, AND THE INTERESTING CASE. The hidden rule DOES learn")
        print(f"     ({m_pz*100:+.2f}pp over zero) and the ROUTING is specific")
        print(f"     ({m_ps*100:+.2f}pp over shuffled), but it does NOT beat the scalar")
        print(f"     ({m_pc*100:+.2f}pp). So hidden learning is possible -- contrary to what")
        print(f"     rounds 4/11 could establish -- but DECOLLE is not a better signal than")
        print(f"     what SCAR-X already has.")
        verdict = "decolle_learns_but_not_better"
    elif learned:
        print(f"\n  => The rule learns but the routing is NOT specific: permuting the")
        print(f"     per-neuron error costs {m_ps*100:+.2f}pp, so any non-zero factor works.")
        print(f"     That is generic update noise, not credit assignment.")
        verdict = "generic_noise"
    else:
        print(f"\n  => REJECTED. The hidden rule does not learn even with per-neuron error.")
        verdict = "decolle_rejected"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "decolle-transfer-recomputed.json").write_text(json.dumps({
        "source_run": "r7_r43_decolle.py (15 seeds ran; summary crashed on dict subtraction)",
        "note": "probe_mean per seed transcribed from the run's stdout; n=10 of 15",
        "seeds": SEEDS, "arms": {k: list(map(float, v)) for k, v in ARMS.items()},
        "paired": rows, "verdict": verdict,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: decolle-transfer-recomputed.json")


if __name__ == "__main__":
    main()
