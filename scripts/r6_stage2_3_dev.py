"""Round 6C Stage 2 & Stage 3 Development Screen on the 3 pre-registered dev seeds (120011, 120017, 120023).

Evaluates:
- All 10 Stage 2 mandatory baselines & controls
- Single-factor ablation ladder from R5 to CKCER-ModeA
- Stage 3 adversarial benchmark suite for R5, CKCER-ModeA, and CKCER-ModeB
- Protected tasks (sim_xor, xor_separated, order_task)
- Auxiliary generalization tasks (structured 3rd write with variable windows & 3-write key overwrite)
- All 10 pre-registered GO/STOP protocol gates (G1..G10)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scarx.evaluation import r6_associative as R6  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

DEV_SEEDS = (120011, 120017, 120023)


def main() -> int:
    thread_env = R6.enforce_single_thread_env()
    proto = ROOT / "experiments/collab/scarx-neural-round1-20261111/gemini-r6/protocol-v1.md"
    theory = ROOT / "docs/research/collab/scarx-neural-round1-20261111/gemini-r6/theory.md"
    proto_sha = hashlib.sha256(proto.read_bytes()).hexdigest()
    theory_sha = hashlib.sha256(theory.read_bytes()).hexdigest()

    run_dir = new_run(
        ROOT,
        "r6c-dev-screen-v1",
        {
            "protocol_sha256": proto_sha,
            "theory_sha256": theory_sha,
            "dev_seeds": list(DEV_SEEDS),
            "thread_env": thread_env,
        },
    )

    print("=" * 92)
    print("SCAR-X ROUND 6C — STAGE 2 & STAGE 3 DEVELOPMENT SCREEN (SEEDS: 120011, 120017, 120023)")
    print("=" * 92)

    seed_results = []
    with ResourceMonitor(interval=0.1) as mon:
        for seed in DEV_SEEDS:
            sr = R6.evaluate_seed_full(seed)
            seed_results.append(sr)
            mon.check()
            c_r5 = sr["conditions"]["r5"]["clean"]
            c_a = sr["conditions"]["ckcer_mode_a"]["clean"]
            c_b = sr["conditions"]["ckcer_mode_b"]["clean"]
            print(
                f"  [seed {seed}] R5 clean: dir={c_r5['direct']:.4f} bothA={c_r5['both_A']:.4f} | "
                f"CKCER-A: dir={c_a['direct']:.4f} bothA={c_a['both_A']:.4f} | "
                f"CKCER-B: dir={c_b['direct']:.4f} bothA={c_b['both_A']:.4f} (gate={c_b['oja_gate_purity']:.4f})",
                flush=True,
            )

    res_report = mon.report()
    peak_bytes = int(res_report.get("peak_commit_or_vms_bytes", 0))
    summary = R6.summarize_across_seeds(seed_results, bootstrap_seed=2101)
    gate_eval = R6.evaluate_protocol_gates(summary, peak_commit_bytes=peak_bytes)

    payload = {
        "stage": "stage2_3_dev_screen",
        "protocol_sha256": proto_sha,
        "theory_sha256": theory_sha,
        "dev_seeds": list(DEV_SEEDS),
        "seed_results": seed_results,
        "summary": summary,
        "gates": gate_eval,
        "resources": res_report,
    }

    write_json(run_dir / "result.json", payload)
    out_file = ROOT / "artifacts/gemini-r6/dev-screen.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n--- STAGE 2 BASELINES (CLEAN, MEAN ACROSS 3 DEV SEEDS) ---")
    for bk, bv in summary["baselines_clean"].items():
        print(
            f"  {bk:28s} direct={bv['direct']['mean']:.4f}  both_all={bv['both_all']['mean']:.4f}  "
            f"both_A={bv['both_A']['mean']:.4f}  both_B={bv['both_B']['mean']:.4f}  "
            f"flip_A={bv['flip_rate_A']['mean']:.4f}"
        )

    print("\n--- SINGLE-FACTOR ABLATION LADDER (CLEAN, MEAN ACROSS 3 DEV SEEDS) ---")
    for ak, av in summary["ablations_clean"].items():
        print(
            f"  {ak:32s} direct={av['direct']['mean']:.4f}  both_A={av['both_A']['mean']:.4f}  "
            f"both_B={av['both_B']['mean']:.4f}  wrong_fk_qB={av['wrong_first_key_qB_diff']['mean']:.4f}"
        )

    print("\n--- STAGE 3 ADVERSARIAL CONDITIONS (MEAN ACROSS 3 DEV SEEDS) ---")
    print(f"  {'Condition':34s} {'R5 dir/bothA':>16s} {'CKCER-A dir/bothA':>20s} {'CKCER-B dir/bothA':>20s}")
    for ck in summary["conditions"]["r5"]:
        r5_c = summary["conditions"]["r5"][ck]
        ca_c = summary["conditions"]["ckcer_mode_a"][ck]
        cb_c = summary["conditions"]["ckcer_mode_b"][ck]
        print(
            f"  {ck:34s} {r5_c['direct']['mean']:.4f}/{r5_c['both_A']['mean']:.4f}   "
            f"{ca_c['direct']['mean']:.4f}/{ca_c['both_A']['mean']:.4f}       "
            f"{cb_c['direct']['mean']:.4f}/{cb_c['both_A']['mean']:.4f}"
        )

    print("\n--- PROTECTED BENCHMARKS (MEAN ACROSS 3 DEV SEEDS) ---")
    for pk in ("sim_xor", "xor_separated", "order_task"):
        print(
            f"  {pk:20s} R5={summary['protected']['r5'][pk]['mean']:.4f}  "
            f"CKCER={summary['protected']['ckcer'][pk]['mean']:.4f}"
        )

    print("\n--- PRE-REGISTERED PROTOCOL GATES (G1..G10) ---")
    for gk, gv in gate_eval["gates"].items():
        print(f"  [{'PASS' if gv else 'FAIL'}] {gk}")
    print(f"\nDEV VERDICT: {gate_eval['verdict']} | Peak Commit: {peak_bytes:,} B | Saved: {out_file}")
    return 0 if gate_eval["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
