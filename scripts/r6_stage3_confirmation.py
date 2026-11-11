"""Round 6C Stage 3 Confirmation Benchmark on the 8 pre-registered confirmation seeds.

Seeds: 121011, 121017, 121023, 121029, 121031, 121037, 121041, 121043.
Opened ONLY after all 10 pre-registered gates (G1..G10) pass on the 3 development seeds.
Zero hyperparameter changes between dev and confirmation.
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

CONF_SEEDS = (121011, 121017, 121023, 121029, 121031, 121037, 121041, 121043)


def main() -> int:
    dev_file = ROOT / "artifacts/gemini-r6/dev-screen.json"
    if not dev_file.exists():
        raise RuntimeError("Cannot run confirmation before dev-screen.json exists.")
    dev_payload = json.loads(dev_file.read_text(encoding="utf-8"))
    if not dev_payload.get("gates", {}).get("all_passed", False):
        raise RuntimeError("Dev screen did not pass all 10 gates; confirmation run is blocked by protocol.")

    thread_env = R6.enforce_single_thread_env()
    proto = ROOT / "experiments/collab/scarx-neural-round1-20261111/gemini-r6/protocol-v1.md"
    theory = ROOT / "docs/research/collab/scarx-neural-round1-20261111/gemini-r6/theory.md"
    proto_sha = hashlib.sha256(proto.read_bytes()).hexdigest()
    theory_sha = hashlib.sha256(theory.read_bytes()).hexdigest()

    run_dir = new_run(
        ROOT,
        "r6c-confirmation-v1",
        {
            "protocol_sha256": proto_sha,
            "theory_sha256": theory_sha,
            "conf_seeds": list(CONF_SEEDS),
            "thread_env": thread_env,
        },
    )

    print("=" * 92)
    print("SCAR-X ROUND 6C — STAGE 3 CONFIRMATION BENCHMARK (8 SEEDS: 121011..121043)")
    print("=" * 92)

    seed_results = []
    with ResourceMonitor(interval=0.1) as mon:
        for seed in CONF_SEEDS:
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
        "stage": "stage3_confirmation",
        "protocol_sha256": proto_sha,
        "theory_sha256": theory_sha,
        "conf_seeds": list(CONF_SEEDS),
        "seed_results": seed_results,
        "summary": summary,
        "gates": gate_eval,
        "resources": res_report,
    }

    write_json(run_dir / "result.json", payload)
    out_file = ROOT / "artifacts/gemini-r6/confirmation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n--- STAGE 2 BASELINES (CLEAN, MEAN ACROSS 8 CONFIRMATION SEEDS) ---")
    for bk, bv in summary["baselines_clean"].items():
        print(
            f"  {bk:28s} direct={bv['direct']['mean']:.4f} (sd={bv['direct']['sd']:.4f})  "
            f"both_all={bv['both_all']['mean']:.4f}  both_A={bv['both_A']['mean']:.4f} (sd={bv['both_A']['sd']:.4f})  "
            f"both_B={bv['both_B']['mean']:.4f}  flip_A={bv['flip_rate_A']['mean']:.4f}"
        )

    print("\n--- SINGLE-FACTOR ABLATION LADDER (CLEAN, MEAN ACROSS 8 CONFIRMATION SEEDS) ---")
    for ak, av in summary["ablations_clean"].items():
        print(
            f"  {ak:32s} direct={av['direct']['mean']:.4f}  both_A={av['both_A']['mean']:.4f}  "
            f"both_B={av['both_B']['mean']:.4f}  wrong_fk_qB={av['wrong_first_key_qB_diff']['mean']:.4f}"
        )

    print("\n--- STAGE 3 ADVERSARIAL CONDITIONS (MEAN ACROSS 8 CONFIRMATION SEEDS) ---")
    print(
        f"  {'Condition':34s} {'R5 dir/bothA':>16s} {'CKCER-A dir/bothA':>20s} "
        f"{'CKCER-B dir/bothA':>20s} {'Δ_A bothA [95% CI]':>26s}"
    )
    for ck in summary["conditions"]["r5"]:
        r5_c = summary["conditions"]["r5"][ck]
        ca_c = summary["conditions"]["ckcer_mode_a"][ck]
        cb_c = summary["conditions"]["ckcer_mode_b"][ck]
        da_c = summary["conditions"]["paired_deltas_mode_a_vs_r5"][ck]["both_A"]
        print(
            f"  {ck:34s} {r5_c['direct']['mean']:.4f}/{r5_c['both_A']['mean']:.4f}   "
            f"{ca_c['direct']['mean']:.4f}/{ca_c['both_A']['mean']:.4f}       "
            f"{cb_c['direct']['mean']:.4f}/{cb_c['both_A']['mean']:.4f}   "
            f"{da_c['mean_delta']:+.4f} [{da_c['ci95_low']:+.4f}, {da_c['ci95_high']:+.4f}] ({da_c['wins']}/8)"
        )

    print("\n--- PROTECTED BENCHMARKS (MEAN ACROSS 8 CONFIRMATION SEEDS) ---")
    for pk in ("sim_xor", "xor_separated", "order_task"):
        r5_p = summary["protected"]["r5"][pk]
        ck_p = summary["protected"]["ckcer"][pk]
        dp = summary["protected"]["paired_deltas"][pk]
        print(
            f"  {pk:20s} R5={r5_p['mean']:.4f}  CKCER={ck_p['mean']:.4f}  "
            f"Δ={dp['mean_delta']:+.4f} [{dp['ci95_low']:+.4f}, {dp['ci95_high']:+.4f}]"
        )

    print("\n--- PRE-REGISTERED PROTOCOL GATES (G1..G10) ---")
    for gk, gv in gate_eval["gates"].items():
        print(f"  [{'PASS' if gv else 'FAIL'}] {gk}")
    print(f"\nCONFIRMATION VERDICT: {gate_eval['verdict']} | Peak Commit: {peak_bytes:,} B | Saved: {out_file}")
    return 0 if gate_eval["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
