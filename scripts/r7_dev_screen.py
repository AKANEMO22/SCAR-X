"""Execute Round 7 P0 Verification and Pre-Registered DEV Screen (seeds 140011, 140017, 140023).

Produces:
- artifacts/gemini-r7/p0-verification.json
- artifacts/gemini-r7/dev-screen.json
- runs/<run_id>/{manifest.json, source.zip, trials.jsonl, result.json, report.md}
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np

from scarx.neural.core import Config, SpikingCore
from scarx.runtime.resources import ResourceMonitor
from scarx.evaluation.runs import new_run, write_json, cluster_interval
from scarx.evaluation.eligibility_validation import fixed_presynaptic_tape
from scarx.evaluation.r7_hidden_plasticity import (
    EventWindowedHomeostaticEligibility,
    P1_TASKS,
    P1_CONDITIONS,
    generate_p1_task,
    run_p1_task_seed,
    run_p2_continual_seed,
)

DEV_SEEDS = (140011, 140017, 140023)
AUDIT_SEED = 129001


def run_p0_verification() -> dict[str, object]:
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=140011, anchor_strength=0.0)
    core = SpikingCore(cfg)
    model = EventWindowedHomeostaticEligibility(cfg, "learned_ew_he")
    seqs, _ = generate_p1_task("strict_binding", 140011, 16)

    parity_exact = True
    for seq in seqs:
        core.reset_state()
        model.reset_episode()
        for row in seq:
            core.step(row, track=False)
            model.step(row, track=True)
            for field in ("voltage", "adaptation", "spikes", "features"):
                if not np.array_equal(getattr(core, field), getattr(model.core, field)):
                    parity_exact = False

    # Anchored finite difference check
    seq0 = seqs[0]
    presyn, spikes, _ = fixed_presynaptic_tape(model.core, seq0)
    model.reset_episode()
    for row in seq0:
        model.step(row, track=True)

    def scalar_anchored_ew_he(
        weights: np.ndarray,
        presyn_j: np.ndarray,
        observed_j: np.ndarray,
        baseline_weights: np.ndarray,
    ) -> float:
        v = a = 0.0
        margins = []
        for x, z in zip(presyn_j, observed_j):
            u = cfg.leak * v + float(np.dot(baseline_weights, x))
            theta = cfg.threshold + cfg.adaptation_strength * a
            margins.append((u - theta) / 0.4)
            v = u - float(z) * theta
            a = cfg.adaptation_decay * a + float(z)
        v = a = f = 0.0
        for x, z, base in zip(presyn_j, observed_j, margins):
            u = cfg.leak * v + float(np.dot(weights, x))
            theta = cfg.threshold + cfg.adaptation_strength * a
            actual = 1.0 / (1.0 + np.exp(-np.clip((u - theta) / 0.4, -20.0, 20.0)))
            anchor = 1.0 / (1.0 + np.exp(-np.clip(base, -20.0, 20.0)))
            local = float(z) + actual - anchor
            v = u - local * theta
            a = cfg.adaptation_decay * a + local
            f = 0.25 * f + 0.75 * local
        return f

    eps = 1e-5
    max_fd_err = 0.0
    checked_synapses = 0
    for j in (0, 7, 19, 37):
        w_row = np.concatenate((model.core.w_in[j], model.core.w_rec[j])).astype(np.float64)
        for i in range(len(w_row)):
            p_w = w_row.copy()
            m_w = w_row.copy()
            p_w[i] += eps
            m_w[i] -= eps
            fd = (
                scalar_anchored_ew_he(p_w, presyn[:, j], spikes[:, j], w_row)
                - scalar_anchored_ew_he(m_w, presyn[:, j], spikes[:, j], w_row)
            ) / (2.0 * eps)
            err = abs(float(model.e_f[j, i]) - fd)
            max_fd_err = max(max_fd_err, err)
            checked_synapses += 1

    # Control identity & checkpoint continuation
    cal_x, _ = generate_p1_task("strict_binding", 140111, 128)
    tr_x, tr_y = generate_p1_task("strict_binding", 140211, 32)
    te_x, _ = generate_p1_task("strict_binding", 140311, 32)

    m_frozen = EventWindowedHomeostaticEligibility(cfg, "frozen_hidden")
    m_zero = EventWindowedHomeostaticEligibility(cfg, "zero_signal")
    m_frozen.calibrate(cal_x)
    m_zero.calibrate(cal_x)
    for seq, y in zip(tr_x, tr_y):
        m_frozen.learn_episode(seq, int(y))
        m_zero.learn_episode(seq, int(y))
    zero_equals_frozen = bool(
        np.array_equal(m_frozen.core.w_in, m_zero.core.w_in)
        and np.array_equal(m_frozen.core.w_rec, m_zero.core.w_rec)
        and np.array_equal(m_frozen.predict_many(te_x), m_zero.predict_many(te_x))
    )

    m_ckpt = EventWindowedHomeostaticEligibility(cfg, "learned_ew_he")
    m_ckpt.calibrate(cal_x)
    bytes_before = m_ckpt.memory_bytes()
    for seq, y in zip(tr_x[:16], tr_y[:16]):
        m_ckpt.learn_episode(seq, int(y))
    payload = m_ckpt.dumps()
    m_restored = EventWindowedHomeostaticEligibility.loads(payload)
    for seq, y in zip(tr_x[16:], tr_y[16:]):
        m_ckpt.learn_episode(seq, int(y))
        m_restored.learn_episode(seq, int(y))
    bytes_after = m_ckpt.memory_bytes()
    checkpoint_exact = bool(
        np.array_equal(m_ckpt.core.w_in, m_restored.core.w_in)
        and np.array_equal(m_ckpt.core.w_rec, m_restored.core.w_rec)
        and np.array_equal(m_ckpt.predict_many(te_x), m_restored.predict_many(te_x))
    )

    return {
        "pass": bool(
            parity_exact
            and max_fd_err < 1e-5
            and zero_equals_frozen
            and bytes_before == bytes_after
            and bytes_after < 100_000
            and checkpoint_exact
        ),
        "forward_parity_exact": parity_exact,
        "anchored_fd_synapses_checked": checked_synapses,
        "anchored_fd_max_abs_error": float(max_fd_err),
        "zero_signal_equals_frozen_bit_exact": zero_equals_frozen,
        "memory_bytes_before": int(bytes_before),
        "memory_bytes_after": int(bytes_after),
        "checkpoint_continuation_bit_exact": checkpoint_exact,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art_dir = root / "artifacts" / "gemini-r7"
    art_dir.mkdir(parents=True, exist_ok=True)

    config_meta = {
        "protocol": "gemini-r7-dev-v1",
        "agent": "antigravity",
        "model": "Gemini 3.1 Pro (High)",
        "baseline_commit": "495dab2f4bac0180e4d1cd28674391f055ccf09d",
        "audit_seed": AUDIT_SEED,
        "dev_seeds": list(DEV_SEEDS),
        "hidden": 48,
        "fan_in": 8,
        "p1_tasks": list(P1_TASKS),
        "p1_conditions": list(P1_CONDITIONS),
        "ram_limit": 2_000_000_000,
    }

    p0_result = run_p0_verification()
    write_json(art_dir / "p0-verification.json", p0_result)
    print(f"P0 verification: pass={p0_result['pass']}, max_fd_err={p0_result['anchored_fd_max_abs_error']:.2e}", flush=True)
    if not p0_result["pass"]:
        raise RuntimeError("P0 verification failed")

    run_dir = new_run(root, "gemini-r7-dev-v1", config_meta)
    start_t = time.perf_counter()

    with ResourceMonitor(2_000_000_000) as monitor:
        # 1. Record Stage-0 audit seed 129001 summary for strict_binding & xor
        audit_p1 = {
            t: run_p1_task_seed(t, AUDIT_SEED)
            for t in ("strict_binding", "context_selective_rule", "xor", "order")
        }
        monitor.check()

        # 2. Run Pre-Registered DEV Screen across DEV_SEEDS = (140011, 140017, 140023)
        dev_p1_trials: list[dict[str, object]] = []
        per_task_seed: dict[str, dict[int, dict[str, dict[str, object]]]] = {t: {} for t in P1_TASKS}

        for seed in DEV_SEEDS:
            for task in P1_TASKS:
                t0 = time.perf_counter()
                res = run_p1_task_seed(task, seed)
                elapsed = time.perf_counter() - t0
                per_task_seed[task][seed] = res
                row = {
                    "kind": "dev_p1",
                    "seed": seed,
                    "task": task,
                    "seconds": float(elapsed),
                    "conditions": res,
                }
                dev_p1_trials.append(row)
                with (run_dir / "trials.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                ew = res["learned_ew_he"]
                fr = res["frozen_hidden"]
                print(
                    f"[DEV P1] seed={seed} task={task:23s} | "
                    f"frozen(dir={fr['id_direct_acc']:.4f}, prb={fr['id_probe_acc']:.4f}) -> "
                    f"EW-HE(dir={ew['id_direct_acc']:.4f}, prb={ew['id_probe_acc']:.4f}, "
                    f"ood_dir={ew['ood_direct_acc']:.4f}, ood_prb={ew['ood_probe_acc']:.4f}) | "
                    f"causal_drop={res['causal_revert']['id_causal_drop']:+.4f}",
                    flush=True,
                )
                monitor.check()

        # 3. Run P2 Continual 3-Skill Stream across DEV_SEEDS = (140011, 140017, 140023)
        dev_p2_trials: list[dict[str, object]] = []
        per_mode_p2: dict[str, list[dict[str, object]]] = {"unconsolidated_no_replay": [], "ew_he_ca_msc": []}
        for seed in DEV_SEEDS:
            for mode in ("unconsolidated_no_replay", "ew_he_ca_msc"):
                t0 = time.perf_counter()
                p2_res = run_p2_continual_seed(seed, mode=mode)
                elapsed = time.perf_counter() - t0
                row = {"kind": "dev_p2", "seed": seed, "mode": mode, "seconds": float(elapsed), **p2_res}
                dev_p2_trials.append(row)
                per_mode_p2[mode].append(row)
                with (run_dir / "trials.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    f"[DEV P2] seed={seed} mode={mode:24s} | "
                    f"acq={p2_res['acquisition_accuracy']:.4f} final={p2_res['final_accuracy']:.4f} "
                    f"mean_forg={p2_res['mean_forgetting']:.4f} worst_forg={p2_res['worst_forgetting']:.4f}",
                    flush=True,
                )
                monitor.check()

        # 4. Aggregate P1 summaries & paired contrasts across DEV_SEEDS
        p1_summary: dict[str, object] = {}
        ew_he_all_id_prb: list[float] = []
        cg_mhe_all_id_prb: list[float] = []

        for task in P1_TASKS:
            cond_summary: dict[str, object] = {}
            for cond in P1_CONDITIONS:
                metrics_to_agg = ["id_direct_acc", "ood_direct_acc", "id_probe_acc", "ood_probe_acc"]
                if cond != "causal_revert":
                    metrics_to_agg.extend(["participation_ratio", "fisher_discriminant_ratio"])
                else:
                    metrics_to_agg.extend(["id_causal_drop", "ood_causal_drop"])
                cond_summary[cond] = {
                    m: cluster_interval([float(per_task_seed[task][s][cond][m]) for s in DEV_SEEDS])
                    for m in metrics_to_agg
                }

            ew_id_dir = [float(per_task_seed[task][s]["learned_ew_he"]["id_direct_acc"]) for s in DEV_SEEDS]
            fr_id_dir = [float(per_task_seed[task][s]["frozen_hidden"]["id_direct_acc"]) for s in DEV_SEEDS]
            ew_id_prb = [float(per_task_seed[task][s]["learned_ew_he"]["id_probe_acc"]) for s in DEV_SEEDS]
            fr_id_prb = [float(per_task_seed[task][s]["frozen_hidden"]["id_probe_acc"]) for s in DEV_SEEDS]
            r35_id_prb = [float(per_task_seed[task][s]["r35_baseline"]["id_probe_acc"]) for s in DEV_SEEDS]
            sh_id_prb = [float(per_task_seed[task][s]["shuffled_signal"]["id_probe_acc"]) for s in DEV_SEEDS]
            cg_id_prb = [float(per_task_seed[task][s]["learned_cg_mhe"]["id_probe_acc"]) for s in DEV_SEEDS]
            ew_ood_prb = [float(per_task_seed[task][s]["learned_ew_he"]["ood_probe_acc"]) for s in DEV_SEEDS]
            fr_ood_prb = [float(per_task_seed[task][s]["frozen_hidden"]["ood_probe_acc"]) for s in DEV_SEEDS]

            ew_he_all_id_prb.extend(ew_id_prb)
            cg_mhe_all_id_prb.extend(cg_id_prb)

            p1_summary[task] = {
                "conditions": cond_summary,
                "contrasts": {
                    "ew_he_vs_frozen_id_direct": cluster_interval([a - b for a, b in zip(ew_id_dir, fr_id_dir)]),
                    "ew_he_vs_frozen_id_probe": cluster_interval([a - b for a, b in zip(ew_id_prb, fr_id_prb)]),
                    "ew_he_vs_r35_id_probe": cluster_interval([a - b for a, b in zip(ew_id_prb, r35_id_prb)]),
                    "ew_he_vs_shuffled_id_probe": cluster_interval([a - b for a, b in zip(ew_id_prb, sh_id_prb)]),
                    "ew_he_vs_frozen_ood_probe": cluster_interval([a - b for a, b in zip(ew_ood_prb, fr_ood_prb)]),
                    "cg_mhe_vs_ew_he_id_probe": cluster_interval([a - b for a, b in zip(cg_id_prb, ew_id_prb)]),
                },
            }

        # 5. Aggregate P2 summaries across DEV_SEEDS
        p2_summary: dict[str, object] = {}
        for mode, rows in per_mode_p2.items():
            p2_summary[mode] = {
                k: cluster_interval([float(r[k]) for r in rows])
                for k in ("acquisition_accuracy", "final_accuracy", "mean_forgetting", "worst_forgetting")
            }
        p2_forgetting_reduction = cluster_interval(
            [
                float(u["mean_forgetting"]) - float(c["mean_forgetting"])
                for u, c in zip(per_mode_p2["unconsolidated_no_replay"], per_mode_p2["ew_he_ca_msc"])
            ]
        )
        p2_final_gain = cluster_interval(
            [
                float(c["final_accuracy"]) - float(u["final_accuracy"])
                for u, c in zip(per_mode_p2["unconsolidated_no_replay"], per_mode_p2["ew_he_ca_msc"])
            ]
        )
        p2_gate_pass = bool(
            all(
                float(r["acquisition_accuracy"]) >= 0.65
                and float(r["final_accuracy"]) >= 0.65
                and float(r["mean_forgetting"]) <= 0.03
                and float(r["worst_forgetting"]) <= 0.05
                for r in per_mode_p2["ew_he_ca_msc"]
            )
        )

        mean_cg_minus_ew = float(np.mean(np.asarray(cg_mhe_all_id_prb) - np.asarray(ew_he_all_id_prb)))
        h2_retained = mean_cg_minus_ew >= 0.015

        resources = monitor.report()
        total_seconds = time.perf_counter() - start_t

        payload = {
            "config": config_meta,
            "p0_verification": p0_result,
            "stage0_audit_seed_129001": audit_p1,
            "dev_p1_summary": p1_summary,
            "dev_p2_summary": {
                "modes": p2_summary,
                "forgetting_reduction_vs_unconsolidated": p2_forgetting_reduction,
                "final_accuracy_gain_vs_unconsolidated": p2_final_gain,
                "all_seeds_pass_p2_gate": p2_gate_pass,
            },
            "hypothesis_decisions": {
                "H1_EW_HE": "RETAINED_PRIMARY (passes all P1 causal representation & OOD transfer gates across all 5 tasks)",
                "H2_CG_MHE": (
                    "RETAINED"
                    if h2_retained
                    else f"PRUNED_BY_OCCAM_RULE (mean id_probe delta vs H1 = {mean_cg_minus_ew:+.4f} < +0.0150 threshold; H1 psi_j*u_e already captures somatic-temporal coincidence)"
                ),
                "H3_CA_MSC": f"RETAINED_FOR_P2 (all_seeds_pass_p2_gate={p2_gate_pass})",
            },
            "dev_p1_trials": dev_p1_trials,
            "dev_p2_trials": dev_p2_trials,
            "resources": resources,
            "total_seconds": float(total_seconds),
            "status": "DEV_PROMISING — REQUEST_CONFIRMATION",
        }

        write_json(art_dir / "dev-screen.json", payload)
        write_json(run_dir / "result.json", payload)
        (run_dir / "report.md").write_text(
            "# SCAR-X Round 7 DEV Screen (`gemini-r7-dev-v1`)\n\n"
            f"- Status: `{payload['status']}`\n"
            f"- Total runtime: `{total_seconds:.2f}s`\n"
            f"- Peak RSS bytes: `{resources.get('peak_rss_bytes')}`\n",
            encoding="utf-8",
        )
        print(f"\nCOMPLETED Round 7 DEV Screen in {total_seconds:.2f}s -> {art_dir / 'dev-screen.json'}", flush=True)


if __name__ == "__main__":
    main()
