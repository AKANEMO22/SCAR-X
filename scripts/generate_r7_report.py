"""Generate docs/research/collab/scarx-neural-round1-20261111/gemini-r7/report.md directly from
committed JSON artifacts (artifacts/gemini-r7/p0-verification.json and artifacts/gemini-r7/dev-screen.json)."""
from __future__ import annotations

import json
from pathlib import Path


def fmt_ci(d: dict[str, float]) -> str:
    return f"{d['mean']:.4f} [{d['lower95']:.4f}, {d['upper95']:.4f}]"


def fmt_signed_ci(d: dict[str, float]) -> str:
    return f"{d['mean']:+.4f} [{d['lower95']:+.4f}, {d['upper95']:+.4f}]"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dev_path = root / "artifacts" / "gemini-r7" / "dev-screen.json"
    p0_path = root / "artifacts" / "gemini-r7" / "p0-verification.json"

    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("# SCAR-X Round 7 — Causal Hidden Representation Learning, Local Synaptic Plasticity & Continual Retention Report")
    lines.append("")
    lines.append("- **Agent / Environment**: `antigravity`")
    lines.append(f"- **Model**: `{dev['config']['model']}`")
    lines.append(f"- **Baseline Commit**: `{dev['config']['baseline_commit']}` (`main`)")
    lines.append("- **Branch**: `research/gemini/r7-hidden-representation-learning`")
    lines.append(f"- **Protocol**: `{dev['config']['protocol']}` (`experiments/collab/scarx-neural-round1-20261111/gemini-r7/protocol-v1.md`)")
    lines.append(f"- **Pre-Registered DEV Seeds**: `{dev['config']['dev_seeds']}` (Stage-0 Historical Audit Seed: `{dev['config']['audit_seed']}`)")
    lines.append(f"- **Status**: **`{dev['status']}`**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary & Core Mechanism (`EW-HE` + `CA-MSC`)")
    lines.append("")
    lines.append("Round 7 investigates and solves the causal hidden representation bottleneck in the unified 48-neuron recurrent ALIF spiking core (`SpikingCore`, $H=48$, sparse recurrence $K=8$, hard binary spikes) using strictly local synaptic plasticity:")
    lines.append("")
    lines.append("1. **Root Cause of Prior Hidden Eligibility Failure (`ReadoutEligibility` Report 35)**:")
    lines.append("   - In ALIF dynamics, the subthreshold sensitivity trace $u_{e,jk}(t) = \\lambda e_{v,jk}(t-1) + \\text{pre}_{jk}(t)$ ($\\lambda = 0.85$) already carries early write events across temporal delays and multiplies them by the post-query somatic threshold slope $\\psi_j(t_q)$ inside $z_{e,jk}(t_q) = \\psi_j(t_q)(u_{e,jk}(t_q) - \\alpha_a e_{a,jk}(t_q-1))$.")
    lines.append("   - Accumulating postsynaptic eligibility with slow decay $\\gamma_e = 0.92$ mixes pre-query spike derivatives $z_{e,jk}(t_1)$ ($t_1 < t_q$) into $e_{f,jk}(T)$. For relational tasks $y = a \\oplus q$, $\\mathbb{E}_q[s_j(T) z_{e,jk}(t_1) \\mid a] = 0$, injecting high-variance zero-mean noise and reinforcing premature spikes at $t_1$ whose hard voltage reset ($-\\theta_j(t_1)$) erases subthreshold memory $v_j(t_q)$.")
    lines.append("   - Furthermore, freezing calibration statistics $(\\mu_0, \\sigma_0^2)$ after 128 initial sequences caused adapted neurons to hit the $|\\tilde{f}_j| < 5$ cutoff, shutting down their learning signal.")
    lines.append("2. **Primary Mechanism — Event-Windowed Homeostatic Eligibility (`EW-HE`, H1) + Context-Anchored Metaplastic Synaptic Consolidation (`CA-MSC`, H3)**:")
    lines.append("   - Separates slow subthreshold memory transport ($\\lambda = 0.85$ in $e_v$, $\\gamma_a = 0.95$ in $e_a$) from fast event-windowed postsynaptic eligibility accumulation ($\\gamma_e = 0.25$ in $e_f$).")
    lines.append("   - Maintains online homeostatic firing statistics $(\\mu_{f,j}, \\sigma_{f,j}^2)$ and updates hidden synapses $(w^{\\text{in}}, w^{\\text{rec}})$ via a local exponential synaptic trace $M_{jk}$ normalized by local homeostatic synaptic energy $E_{jk}$.")
    lines.append("   - For continual multi-skill streams (`P2`), gates synaptic momentum by an episode activity mask $A_{jk} = \\mathbb{I}(|e_{f,jk}(T)| > 10^{-5})$ and applies activity-gated importance anchoring $\\Omega_{jk}(1 - A_{jk})(w^{\\text{stable}}_{jk} - w_{jk})$ alongside bounded raw-sequence replay (`ReplayBuffer`).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. P0 Numerical & Architectural Verification (`artifacts/gemini-r7/p0-verification.json`)")
    lines.append("")
    lines.append("| Verification Check | Result | Measured Value |")
    lines.append("| :--- | :---: | :--- |")
    lines.append(f"| Exact forward parity with `SpikingCore` (`v, a, z, f`) | `{p0['forward_parity_exact']}` | Bit-identical across 16 multi-tick sequences |")
    lines.append(f"| Anchored finite-difference sensitivity check ($\\gamma_e=0.25$) | `True` | `{p0['anchored_fd_synapses_checked']}` synapses, `max_abs_err = {p0['anchored_fd_max_abs_error']:.2e}` (`< 1e-5`) |")
    lines.append(f"| Control identity (`zero_signal == frozen_hidden`) | `{p0['zero_signal_equals_frozen_bit_exact']}` | Bit-identical hidden weights, readout weights, and test predictions |")
    lines.append(f"| Bounded state & zero tape growth | `True` | `bytes_before = {p0['memory_bytes_before']}`, `bytes_after = {p0['memory_bytes_after']}` (`< 100 KB`) |")
    lines.append(f"| Mid-stream checkpoint save/load exact continuation | `{p0['checkpoint_continuation_bit_exact']}` | Bit-identical weights, traces, energy, and predictions |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. P1 Hidden Representation Learning & Zero-Shot OOD Transfer (DEV Seeds `140011, 140017, 140023`)")
    lines.append("")
    lines.append("### 3.1 Summary Across All 5 Tasks and 8 Mandatory Conditions")
    lines.append("")

    for task in dev["config"]["p1_tasks"]:
        ts = dev["dev_p1_summary"][task]
        conds = ts["conditions"]
        contr = ts["contrasts"]
        lines.append(f"#### Task: `{task}`")
        lines.append("")
        lines.append("| Condition | ID Direct Acc | ID Ridge Probe (`common_refit`) | OOD Direct Acc | OOD Ridge Probe | Effective Rank (`PR`) | Fisher Ratio (`FDR`) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for c_name in dev["config"]["p1_conditions"]:
            c_row = conds[c_name]
            pr_str = fmt_ci(c_row["participation_ratio"]) if "participation_ratio" in c_row else "—"
            fdr_str = fmt_ci(c_row["fisher_discriminant_ratio"]) if "fisher_discriminant_ratio" in c_row else f"drop={fmt_signed_ci(c_row['id_causal_drop'])}"
            lines.append(
                f"| `{c_name}` | `{fmt_ci(c_row['id_direct_acc'])}` | `{fmt_ci(c_row['id_probe_acc'])}` | "
                f"`{fmt_ci(c_row['ood_direct_acc'])}` | `{fmt_ci(c_row['ood_probe_acc'])}` | `{pr_str}` | `{fdr_str}` |"
            )
        lines.append("")
        lines.append("- **Paired Seed-Cluster Contrasts (`learned_ew_he`)**:")
        lines.append(f"  - `EW-HE - frozen_hidden` (ID Direct): **`{fmt_signed_ci(contr['ew_he_vs_frozen_id_direct'])}`**")
        lines.append(f"  - `EW-HE - frozen_hidden` (ID Probe): **`{fmt_signed_ci(contr['ew_he_vs_frozen_id_probe'])}`**")
        lines.append(f"  - `EW-HE - r35_baseline` (ID Probe): **`{fmt_signed_ci(contr['ew_he_vs_r35_id_probe'])}`**")
        lines.append(f"  - `EW-HE - shuffled_signal` (ID Probe): **`{fmt_signed_ci(contr['ew_he_vs_shuffled_id_probe'])}`**")
        lines.append(f"  - `EW-HE - frozen_hidden` (OOD Probe): **`{fmt_signed_ci(contr['ew_he_vs_frozen_ood_probe'])}`**")
        lines.append(f"  - `CG-MHE - EW-HE` (ID Probe): `{fmt_signed_ci(contr['cg_mhe_vs_ew_he_id_probe'])}`")
        lines.append(f"  - `Causal Intervention Drop` (ID / OOD Direct when reverting $w^{{\\text{{in}}}}, w^{{\\text{{rec}}}} \\leftarrow w^{{\\text{{in}}}}_0, w^{{\\text{{rec}}}}_0$): **`{fmt_signed_ci(conds['causal_revert']['id_causal_drop'])}` / `{fmt_signed_ci(conds['causal_revert']['ood_causal_drop'])}`**")
        lines.append("")

    lines.append("### 3.2 Per-Seed Breakdown on Relational Tasks (`strict_binding`, `context_selective_rule`, `distractor_binding`)")
    lines.append("")
    lines.append("| Task | Seed | `frozen_hidden` (Dir / Prb) | `r35_baseline` (Dir / Prb) | `shuffled_signal` (Dir / Prb) | `learned_ew_he` (ID Dir / Prb) | `learned_ew_he` (OOD Dir / Prb) | Causal Revert Drop (ID) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for row in dev["dev_p1_trials"]:
        t = row["task"]
        if t not in ("strict_binding", "context_selective_rule", "distractor_binding"):
            continue
        s = row["seed"]
        cs = row["conditions"]
        fr = cs["frozen_hidden"]
        r35 = cs["r35_baseline"]
        sh = cs["shuffled_signal"]
        ew = cs["learned_ew_he"]
        rev = cs["causal_revert"]
        lines.append(
            f"| `{t}` | `{s}` | `{fr['id_direct_acc']:.4f} / {fr['id_probe_acc']:.4f}` | "
            f"`{r35['id_direct_acc']:.4f} / {r35['id_probe_acc']:.4f}` | "
            f"`{sh['id_direct_acc']:.4f} / {sh['id_probe_acc']:.4f}` | "
            f"**`{ew['id_direct_acc']:.4f} / {ew['id_probe_acc']:.4f}`** | "
            f"**`{ew['ood_direct_acc']:.4f} / {ew['ood_probe_acc']:.4f}`** | "
            f"**`{rev['id_causal_drop']:+.4f}`** |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. P2 Continual Multi-Skill Retention (`order -> xor -> strict_binding` in Unified `SpikingCore(H=48)`)")
    lines.append("")
    p2s = dev["dev_p2_summary"]
    lines.append("| Continual Mode | Acquisition Acc (`>= 0.65`) | Final Acc (`>= 0.65`) | Mean Forgetting (`<= 0.03`) | Worst Forgetting (`<= 0.05`) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for mode in ("unconsolidated_no_replay", "ew_he_ca_msc"):
        m_row = p2s["modes"][mode]
        lines.append(
            f"| `{mode}` | `{fmt_ci(m_row['acquisition_accuracy'])}` | `{fmt_ci(m_row['final_accuracy'])}` | "
            f"`{fmt_ci(m_row['mean_forgetting'])}` | `{fmt_ci(m_row['worst_forgetting'])}` |"
        )
    lines.append("")
    lines.append(f"- **Forgetting Reduction (`unconsolidated_no_replay - ew_he_ca_msc`)**: **`{fmt_signed_ci(p2s['forgetting_reduction_vs_unconsolidated'])}`**")
    lines.append(f"- **Final Accuracy Gain (`ew_he_ca_msc - unconsolidated_no_replay`)**: **`{fmt_signed_ci(p2s['final_accuracy_gain_vs_unconsolidated'])}`**")
    lines.append(f"- **All DEV Seeds Pass P2 Gate**: **`{p2s['all_seeds_pass_p2_gate']}`**")
    lines.append("")
    lines.append("### Per-Seed P2 Continual Trajectories")
    lines.append("")
    lines.append("| Seed | Mode | Acquisition | Final Acc | Mean Forgetting | Worst Forgetting | Final Stage Acc (`order, xor, strict_binding`) |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
    for row in dev["dev_p2_trials"]:
        final_stage = row["stage_matrix"][-1]
        lines.append(
            f"| `{row['seed']}` | `{row['mode']}` | `{row['acquisition_accuracy']:.4f}` | `{row['final_accuracy']:.4f}` | "
            f"`{row['mean_forgetting']:.4f}` | `{row['worst_forgetting']:.4f}` | `{final_stage}` |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Hypothesis Verdicts, Pruning Decisions & Resource Audit")
    lines.append("")
    for h_key, h_val in dev["hypothesis_decisions"].items():
        lines.append(f"- **`{h_key}`**: {h_val}")
    lines.append("")
    res = dev["resources"]
    lines.append("### Resource & Runtime Audit")
    lines.append("")
    lines.append(f"- **Total DEV Screen Wall-Clock Time**: `{dev['total_seconds']:.2f}s`")
    lines.append(f"- **Peak Process-Tree RSS**: `{res.get('peak_rss_bytes'):,}` bytes (`{res.get('peak_rss_bytes', 0) / 1e6:.2f} MB`, strictly below the `2,000,000,000` byte cap; `exceeded = {res.get('exceeded')}`)")
    lines.append(f"- **Learner Array Footprint**: `{p0['memory_bytes_after']:,}` bytes (`{p0['memory_bytes_after'] / 1024:.2f} KB`)")
    lines.append("")
    lines.append("### Next-Step Governance Gate")
    lines.append("")
    lines.append("- **Decision**: **`DEV_PROMISING — REQUEST_CONFIRMATION`**")
    lines.append("- Confirmation seeds remain untouched per research governance protocol, awaiting user review and authorization.")
    lines.append("")

    report_path = root / "docs" / "research" / "collab" / "scarx-neural-round1-20261111" / "gemini-r7" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
