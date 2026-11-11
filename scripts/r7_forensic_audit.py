"""Forensic Audit of Protocol-v1 vs Code Implementation in Round 7."""
from __future__ import annotations

import json
from pathlib import Path


def audit_protocol_vs_code() -> dict[str, object]:
    deviations: list[dict[str, str]] = [
        {
            "item": "p1_gate_specification",
            "protocol_v1_text": "Protocol mentions minimum_accuracy >= 0.65 and representation_gain >= 0.05 from Phase 2.1 configs, but does not formalize per-task gates vs r35_baseline or distractor tolerances.",
            "code_reality": "scripts/r7_dev_screen.py had hardcoded hypothesis_decisions['H1_EW_HE'] claiming 'passes all P1 causal representation & OOD transfer gates across all 5 tasks'.",
            "severity": "CRITICAL",
            "impact": "False sense of complete success when distractor_binding failed (52.34%) and XOR trailed R35 (86.46% vs 96.48%).",
            "remedy": "Replace hardcoded text with programmatic gate evaluator that reports PASS, FAIL, or P1_GATE_UNDER_SPECIFIED based on raw data.",
        },
        {
            "item": "distractor_binding_channels_and_timing",
            "protocol_v1_text": "Section 3.1 specified random distractor pulses on channels 6 and 7 during delay ticks {3, 7, 8}.",
            "code_reality": "r7_hidden_plasticity.py distractor_binding_task only pulsed channel 6 (x[i, dt, 6] += +/-1.2), omitting channel 7 because channel 7 was reserved for end-marker at tick 11.",
            "severity": "MODERATE",
            "impact": "Code did not pulse channel 7 during delay to avoid colliding with end-marker channel semantics.",
            "remedy": "Explicitly document this deviation and formalize in protocol-v2.",
        },
        {
            "item": "r35_baseline_hyperparameter_mismatch",
            "protocol_v1_text": "Section 3.2 specified r35_baseline as slow eligibility decay (gamma_e = 0.92) control.",
            "code_reality": "In r7_hidden_plasticity.py, condition='r35_baseline' was run with lr_hid=0.002, clip=0.002, while learned_ew_he ran with lr_hid=0.008, clip=0.010. Furthermore, r35_baseline ran with online homeostatic standardization while original Report 35 ran with frozen calibrate().",
            "severity": "HIGH",
            "impact": "R35 baseline was confounded with learning rate and clipping differences, obscuring whether the gain came from gamma_e=0.25 or from lr/clip.",
            "remedy": "Implement a full factorial ablation suite matching lr, clip, decay, and normalization individually.",
        },
        {
            "item": "p2_oracle_skill_cue",
            "protocol_v1_text": "Section 4 specified sequential 3-skill continual learning (order -> xor -> strict_binding) in unified SpikingCore(H=48).",
            "code_reality": "wrap_skill_pulse() expanded input dimension from D=8 to D=8+3=11, and injected a distinct 0.8 pulse on channel 8+s_id at tick 11 for each skill.",
            "severity": "HIGH",
            "impact": "This provided an implicit task context cue (task-aware continual learning), rather than pure task-agnostic continual learning from sensory input alone.",
            "remedy": "Label existing result as TASK_AWARE_P2, and design a truly TASK_AGNOSTIC_P2 with no skill cue channels.",
        },
        {
            "item": "ca_msc_replay_confounding",
            "protocol_v1_text": "Section 4 compared unconsolidated_no_replay vs ew_he_ca_msc.",
            "code_reality": "unconsolidated_no_replay had CA-MSC OFF and Replay OFF. ew_he_ca_msc had CA-MSC ON and Replay ON (sampling 1 replayed episode per fresh episode, doubling the effective update budget!).",
            "severity": "HIGH",
            "impact": "Cannot attribute forgetting reduction specifically to synaptic consolidation vs replay buffer vs 2x update budget.",
            "remedy": "Run 2x2 factorial matrix (CA-MSC on/off x Replay on/off) with matched update budgets.",
        },
        {
            "item": "causal_revert_scope",
            "protocol_v1_text": "Section 3.2 specified reverting (w_in, w_rec) <- (w_in_0, w_rec_0) while keeping learned readout.",
            "code_reality": "Only reverted both w_in and w_rec together, without decomposing the separate causal contributions of w_in vs w_rec.",
            "severity": "MODERATE",
            "impact": "Does not prove whether recurrent synapses w_rec or input synapses w_in drove the representation change.",
            "remedy": "Run separate w_in-only and w_rec-only revert interventions.",
        },
        {
            "item": "p0_anchored_surrogate_limitation",
            "protocol_v1_text": "Section 2 specified anchored finite-difference sensitivity check.",
            "code_reality": "test_p0_2 verifies own-neuron forward sensitivity d f_j / d w_j holding presynaptic spike train fixed. It omits the recurrent network Jacobian (d z_m / d z_j) and hard-spike threshold discontinuities.",
            "severity": "INFORMATIONAL",
            "impact": "Must be transparently communicated so external reviewers do not mistake it for full-network BPTT gradient check.",
            "remedy": "State exact mathematical scope and omitted Jacobian terms in report and peer review packet.",
        },
    ]
    return {"deviations": deviations}


if __name__ == "__main__":
    report = audit_protocol_vs_code()
    print(json.dumps(report, indent=2))
