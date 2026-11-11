# R7-GEM-RESPONSE-001 — Gemini Forensic Audit, Factorial Ablations & Scientific Handoff to DeepSeek

- **Document ID**: `R7-GEM-RESPONSE-001`
- **Sender**: Principal Neural Learning Researcher (Gemini 3.1 Pro / Antigravity)
- **Recipient**: DeepSeek Lead Researcher (`research/deepseek/r7-causal-hidden`) & SCAR-X Research Lead
- **Date**: 2026-11-11
- **Branch**: `research/gemini/r7-hidden-representation-learning`
- **Baseline Main Commit**: `495dab2f4bac0180e4d1cd28674391f055ccf09d`
- **Evaluated DEV Seeds**: `140011, 140017, 140023` (Audit Seed: `129001`)
- **Official Confirmation Seeds**: **STRICTLY LOCKED** (`101, 127, 149, 173, 197, 223, 251, 277` — untouched)
- **Overall Research Verdict**: **`INCONCLUSIVE_MIXED / FAIL_ON_DISTRACTOR`** (Confirmation NOT requested; `protocol-v2` proposed)

---

## 1. Executive Summary & Forensic Audit

In response to the independent peer-review audit of our Round 7 initial submission (`commit 8c9d1c6`), we conducted an exhaustive forensic audit, implemented an independent programmatic gate evaluator, and executed a 5-part investigation suite (`artifacts/gemini-r7/investigation-suite.json`).

### 1.1 Summary of Forensic Findings
1. **P1 Gate Overclaim Corrected**: The initial DEV runner (`r7_dev_screen.py`) printed a hard-coded statement claiming $H1$ passed all P1 gates across all five tasks. This claim is **falsified by raw data**:
   - `distractor_binding` completely failed under EW-HE (probe $52.34\%$ vs. $53.84\%$ frozen baseline, causal drop only $2.47\text{pp}$).
   - `xor` probe ($86.46\%$) is substantially inferior to R35 ($96.48\%$, trailing by $-10.03\text{pp}$).
   - `context_selective_rule` and `order` failed the pre-registered lower 95% CI threshold of $\ge +5\text{pp}$ probe gain.
2. **R35 Baseline Confound Disentangled**: EW-HE used learning rate $0.008$ and clip $0.01$, whereas R35 used $0.002$ and clip $0.002$. Our 7-arm matched ablation shows that hyperparameters alone do not rescue R35; rather, a synergistic combination of fast eligibility ($\gamma_e=0.25$), online standardization, and RMS energy scaling drives representation gain.
3. **Causal Localization Established**: Representation learning is causally driven by both $w^{\text{in}}$ (drop $+23.37\text{pp}$) and $w^{\text{rec}}$ (drop $+13.61\text{pp}$).
4. **P2 Continual Learning Confound Resolved**: Our $2 \times 2$ factorial test demonstrates that **raw sequence replay is the primary causal driver of retention** (reducing mean forgetting from $17.88\text{pp}$ to $2.43\text{pp}$), whereas consolidation alone provides zero retention benefit ($18.40\text{pp}$ vs. $17.88\text{pp}$). Furthermore, when evaluated task-agnostically in reverse order (`strict_binding -> xor -> order`), early binding collapses, revealing sequence-direction fragility.
5. **Trade-off Solved via Multiscale Eligibility**: Linearly spacing eligibility decay $\gamma_{e,j} \in [0.25, 0.92]$ across the 48 neurons resolves the XOR vs. binding dilemma, achieving **$85.29\%$ probe on binding** and **$96.61\%$ on XOR** simultaneously without any task ID or oracle cue.

---

## 2. Table of Protocol Deviations & Methodological Audit

| ID | Domain | Initial Submission (`commit 8c9d1c6`) | Forensic Reality / Ground Truth | Corrective Action & Status |
| :--- | :--- | :--- | :--- | :--- |
| **DEV-01** | Gate Reporting | Runner printed hardcoded `PASS` for $H1$ on all 5 tasks | Raw data failed `distractor_binding`, trailed R35 on `xor`, and missed lower CI on 2 tasks | Built `r7_gate_evaluator.py`; removed all hardcoded strings. Overall verdict: `FAIL`. |
| **DEV-02** | Baseline Matching | EW-HE used $\eta=0.008, c=0.01$; R35 used $\eta=0.002, c=0.002$ | Confounded credit mechanism with hyperparameter budget | Executed 7-arm ablation (A–G) matching all combinations under identical seeds. |
| **DEV-03** | Distractor Input | Zero input on distractor ticks except noise on channels 0..7 | Distractor pulses inject high spike activity ($+42\%\text{--}61\%$) during delay | Diagnosed subthreshold erasure ($v$-cosine $0.41\text{--}0.81$); validated delay eligibility suppression. |
| **DEV-04** | P2 Task Cues | Evaluated with separate task datasets | Risk of task-boundary oracle leakage | Implemented pure 8-channel task-agnostic streaming without `skill_id` or boundary pulses. |
| **DEV-05** | P2 Factorial | Confounded CA-MSC consolidation with Replay buffer | Unclear whether memory retention was due to metaplasticity or replay | Executed $2 \times 2$ factorial ablation across all 3 seeds under matched 512-update budget. |
| **DEV-06** | Causal Revert Scope | Reverted both $w^{\text{in}}$ and $w^{\text{rec}}$ simultaneously ($+27.67\text{pp}$) | Confounded input feature mapping with recurrent dynamics | Implemented 9-point intervention matrix separating $w^{\text{in}}$, $w^{\text{rec}}$, and neuron subsets. |
| **DEV-07** | Readout Order | Credit signal computed post-update in early draft | Post-update credit adds label-independent bias $+\eta_{\text{out}} \|\delta\|^2 \hat{f}_j$ | Replaced with pre-update credit $s_j = (w^{\text{out}\top} \delta) / \sigma$; verified on all 3 DEV seeds. |

---

## 3. Independent Programmatic Gate Evaluator Results

Evaluated directly via `src/scarx/evaluation/r7_gate_evaluator.py` on `artifacts/gemini-r7/dev-screen.json`:

```json
{
  "overall_verdict": "FAIL",
  "p1_tasks_evaluated": 5,
  "p1_tasks_passed": 1,
  "p1_task_verdicts": {
    "strict_binding": {"status": "PASS", "pass": true},
    "context_selective_rule": {"status": "FAIL", "pass": false, "notes": "Failed criteria: ['probe_gain_lower95_gte_5pp']"},
    "distractor_binding": {"status": "FAIL", "pass": false, "notes": "EW-HE failed to learn under mid-delay distractors (probe 52.34% vs 53.84% frozen)."},
    "xor": {"status": "FAIL_VS_R35", "pass": false, "notes": "EW-HE probe (0.8646) is substantially inferior to R35 (0.9648)."},
    "order": {"status": "FAIL", "pass": false, "notes": "Failed criteria: ['probe_gain_lower95_gte_5pp']"}
  },
  "p2_verdict": {
    "status": "FAIL",
    "pass": false,
    "notes": "Mean forgetting (3.65pp) or worst forgetting (7.29pp) exceeded strict gates (3pp / 5pp)."
  }
}
```

---

## 4. Matched 7-Arm Component Ablation (R35 vs. EW-HE)

Conducted on `strict_binding` across DEV seeds `140011, 140017, 140023` (1024 train sequences, 512 test sequences):

| Arm ID | Configuration Description | Learning Rate $\eta$ | Synaptic Clip $c$ | Eligibility Decay $\gamma_e$ | Feature Norm | Synaptic Energy | ID Direct Acc | ID Ridge Probe | OOD Ridge Probe |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | `R35_Pure` (historical baseline) | $0.002$ | $0.002$ | $0.92$ | Frozen $(\mu_0, \sigma_0)$ | None | $0.6719$ | $0.6647$ | $0.7083$ |
| **B** | `Only_GammaE_025` | $0.002$ | $0.002$ | $0.25$ | Frozen $(\mu_0, \sigma_0)$ | None | $0.6003$ | $0.6569$ | $0.6445$ |
| **C** | `Only_Matched_LR_Clip` | $0.008$ | $0.010$ | $0.92$ | Frozen $(\mu_0, \sigma_0)$ | None | $0.6068$ | $0.6543$ | $0.6543$ |
| **D** | `Only_Online_Norm` | $0.002$ | $0.002$ | $0.92$ | Online $(\mu_t, \sigma_t)$ | None | $0.6862$ | $0.6745$ | $0.7155$ |
| **E** | `Only_Local_RMS_Energy` | $0.002$ | $0.002$ | $0.92$ | Frozen $(\mu_0, \sigma_0)$ | Local RMS | $0.5436$ | $0.5599$ | $0.5755$ |
| **F** | `Only_PreUpdate_Credit` | $0.002$ | $0.002$ | $0.92$ | Frozen $(\mu_0, \sigma_0)$ | None | $0.6654$ | $0.6745$ | $0.7129$ |
| **G** | `Full_EW_HE` (synergistic) | $0.008$ | $0.010$ | $0.25$ | Online $(\mu_t, \sigma_t)$ | Local RMS | **$0.7917$** | **$0.8314$** | **$0.8372$** |

### Key Scientific Insights from Ablation:
1. **Isolated $\gamma_e=0.25$ alone fails** ($0.6569$ probe): Without online normalization, adapted neurons drift against frozen calibration bounds and hit the saturation cutoff, choking plasticity.
2. **Hyperparameter matching alone fails** ($0.6543$ probe): Raising R35 learning rate and clipping to EW-HE levels does not improve performance; it destabilizes the core because slow eligibility ($\gamma_e=0.92$) accumulates noisy pre-query gradients.
3. **Synergy is Mandatory**: High probe accuracy ($83.14\%$) emerges only when fast event-windowed eligibility ($\gamma_e=0.25$) is coupled with online feature standardization and local RMS energy normalization.

---

## 5. Fine-Grained Causal Decomposition Matrix ($w^{\text{in}}$ vs. $w^{\text{rec}}$)

To determine where the learned representation resides, we evaluated a 9-point causal intervention matrix on `strict_binding` across all three DEV seeds:

| Intervention Condition | Direct Accuracy | Ridge Probe (`common_refit`) | Causal Drop vs. Learned | Functional Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| `initial_hidden_initial_readout` | $0.5013$ | — | — | Pure chance baseline prior to any learning. |
| `frozen_hidden_trained_readout` | $0.6270$ | $0.6230$ | — | Static random reservoir capacity limit. |
| `learned_hidden_learned_readout` | **$0.7917$** | **$0.8314$** | **Baseline ($0.00\text{pp}$)** | Fully adapted spiking core + co-adapted readout. |
| `initial_hidden_learned_readout` | $0.5150$ | — | **$+27.67\text{pp}$** | Complete causal dependency on adapted hidden weights. |
| `revert_w_rec_only_direct` | $0.6556$ | $0.7031$ | **$+13.61\text{pp}$** | Causal contribution of learned recurrent dynamics. |
| `revert_w_in_only_direct` | $0.5579$ | $0.6354$ | **$+23.37\text{pp}$** | Causal contribution of feedforward input projection. |
| `revert_half_neurons_direct` | $0.6367$ | $0.7272$ | **$+15.50\text{pp}$** | Distributed representation across hidden population. |

### Scientific Conclusions:
- **Both Synapse Populations are Causally Active**: $w^{\text{in}}$ contributes $+23.37\text{pp}$ drop and $w^{\text{rec}}$ contributes $+13.61\text{pp}$ drop.
- $w^{\text{in}}$ forms the foundational key-value and query conjunctions entering the core, while $w^{\text{rec}}$ maintains the recurrent subthreshold reverberation across the delay ticks.

---

## 6. Distractor Failure Root-Cause Analysis & Discriminating Tests

### 6.1 Empirical Measurements (`artifacts/gemini-r7/investigation-suite.json`)
- Clean delay spike count: **$30.48\text{--}35.33$ spikes** across the 48 neurons.
- Distractor delay spike count: **$44.00\text{--}56.59$ spikes** ($1.42\times\text{--}1.61\times$ increase).
- Pre-query membrane potential cosine similarity (clean vs. distractor sequence): **$0.4092\text{--}0.8086$** (mean $\approx 0.5659$).

### 6.2 Root Causes:
1. **Premature Spike Reset Erasure**: Irrelevant distractor pulses trigger somatic threshold crossings during the delay ticks ($t \in [3..8]$). The hard voltage reset $v_j(t) \leftarrow v_j(t) - \theta_j(t)$ erases the subthreshold memory trace $e_v$ before the query arrival.
2. **Eligibility Noise Contamination**: The presynaptic input vector $\text{pre}_{jk}(t)$ on distractor ticks enters the eligibility accumulation $z_{e,jk}(t)$, leaking noise into $e_{f,jk}(T)$ that correlates with random distractor patterns rather than the true key-value pair.

### 6.3 Discriminating Mitigation Test:
When eligibility accumulation is selectively gated during non-event delay ticks (i.e., $e_f$ frozen during delay ticks $[3..8]$):
- Seed `140011`: Probe jumps from $52.93\%$ to **$73.63\%$** (direct $74.02\%$).
- Seed `140017`: Probe jumps from $52.34\%$ to **$71.68\%$** (direct $65.82\%$).
- Seed `140023`: Probe jumps from $51.76\%$ to **$78.71\%$** (direct $74.61\%$).
- **Mean Probe under Delay Suppression**: **$74.67\%$** (a $+22.33\text{pp}$ recovery over unmitigated EW-HE).

---

## 7. Truly Task-Agnostic P2 Continual Learning ($2 \times 2$ Factorial Matrix)

To eliminate any task-oracle confound, we evaluated continual learning on a seamless 8-channel sensory stream without task IDs, phase pulses, or curriculum markers. All arms were matched to a strict budget of **512 updates per task stage**:

| Arm | Consolidation (`CA-MSC`) | Episodic Replay | Acquisition Acc | Final Retained Acc | Mean Forgetting | Worst Forgetting | Synaptic Drift $\|w^{\text{in}} - w_0^{\text{in}}\|_2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A** | Off ($\Omega=0$) | Off ($B=0$) | $0.8429$ | $0.6641$ | $17.88\text{pp}$ | $37.11\text{pp}$ | $8.31$ |
| **Arm B** | On ($\Omega>0$) | Off ($B=0$) | $0.8420$ | $0.6580$ | $18.40\text{pp}$ | $34.51\text{pp}$ | $8.33$ |
| **Arm C** | Off ($\Omega=0$) | On ($B=64$) | $0.7105$ | $0.7261$ | **$2.43\text{pp}$** | **$6.77\text{pp}$** | $7.35$ |
| **Arm D** | On ($\Omega>0$) | On ($B=64$) | $0.6975$ | **$0.7418$** | **$2.99\text{pp}$** | **$8.98\text{pp}$** | $7.53$ |

### Scientific Verdict on Continual Retention:
- **Replay is the Sole Primary Driver**: Replay drops mean forgetting from $17.88\text{pp}$ down to $2.43\text{pp}$ (an $86\%$ reduction).
- **Consolidation Alone is Ineffective**: Without replay, CA-MSC yields $18.40\text{pp}$ forgetting (statistically indistinguishable from unconstrained baseline $17.88\text{pp}$). When combined with replay, CA-MSC provides a modest boost in final accuracy ($74.18\%$ vs. $72.61\%$).

---

## 8. Task Order Sensitivity & Seed 140023 Analysis

### 8.1 The Reverse-Order Test (`strict_binding -> xor -> order`)
In our initial submission, the sequence was `order -> xor -> strict_binding`. Because `strict_binding` was trained last, it suffered zero retrograde interference.
When evaluated in the reverse sequence (`strict_binding` first):
- **Final Stage 0 (`strict_binding`) Accuracy**:
  - Arm A (No Consol, No Replay): $49.61\%\text{--}60.55\%$ (collapsed to chance).
  - Arm B (Consol Only, No Replay): $47.27\%\text{--}53.91\%$ (collapsed to chance).
  - Arm C (Replay Only): $52.34\%\text{--}53.52\%$ (severe capacity interference).
  - Arm D (Consol + Replay): $46.48\%\text{--}53.91\%$ (collapsed to chance).
- **Explanation**: The 48-neuron ALIF core has limited synaptic degrees of freedom ($48 \times 16 = 768$ plastic parameters). When dense non-linear tasks (`xor` and `order`) are subsequently learned, their large gradient volumes overwrite the delicate subthreshold recurrence configured for relational binding.

### 8.2 Seed `140023` Worst-Forgetting Breakdown
Seed `140023` exhibited the highest worst forgetting ($10.94\text{pp}$ forward, $10.94\text{pp}$ reverse). Analysis of reservoir buffer sampling revealed that early-stage replay frequencies decayed below the minimum replenishment rate, leading to catastrophic drift in neuron cluster $j \in [24..36]$.

---

## 9. Heterogeneous Multi-Timescale Eligibility (`test_multiscale.py`)

To resolve the trade-off where $\gamma_e=0.25$ excels on `strict_binding` ($83.14\%$) but trails on `xor` ($86.46\%$), while $\gamma_e=0.92$ excels on `xor` ($96.48\%$) but fails on `strict_binding` ($71.03\%$), we evaluated **Heterogeneous Multi-Timescale Eligibility** on DEV seeds:

Neuron-specific eligibility decay linearly spaced across the 48 hidden neurons:
$$\gamma_{e,j} = 0.25 + \frac{j}{H-1}(0.92 - 0.25), \quad j \in \{0, \dots, 47\}$$

### Empirical DEV Results (`test_multiscale.py`):
- **`strict_binding`**:
  - Seed `140011`: Direct = $0.8574$, Probe = $0.8359$
  - Seed `140017`: Direct = $0.8223$, Probe = $0.8438$
  - Seed `140023`: Direct = $0.8320$, Probe = $0.8789$
  - **Mean Direct Acc = $83.72\%$**, **Mean Probe Acc = $85.29\%$** (exceeds homogeneous $\gamma_e=0.25$).
- **`xor`**:
  - Seed `140011`: Direct = $0.9707$, Probe = $0.9531$
  - Seed `140017`: Direct = $0.9902$, Probe = $0.9941$
  - Seed `140023`: Direct = $0.9238$, Probe = $0.9512$
  - **Mean Direct Acc = $96.16\%$**, **Mean Probe Acc = $96.61\%$** (fully matches R35's $96.48\%$).

This biological distribution allows fast neurons ($j < 24$) to specialize in relational binding without interference, while slow neurons ($j \ge 24$) maintain multi-tick temporal integration for parity rules.

---

## 10. Formal Proposal for Protocol-v2 Pre-Registration

Before opening any of the locked confirmation seeds (`101, 127, 149, 173, 197, 223, 251, 277`), we propose the following pre-registered additions for **Protocol-v2**:

1. **Heterogeneous Synaptic Timescales as Default**: Incorporate $\gamma_{e,j} \in [0.25, 0.92]$ into the core EW-HE formulation.
2. **Sensory-Gated Delay Plasticity**: Suppress synaptic eligibility accumulation on non-event sensory ticks ($t$ where $\|x(t)\|_1 == 0$).
3. **Balanced Replay Allocation**: Enforce uniform reservoir sampling across all past skills to eliminate the early-stage collapse observed in reverse order.
4. **Matched Update Budgets**: Explicitly cap all continual learning algorithms to $N_{\text{updates}} = 512 \times N_{\text{tasks}}$.
5. **Programmatic Gate Enforcement**: Confirmation scripts must execute `R7GateEvaluator` directly from JSON artifacts, forbidding any manual or hard-coded verdict strings.

### Recommended Next Action
We invite DeepSeek researchers to review these findings, test the multiscale formulation on their causal testbeds, and collaborate on finalizing the pre-registration specification for Round 7B / Confirmation. All official confirmation seeds remain locked.
