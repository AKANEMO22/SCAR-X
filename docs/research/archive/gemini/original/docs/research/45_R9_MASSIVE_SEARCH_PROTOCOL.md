# SCAR-X ROUND 9: PRE-REGISTERED RESEARCH PROTOCOL
# MASSIVE MULTI-AGENT NEURAL MECHANISM DISCOVERY

**Date:** November 12, 2026  
**Lead Research Architect:** Gemini  
**Branch:** `research/gemini/r9-massive-neural-search-20261112`  
**Base Commit:** `607ffca7c9222c5e3fd9741c3ebdf5939e10bedf`  
**Repository:** https://github.com/AKANEMO22/SCAR-X  

---

## 1. Executive Mandate & Independent Research Protocol

This document pre-registers the complete experimental protocol, hypothesis space, search methodology, statistical gates, and confirmation criteria for SCAR-X Round 9 prior to executing search runs.

### 1.1 Parallel Agent Status & Codex Segregation
- **Codex Activity:** Codex is investigating eligibility propagation through subtractive spike reset in parallel local worktrees.
- **Segregation Invariants:**
  - Gemini operates exclusively on branch `research/gemini/r9-massive-neural-search-20261112`.
  - Port `8767` and active Codex processes (PID 7176, etc.) are strictly protected and left untouched.
  - No Codex worktrees or files are modified.
  - Codex results have not been pushed to remote origin (`CODEX_RESULTS_NOT_ACCESSIBLE`). Research proceeds with independent discovery and zero speculative data leakage.

### 1.2 Sub-Agent Execution Reality
- **Tool Inspection:** The execution environment provides standard tool interfaces (`run_command`, `read_file`, `write_file`, `edit_file`, `send_message`) but lacks an automated LLM agent spawner API (`spawn_agent`).
- **Audit Declaration:** As required by Section 2.7 of the mandate, we truthfully record `SUBAGENT_UNAVAILABLE`.
- **Architectural Specialization:** Research roles are modularized into separate, dedicated algorithmic modules, test harnesses, and audit pipelines:
  - **Agent A (Coordinator):** Multi-stage search scheduler, resource Governor, adaptive budget allocation.
  - **Agent B (Eligibility & Credit Assignment):** H1 mechanisms (subtractive reset Jacobian, surrogate gradients, adaptive trace decay).
  - **Agent C (Hidden Representation & Timescales):** H2 mechanisms (factorial timescale distributions x sensory event gating).
  - **Agent D (Distractor Robustness & State Preservation):** H3 mechanisms (input attenuation, noise-sensitive credit, subthreshold state retention).
  - **Agent E (Readout & Associative Memory):** H4 mechanisms (2/4/8/16-key retrieval, CKCER, error attribution).
  - **Agent F (Continual Learning & Interference):** H5 mechanisms (BF-Cascade 2/3-compartment metaplasticity, orthogonal updates, matched-budget replay).
  - **Agent G (Benchmark Integrity & Statistics):** Input D=8 verification, binary spike check, paired metrics, Benjamini-Hochberg FDR control.
  - **Agent H (Independent Red-Team & Falsification):** Adversarial stress testing, counterexamples, strict causal ablations without the `direct >= 95%` loophole.

---

## 2. Hypothesis Space & Pre-Registered Candidate Budget (512 Configurations)

The discovery campaign targets an initial budget of 512 systematically generated candidate configurations across 5 core hypothesis groups:

| Group | Hypothesis | Focus Area | Planned Configs |
|---|---|---|---:|
| **H1** | Temporal Credit & Reset | Subtractive reset Jacobian vs Hard reset, surrogate gradient profiles | 160 |
| **H2** | Hidden Representation & Timescales | Factorial matrix: uniform/heterogeneous timescales x gating ON/OFF | 96 |
| **H3** | Distractor Robustness & State Preservation | Subthreshold voltage preservation, learned sensory attenuation | 96 |
| **H4** | Readout & Associative Memory | Scaled associative memory (2, 4, 8, 16 keys), context gating, CKCER | 80 |
| **H5** | Continual Learning & Interference | Multi-timescale synaptic cascade (BF-Cascade), orthogonal updates | 80 |
| **Total** | | | **512** |

### 2.1 Group H1: Temporal Credit Assignment & Subtractive Spike Reset (160 Configs)
- **Problem Statement:** In ALIF SpikingCore, spike generation triggers subtractive reset $v(t) \leftarrow u(t) - \theta(t)$, unlike integrate-and-fire models that clamp $v(t) \leftarrow 0$. Standard e-prop derived for hard reset drops eligibility upon spiking.
- **Mathematical Formulations:**
  - *Full Subtractive Jacobian:*
    $$e_v(t) = u_e - \theta(t) \psi(t) [u_e - \beta \rho e_a(t-1)] - z(t) \beta \rho e_a(t-1)$$
  - *Hard Reset Baseline:*
    $$e_v(t) = u_e \cdot (1 - z(t)) - \theta(t) \psi(t) [u_e - \beta \rho e_a(t-1)]$$
  - *Damped Subtractive Reset:*
    $$e_v(t) = (u_e - \theta(t) z_e) \cdot (1 - \lambda_{\text{reset}} z(t)), \quad \lambda_{\text{reset}} \in [0.0, 1.0]$$
  - *Surrogate Gradients:*
    - FastSigmoid: $\psi(x) = \frac{\gamma}{(1 + |x|)^2}, \quad \gamma \in \{0.2, 0.4, 0.6, 0.8, 1.0\}$
    - Exponential: $\psi(x) = \gamma \exp(-|x| / \sigma), \quad \sigma \in \{0.2, 0.5, 1.0\}$
    - Rectangular: $\psi(x) = \gamma \cdot \mathbb{I}(|x| < \delta), \quad \delta \in \{0.3, 0.5, 0.8\}$
    - Smooth Gaussian: $\psi(x) = \frac{\gamma}{\sqrt{2\pi}\sigma} \exp(-x^2 / (2\sigma^2))$

### 2.2 Group H2: Factorial Multi-Timescale & Sensory Gating (96 Configs)
- **Factorial Experimental Design:**
  - Factor 1 (Timescale Decay Distribution):
    - `uniform_fast`: $\gamma_e = 0.25$
    - `uniform_mid`: $\gamma_e = 0.60$
    - `uniform_slow`: $\gamma_e = 0.92$
    - `hetero_linear`: $\gamma_{e,j} = 0.25 + j \cdot \frac{0.67}{H-1}$
    - `hetero_log`: $\gamma_{e,j} = 1.0 - 10^{-0.6 - j \cdot \frac{1.2}{H-1}}$
    - `hetero_bimodal`: half fast ($\gamma=0.30$), half slow ($\gamma=0.90$)
  - Factor 2 (Sensory Event Gating):
    - `none`: $G(t) = 1.0$
    - `hard_sensory`: $G(t) = \mathbb{I}(\max_k |x_k(t)| > \theta_s)$
    - `soft_sigmoid`: $G(t) = \sigma(10 (\max_k |x_k(t)| - \theta_s))$
    - `recurrent_only`: Gating applied only to recurrent eligibility $z_{e,\text{rec}}$, leaving feedforward traces ungated.
- **Scientific Goal:** Disentangle whether performance gains originate from heterogeneous timescales, sensory gating, or synergistic interaction.

### 2.3 Group H3: Distractor Robustness & Working Memory Preservation (96 Configs)
- **Core Investigation:** Evaluate whether a single-compartment ALIF soma can preserve working memory across high-noise distractors without transitioning to multi-compartment dendritic architectures.
- **Mechanisms:**
  - *Dynamic Sensory Attenuation:* $\alpha_{\text{in}}(t) = 1.0 / (1.0 + \eta_{\text{atten}} \cdot \|x(t)\|_{\text{distractor\_channel}})$.
  - *Noise-Sensitive Credit Modulation:* $\Delta W \propto \frac{\text{Trace}}{\text{Trace} + \sigma_n^2}$.
  - *Plasticity Energy Thresholding:* Freeze recurrent synaptic updates when input activity is non-salient.
  - *Causal Invariant:* Compare pre-spike vs post-spike voltage and recurrent representation preservation under distractor amplitudes up to $\pm 2.0$.

### 2.4 Group H4: Readout & Associative Memory Scalability (80 Configs)
- **Tasks:** Scaled key-value associative retrieval across $N \in \{2, 4, 8, 16\}$ keys.
- **Mechanisms:**
  - Linear Readout (Baseline).
  - Context-Conditioned Readout (CKCER).
  - Bilinear / Feature-Gated Readout.
- **Failure Taxonomy:**
  - `STORAGE_FAILURE`: Frozen hidden representation cannot be decoded by ridge probe ($\text{probe} < 65\%$).
  - `RETRIEVAL_FAILURE`: Probe decodes representation ($\text{probe} \ge 80\%$), but direct readout fails ($\text{direct} < 65\%$).
  - `OPTIMIZATION_FAILURE`: Loss fails to converge or gradients vanish/explode.
  - `BENCHMARK_FAILURE`: Data leakage or degenerate task encoding.

### 2.5 Group H5: Task-Agnostic Continual Learning & Interference (80 Configs)
- **Benchmark:** Sequential learning stream without task IDs, without task boundary signals, and without unconstrained replay buffers.
- **Mechanisms:**
  - Benna-Fusi (BF-Cascade) 2-compartment metaplastic diffusion:
    $\dot{u}_1 = -\kappa_1 (u_1 - u_2) + \Delta W$, $\dot{u}_2 = \kappa_2 (u_1 - u_2)$.
    Grid: $\kappa_1 \in [0.001, 0.02]$, $\kappa_2 \in [0.0002, 0.005]$.
  - BF-Cascade 3-compartment extension ($u_1, u_2, u_3$).
  - Orthogonal weight projection.
  - Matched-budget replay baseline (FLOPs and updates strictly matched).

---

## 3. Strict Experimental Invariants & Architectural Constraints

1. **Network Core:**
   - Single-compartment recurrent ALIF (`SpikingCore`).
   - Hidden units: $H = 48$.
   - Recurrent fan-in: $K = 8$.
   - Input channels: $D = 8$ strictly (no extra task indicator channels).
   - Hard binary spikes: $z_j(t) \in \{0.0, 1.0\}$.
   - Purely local plasticity: No global BPTT, no backward backprop.
2. **Resource Constraints:**
   - Maximum 2 concurrent CPU worker processes.
   - BLAS thread count pinned: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.
   - Maximum memory per process $\le 2.0$ GB.
   - Total campaign RAM $\le 6.0$ GB.
3. **No-Shortcut Rules:**
   - Shuffled control test must shuffle ONLY the credit signal across neurons while leaving architecture, decays, gating, and optimizer untouched.
   - Causal weight revert must reset learned hidden weights to initialization while evaluating with learned readout.
   - **Crucial Rule:** Absolutely NO `direct_accuracy >= 95%` bypass of the causal revert test! If a task hits ceiling, it must be evaluated on an extended delay or higher distractor amplitude to verify true causal reliance.

---

## 4. Multi-Stage Search Architecture

```
[Stage A: Candidate Generation] -> 512 Canonical Configs (Deterministic Seeds)
           |
[Stage B: Fast DEV Screening]   -> 384 eps/task, 2 DEV Seeds (140011, 140017)
           |
[Stage C: Adaptive Search]      -> Refined parameter neighborhoods around top performers
           |
[Stage D: Causal Ablations]     -> Top 16 Candidates: 10 Factorial Ablation Conditions
           |
[Stage E: Robustness Testing]   -> Top 5 Candidates: Distractor x2.0, Long Delay, Reverse Stream
           |
[Stage F: Locked Confirmation]  -> Top <=2 Candidates: 10 Unseen Seeds, Frozen Code & Protocol
           |
[Stage G: Red-Team Falsification & Statistical Audit]
```

### 4.1 Pre-Registered Seeds
- **DEV Seeds:** `[140011, 140017, 140023]`
- **Robustness Seeds:** `[160011, 160017, 160023]`
- **Locked Confirmation Seeds (10 Fresh Seeds):**
  `[191001, 191007, 191013, 191019, 191025, 191031, 191037, 191043, 191049, 191055]`

### 4.2 Gate Criteria & PASS/FAIL Thresholds
- **Acquisition Accuracy:** $\ge 65.0\%$ across all tasks.
- **Final Accuracy:** $\ge 65.0\%$.
- **Mean Forgetting:** $\le 3.0$ percentage points ($0.030$).
- **Worst Forgetting:** $\le 5.0$ percentage points ($0.050$).
- **Representation Gain over Frozen:** $\ge +5.0$ percentage points with 95% CI lower bound $> 0.0$.
- **Causal Revert Drop:** $\ge +5.0$ percentage points (no automatic bypass for high accuracy).
- **False Discovery Rate (FDR):** Benjamini-Hochberg corrected $q \le 0.05$.

---

## 5. Pre-Registration Verification
- This protocol is committed to version control before running the candidate search.
- All subsequent analysis and confirmation results will be evaluated strictly against these pre-registered specifications.
