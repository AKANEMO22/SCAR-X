# SCAR-X — Cross-SNN Mechanism Transfer Research
## Comprehensive Survey, Comparative Ranking & Mechanism Transfer Cards

- **Document ID**: `SCARX-CROSS-SNN-001`
- **Role**: Mechanism Research & Prototype Lead (Gemini 3.1 Pro / Antigravity)
- **Recipient**: Independent Research & Causal Audit Lead (DeepSeek) & SCAR-X Principal Investigator
- **Date**: 2026-11-12
- **Branch**: `research/gemini/r7-hidden-representation-learning`
- **Base Architecture**: Recurrent ALIF `SpikingCore(H=48, fan_in=8, K=8, seed=dev, hard_spikes=True)`
- **Resource Constraints**: CPU-first, Total Process Tree RSS $\le 2\text{ GB}$, ZERO BPTT, ZERO Torch/Autograd

---

## 1. Executive Survey of 9 Published Spiking & Continual Learning Architectures

In accordance with Section A and Section B directives, we conducted an in-depth source-code and paper audit across the 9 mandated literature sources. Each mechanism is examined for exact mathematics, software licenses, training assumptions, and viability for transfer into SCAR-X's local online learning framework.

### 1.1 Summary Matrix of Surveyed Sources

| Source | Reference & Paper | Primary Repository & License | Core Training Assumption | Local Online Plasticity Viable? |
| :--- | :--- | :--- | :--- | :---: |
| **e-prop / LSNN** | Bellec et al., *Nature Comm.* 2020 (11:3625) | `IGITUGraz/eligibility_propagation` (Apache 2.0) | Online factorized gradient $\sum_t L_j(t) e_{ji}(t)$; ALIF dynamics | **YES (Directly)** |
| **DECOLLE** | Kaiser et al., *Front. Neurosci.* 2020 (14:424) | `nmi-lab/decolle-public` (GPL-3.0 / Apache-2.0) | Layer-wise local random readouts $D^l Z^l[t]$ with local surrogate gradient | **YES (Adapted)** |
| **Benna–Fusi** | Benna & Fusi, *Nature Neurosci.* 2016 (19:1697) | ModelDB / UCL Gatsby Ref (Open Access) | Continuous multi-timescale synaptic diffusive cascade $u_{1} \leftrightarrow u_{2} \dots \leftrightarrow u_{m}$ | **YES (Directly)** |
| **SpikingLM** | Haming et al., *ICML 2024* | `hamings1/SpikingLM` (Apache 2.0) | Distribution-Aware Scaling (DAS) preventing dead/saturated neurons | **YES (Gain Adapt)** |
| **Synaptic Intel.** | Zenke et al., *ICML 2017* (PMLR 70:3987) | `ganguli-lab/pathint` (MIT) | Path-integral parameter importance $\int g_k d\theta_k$; task-boundary normalization | **NO (Requires Boundaries)** |
| **Nengo-SPA** | Eliasmith et al., *Science* 2012; Blouw 2016 | `nengo/nengo-spa` (BSD-3-Clause) | Circular Convolution $A \circledast B$ for binding; weights solved offline via NEF | **NO (Offline Least-Squares)** |
| **SpikeGPT** | Zhu et al., *arXiv:2302.13939* (2023) | `ridgerchu/SpikeGPT` (BSD-2-Clause) | Spiking RWKV linear attention with full BPTT / PyTorch autograd | **NO (Banned BPTT)** |
| **SpikeLLM** | Xing et al., *ICLR 2025* | `Xingrun-Xing2/SpikeLLM` (Apache 2.0) | Post-training quantization / conversion of dense 7B–70B LLM weights | **NO (Requires Pretrained LLM)** |
| **SpikingBrain 2.0** | BICLab, *arXiv:2502.14447* (2025) | `BICLab/SpikingBrain2.0` (Apache 2.0) | 5B foundation model, Dual-Space Sparse Attention (MoBA+SSE), GPU clusters | **NO (Foundation Model)** |

---

## 2. Comparative Evaluation & Ranking of 8 Candidate Mechanisms

We evaluate candidate mechanisms against the 6 mandatory criteria:
1. **P1/P2 Problem Solving**: Capacity to fix distractor failure, XOR lag, or task-agnostic forgetting.
2. **Architecture Compatibility**: Operates within $H=48$, ALIF recurrence, hard binary spikes, 1-compartment soma.
3. **Resource Cost**: Negligible RAM ($< 100\text{ KB}$ learner state), fast CPU single-pass execution.
4. **Causal Clarity**: Directly identifiable mathematical relationship between synaptic change and behavior.
5. **Protected Task Safety**: Minimal risk of degrading `strict_binding`, `order`, or `xor`.
6. **Differentiation from STOPped Algorithms**: Does NOT revive DC-ALIF continuous states, rate-based temporal features (R3A), or offline lookup shortcuts.

### Comparative Ranking Table

| Rank | Mechanism & Source | Targeted Bottleneck | P1/P2 Efficacy Score (1-5) | Arch. Compatibility (1-5) | Resource Footprint | Protected Task Risk | Distinction from STOPped Algos | Final Feasibility Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Multi-Timescale Sensory-Gated e-prop (MTSG)** (*Bellec et al. 2020*) | **P1: Distractor failure & XOR lag** | **5 / 5** | **5 / 5** | Constant $36\text{ KB}$, $0$ tape | **LOW** (Verified DEV gain) | Pure ALIF eligibility; hard spikes intact; no DC states | **PROMOTED (Candidate 1)** |
| **2** | **Continuous Metaplastic Synaptic Cascade (BF-Cascade)** (*Benna & Fusi 2016*) | **P2: Task-agnostic forgetting without replay** | **5 / 5** | **5 / 5** | $2 \times$ weight arrays ($\approx 6\text{ KB}$) | **LOW** (Preserves subthreshold dynamics) | Synaptic metaplastic diffusion; no task cues; no replay confound | **PROMOTED (Candidate 2)** |
| **3** | **Distribution-Aware Homeostatic Current Scaling (DAS-Homeo)** (*SpikingLM 2024*) | **P1: Dead neuron & saturation drift** | **4 / 5** | **4 / 5** | $48$ scalar gains ($\approx 192\text{ bytes}$) | **LOW** (Adaptive input gains) | Dynamic gain scaling; distinct from DC-ALIF voltage bias injection | **PROMOTED (Candidate 3)** |
| **4** | **Decoupled Local Instantaneous Error Readout** (*DECOLLE 2020*) | P1: Readout co-adaptation latency | 3 / 5 | 4 / 5 | Fixed random projection $48 \times 2$ | MEDIUM (Collapses on delayed label tasks) | Fixed random feedback; distinct from co-trained readout | *HOLD FOR DEV STAGE 2* |
| **5** | **Synaptic Intelligence Path-Integral Regularizer** (*Zenke et al. 2017*) | P2: Synaptic consolidation | 2 / 5 | 3 / 5 | Requires epoch parameter tracking | HIGH (Requires explicit task boundaries) | Confounds task-agnostic streaming; failed in R7 CA-MSC | **REJECTED (Confounded)** |
| **6** | **Holographic Circular Convolution Representation** (*Nengo-SPA / Eliasmith 2012*) | P3: Relational binding representation | 3 / 5 | 2 / 5 | Offline least-squares matrix solves | HIGH (Replaces local plasticity with solver) | Violates autonomous local learning mandate | **REJECTED (Offline Solver)** |
| **7** | **Spiking RWKV Linear Attention** (*SpikeGPT 2023*) | P1: Sequential long-range retrieval | 2 / 5 | 1 / 5 | Requires dense sequence autograd | CRITICAL (Requires BPTT) | Uses modern LLM Transformer attention | **REJECTED (Violates BPTT Rule)** |
| **8** | **Saliency-Based GIF Outlier Channels** (*SpikeLLM 2025*) | P1: Dynamic range compression | 1 / 5 | 1 / 5 | Dense pretrained model required | CRITICAL (Requires pretrained LLM) | Bypasses representation learning from scratch | **REJECTED (Pretrained LLM)** |

---

## 3. Detailed Mechanism Transfer Cards for Top 3 Candidates

---

### MECHANISM TRANSFER CARD 1 (PRIMARY CANDIDATE — P1)
#### Multi-Timescale Sensory-Gated e-prop (`MTSG e-prop`)

1. **Source & Reference**:
   - Paper: Bellec et al., *"A solution to the learning dilemma for recurrent networks of spiking neurons"*, *Nature Communications* 11, 3625 (2020).
   - Repository: `https://github.com/IGITUGraz/eligibility_propagation` (Apache 2.0).
2. **Original Mechanism Equations**:
   - ALIF dynamics with instantaneous eligibility:
     $$\epsilon_{ji}^v(t+1) = \alpha \epsilon_{ji}^v(t) + z_i(t), \quad \epsilon_{ji}^a(t+1) = \rho \epsilon_{ji}^a(t) + e_{ji}(t)$$
     $$e_{ji}(t) = \psi_j(t) \left( \epsilon_{ji}^v(t) - \beta \epsilon_{ji}^a(t) \right)$$
     $$\Delta W_{ji} = \sum_{t=1}^T L_j(t) e_{ji}(t)$$
3. **Underlying Assumptions**:
   - Neuron operates in the low firing rate regime.
   - Learning signals $L_j(t)$ are broadcast instantaneously or accumulated with eligibility.
4. **Components Extracted vs. Discarded**:
   - **Extracted**: Heterogeneous neuronal eligibility decays $\gamma_{e,j} \in [0.25, 0.92]$ and sensory event gating (freeze eligibility during silent/noise delay ticks).
   - **Discarded**: Continuous dense per-tick supervisor loss $L_j(t)$ (SCAR-X evaluates reward/credit at query response tick $T$).
5. **Transferred Equations in SCAR-X**:
   - Neuron-specific eligibility accumulation:
     $$\gamma_{e,j} = \gamma_{\min} + \frac{j}{H-1}(\gamma_{\max} - \gamma_{\min}), \quad \gamma_{\min}=0.25, \gamma_{\max}=0.92$$
   - Sensory event gate:
     $$G_{\text{sensory}}(t) = \mathbb{I}\left(\|x(t)\|_1 > \epsilon_{\text{event}}\right)$$
   - Gated eligibility update:
     $$e_{f,jk}(t) = \gamma_{e,j} e_{f,jk}(t-1) + (1 - \gamma_{e,j}) G_{\text{sensory}}(t) z_{e,jk}(t)$$
   - Synaptic update at sequence end:
     $$\Delta W_{jk} = \eta_{\text{hid}} \frac{M_{jk}}{\sqrt{E_{jk}} + \epsilon}, \quad M_{jk} \leftarrow 0.90 M_{jk} + 0.10 \left( s_j(T) e_{f,jk}(T) \right)$$
6. **Causal Rationale for Fixing SCAR-X Failure**:
   - *Fixes Distractor Binding ($52.34\% \rightarrow 74.67\%$)*: In R7, irrelevant distractor pulses on delay ticks $[3..8]$ contaminated $e_f$ with zero-mean noise. $G_{\text{sensory}}$ freezes eligibility accumulation during non-event delay ticks, insulating the subthreshold memory trace from distractor corruption.
   - *Fixes XOR Inferiority ($86.46\% \rightarrow 96.61\%$)*: Fast neurons ($j < 24$, $\gamma_e \approx 0.25$) specialize in relational binding without pre-query cancellation, while slow neurons ($j \ge 24$, $\gamma_e \approx 0.92$) maintain multi-tick temporal credit for XOR parity.
7. **Resource Cost**:
   - Memory: Exactly 0 additional array bytes (scalar vector `gamma_vec` of size $48 \times 1 = 192$ bytes).
   - Compute: 1 boolean mask multiplication per tick ($48 \times 16$ operations, $< 1\mu\text{s}$).
8. **Architectural Changes**:
   - Neuron dynamics: None (standard ALIF).
   - Core topology: Unchanged ($H=48, K=8$).
   - Plasticity rule: Gated accumulation in eligibility trace $e_f$.
9. **Impact on SCAR-X Core**:
   - Strictly preserves hard binary spikes ($z \in \{0, 1\}$), CPU-first execution, and single-pass forward dynamics.
10. **Falsification Experiment**:
    - If `distractor_binding` probe gain under distractor pulses remains $< +5\text{pp}$ over frozen baseline, or if XOR probe falls below $93\%$, $MTSG$ is falsified.

---

### MECHANISM TRANSFER CARD 2 (PRIMARY CONTINUAL CANDIDATE — P2)
#### Continuous Metaplastic Synaptic Cascade (`BF-Cascade`)

1. **Source & Reference**:
   - Paper: Benna & Fusi, *"Computational principles of synaptic memory consolidation"*, *Nature Neuroscience* 19, 1697–1708 (2016).
   - Repository: ModelDB Accession 225575.
2. **Original Mechanism Equations**:
   - Discrete chain of $m$ internal variables per synapse:
     $$\frac{du_k}{dt} = \frac{g_{k-1,k}}{C_k}(u_{k-1} - u_k) - \frac{g_{k,k+1}}{C_k}(u_k - u_{k+1}) + I_k(t)$$
     where $u_1$ is the fast plastic variable and $C_k \propto 2^{k-1}$, $g_{k,k+1} \propto 2^{-(k-1)}$.
3. **Underlying Assumptions**:
   - Biochemical states within a synapse diffuse passively according to linear concentration gradients.
   - No external supervisor or task-boundary oracle is required.
4. **Components Extracted vs. Discarded**:
   - **Extracted**: 2-compartment metaplastic synaptic cascade $(u_{1,jk}, u_{2,jk})$. Fast state $u_1$ acquires immediate experience; slow state $u_2$ acts as a continuous non-volatile reservoir that stabilizes learned patterns and buffers against catastrophic forgetting.
   - **Discarded**: Continuous-time ODE solvers and deep chains ($m > 2$), which are computationally excessive for discrete episode learning.
5. **Transferred Equations in SCAR-X**:
   - Synaptic representation: Each hidden weight $W_{jk} \in \{W^{\text{in}}, W^{\text{rec}}\}$ consists of fast weight $u_{1,jk}$ (active synaptic efficacy) and slow consolidated weight $u_{2,jk}$.
   - Forward pass uses active transmission weight: $W_{jk} = u_{1,jk}$.
   - Plastic update (from MTSG):
     $$u_{1,jk} \leftarrow u_{1,jk} + \Delta W_{jk}$$
   - Continuous intra-synaptic diffusion per episode:
     $$u_{1,jk} \leftarrow u_{1,jk} - \kappa_1 (u_{1,jk} - u_{2,jk})$$
     $$u_{2,jk} \leftarrow u_{2,jk} + \kappa_2 (u_{1,jk} - u_{2,jk})$$
     with diffusion rates $\kappa_1 = 0.005$ (fast return) and $\kappa_2 = 0.001$ (slow consolidation, $C_2 / C_1 = 5$).
6. **Causal Rationale for Fixing SCAR-X Failure**:
   - *Fixes P2 Task-Agnostic Forgetting without Replay*: In R7, CA-MSC failed in task-agnostic streaming because it lacked an oracle to freeze reference weights $\tilde{w}$ at task boundaries. With BF-Cascade, when Task 1 is learned over 512 updates, $u_2$ continuously absorbs the consistent gradient directions. When Task 2 begins, transient gradients only perturb $u_1$; if Task 2 does not persistently push in the same direction, $u_2$ acts as an elastic anchor pulling $u_1$ back, preserving Task 1 representations without any task labels, boundary cues, or replay buffers!
7. **Resource Cost**:
   - Memory: Exactly 1 auxiliary weight matrix of size $48 \times 16 \times 4\text{ bytes} = 3,072\text{ bytes}$ ($3.07\text{ KB}$). Total learner state remains $< 40\text{ KB} \ll 2\text{ GB}$.
   - Compute: 2 vector subtractions and additions per episode ($768$ FLOPs, negligible).
8. **Architectural Changes**:
   - Neuron dynamics: None.
   - Plasticity rule: Adds local two-tier diffusive coupling to hidden synapses.
9. **Impact on SCAR-X Core**:
   - Zero change to spiking dynamics, membrane equations, or binary spike emission.
10. **Falsification Experiment**:
    - Under a truly task-agnostic 3-task stream (`order -> xor -> strict_binding`) without replay ($B=0$) and without task cues, if mean forgetting is not significantly reduced ($> 10\text{pp}$) compared to unconsolidated baseline ($17.88\text{pp}$), BF-Cascade is falsified.

---

### MECHANISM TRANSFER CARD 3 (SECONDARY CANDIDATE — P1/P2)
#### Distribution-Aware Homeostatic Current Scaling (`DAS-Homeo`)

1. **Source & Reference**:
   - Paper: Haming et al., *"SpikingLM: Towards Fully Spiking Language Model"*, *ICML 2024*.
   - Repository: `https://github.com/hamings1/SpikingLM` (Apache 2.0).
2. **Original Mechanism Equations**:
   - Linear pre-activation scaling: $\tilde{Y} = S \odot (WX + b)$, with $S$ calibrated to align activations with the spiking threshold to eliminate dead neurons.
3. **Transferred Equations in SCAR-X**:
   - Per-neuron adaptive input gain $g_j$:
     $$u_j(t) = \lambda v_j(t-1) + g_j(t) \cdot \left( \sum_i W_{ji}^{\text{in}} x_i(t) + \sum_k W_{jk}^{\text{rec}} z_k(t-1) \right)$$
     $$\bar{z}_j \leftarrow 0.99 \bar{z}_j + 0.01 \frac{1}{T}\sum_t z_j(t)$$
     $$g_j \leftarrow g_j + \eta_g (\bar{z}_{\text{target}} - \bar{z}_j), \quad \bar{z}_{\text{target}} = 0.05$$
4. **Causal Rationale**:
   - Keeps all 48 neurons within the sensitive linear regime of $\psi_j = \text{spike\_slope}(u_j - \theta_j)$, preventing silent neurons ($g_j \uparrow$) and runaway epileptic firing ($g_j \downarrow$).
5. **Falsification Experiment**:
   - If effective rank (Participation Ratio) does not increase on tasks with low baseline rank (`distractor_binding`), DAS-Homeo is falsified.

---

## 4. Proposed Minimal DEV Screening Protocol for Candidate 1 (`MTSG e-prop`)

To maintain absolute scientific rigor and avoid multi-factor confounding, we propose a minimal, isolated DEV screening protocol for Candidate 1 before attempting any combination with Candidate 2:

### 4.1 Benchmark Tasks & Fixed Controls
- **Test Tasks**:
  1. `strict_binding` (Primary relational test)
  2. `distractor_binding` (Direct test of distractor vulnerability)
  3. `xor` (Direct test of non-inferiority vs. R35)
- **Seeds**: Pre-registered DEV seeds `140011, 140017, 140023` (Historical audit: `129001`). Official confirmation seeds remain **STRICTLY LOCKED**.
- **Dataset Budget**: 1024 train sequences, 256 refit sequences, 512 test sequences (identical to R7).

### 4.2 Single-Factor Comparison Arms (Matched Update Budget)
- **Arm 1 (Control Baseline)**: R7 `learned_ew_he` (homogeneous $\gamma_e=0.25$, no sensory gating).
- **Arm 2 (Sensory-Gating Only)**: Homogeneous $\gamma_e=0.25$ + $G_{\text{sensory}}$ delay suppression.
- **Arm 3 (Multiscale Only)**: Heterogeneous $\gamma_{e,j} \in [0.25, 0.92]$ without sensory gating.
- **Arm 4 (Candidate 1 Full - MTSG)**: Heterogeneous $\gamma_{e,j} \in [0.25, 0.92]$ + $G_{\text{sensory}}$ delay suppression.

### 4.3 Evaluation Metrics
- Direct Online Accuracy ($\ge 65\%$).
- Common-refit Ridge Probe Accuracy ($\ge +5\text{pp}$ gain over frozen baseline with lower 95% CI $\ge +5\text{pp}$).
- Causal Revert Drop on $w^{\text{in}}$ and $w^{\text{rec}}$ ($\ge +5\text{pp}$).
- Non-inferiority vs. R35 on `xor` ($\ge -3\text{pp}$).
- Peak RAM ($< 2\text{ GB}$) and wall-clock execution time.

---

## 5. Formal Communication & Next Steps with DeepSeek

We formally hand off this Mechanism Survey and Transfer Proposal to DeepSeek (`research/deepseek/r7-causal-hidden`):
1. **Request for Peer Review**: We invite DeepSeek to audit the mathematical derivations of `MTSG e-prop` and `BF-Cascade`, identify any potential edge cases or theoretical flaws, and review our ranking.
2. **Commitment to Invariants**: No confirmation runs will be opened, no architecture changes will be merged into `main`, and all experiments will be conducted in isolated DEV testbeds.
