# SCAR-X â€” Round 8 Multi-Agent Research Protocol
## Multi-Timescale Sensory-Gated e-prop (MTSG) & Continuous Metaplastic Cascade (BF-Cascade)

- **Protocol ID**: `SCARX-R8-PROTOCOL-001`
- **Lead Researcher**: Gemini â€” Principal Neural Architecture Researcher
- **Date**: 2026-11-12
- **Branch**: `research/gemini/r8-causal-learning-retention-20261112`
- **Base Commit**: `631c89041ec7ed3ce5d6b08373b68e9dc2e4a122`
- **Base Architecture**: Unchanged Recurrent ALIF `SpikingCore(H=48, fan_in=8, inputs=8, outputs=2)`

---

## 1. Multi-Agent Roles & Execution Structure

In compliance with Section 3, since the host environment lacks autonomous daemon-spawning primitives (`SUBAGENT_UNAVAILABLE`), the five mandatory research roles are executed sequentially through strict workspace/module isolation:

1. **SUB-AGENT A (Audit & Benchmark Integrity)**:
   - Audits prior code, detects leakages, establishes Benchmark Integrity Assertions (BIA).
2. **SUB-AGENT B (Hidden Representation & Credit Assignment)**:
   - Implements Candidate 1 (`MTSG e-prop`), verifies mathematical sensitivity, executes P1 development screen.
3. **SUB-AGENT C (Readout, Associative Memory & Retention)**:
   - Implements Candidate 2 (`BF-Cascade`), scales multi-key associative memory (2 -> 4 -> 8 keys), evaluates task-agnostic continual stream.
4. **SUB-AGENT D (Runtime & Resource Efficiency)**:
   - Profiles memory (< 2 GB), latency, CPU FLOPs, and inactive neuron computation.
5. **SUB-AGENT E (Independent Red-Team Reviewer)**:
   - Executes adversarial stress tests, verifies falsification conditions, provides audit verdict.

---

## 2. Formal Hypotheses (Section 7 Mandate)

### Hypothesis H1 â€” Multi-Timescale Sensory-Gated e-prop (MTSG e-prop) for P1
- **H1 (Problem)**: EW-HE collapsed to $52.34\%$ on `distractor_binding` due to delay noise bursts corrupting eligibility traces, and lagged R35 on `xor` ($86.46\%$ vs $96.48\%$) due to uniform fast eligibility decay $\gamma_e = 0.25$.
- **H2 (Causal Mechanism)**:
  1. Fast eligibility decays cannot bridge the 11-step temporal gap of XOR parity, whereas slow eligibility ($\gamma_e=0.92$) enables long temporal credit but blurs fast relational binding. A heterogeneous spectrum $\gamma_{e,j} \in [0.25, 0.92]$ provides both fast and slow temporal integrators.
  2. Noise bursts during non-stimulus delay periods inject zero-mean spurious gradients into synaptic traces. Sensory event gating $G_{\text{sensory}}(t) = \mathbb{I}(\|x(t)\|_1 > \epsilon_{\text{event}})$ freezes eligibility accumulation during silence/distractor delay ticks.
- **H3 (Intervention)**:
  - Heterogeneous decays: $\gamma_{e,j} = 0.25 + \frac{j}{47}(0.92 - 0.25)$.
  - Gated eligibility: $e_{f,jk}(t) = \gamma_{e,j} e_{f,jk}(t-1) + (1 - \gamma_{e,j}) G_{\text{sensory}}(t) z_{e,jk}(t)$.
- **H4 (Prediction)**:
  - `distractor_binding` direct accuracy $\ge 70\%$ and ridge probe $\ge 72\%$ (gain $\ge +15\text{pp}$ over EW-HE).
  - `xor` probe $\ge 94\%$ (eliminating the lag vs R35).
  - `strict_binding` and `order` maintained at $\ge 75\%$ direct accuracy.
- **H5 (Falsifier)**:
  - If `distractor_binding` probe remains $< 65\%$, or if `xor` probe remains $< 90\%$, H1 is FALSIFIED.
- **H6 (Controls)**:
  - Frozen-hidden (no update), Zero-signal (signal=0), Shuffled-signal (permuted error), Readout-only (w_in, w_rec fixed), Homogeneous $\gamma_e=0.25$ without gating.
- **H7 (Cost)**:
  - Exactly 0 additional array parameters ($48 \times 1$ scalar decay vector). Compute: 1 boolean mask check per tick.
- **H8 (Scope)**:
  - Proves hidden representation credit assignment under temporal delay and noise without modifying spiking core dynamics.

### Hypothesis H2 â€” Continuous Metaplastic Cascade (BF-Cascade) for P2
- **H1 (Problem)**: Continual learning without replay buffer suffers catastrophic forgetting ($17.88\%$ mean forgetting). Previous CA-MSC required oracle task-boundary snapshots and channel leakage ($D=11$).
- **H2 (Causal Mechanism)**:
  - Intra-synaptic diffusive cascade (Benna & Fusi, *Nature Neurosci.* 2016): each synapse maintains an active efficacy $u_1$ and a slow reservoir $u_2$. Continuous passive diffusion $u_1 \leftrightarrow u_2$ buffers transient learning interference without any external task boundary cue.
- **H3 (Intervention)**:
  - State: $u_{1,jk} = W_{jk}$, $u_{2,jk} = W_{jk}^{\text{init}}$.
  - Update: $u_1 \leftarrow u_1 + \Delta W$.
  - Continuous diffusion per episode:
    $u_1 \leftarrow u_1 - \kappa_1 (u_1 - u_2)$
    $u_2 \leftarrow u_2 + \kappa_2 (u_1 - u_2)$ with $\kappa_1 = 0.005, \kappa_2 = 0.001$.
- **H4 (Prediction)**:
  - In a truly task-agnostic 3-skill stream ($D=8$, zero skill cues, zero replay buffer), mean forgetting is significantly reduced compared to the unconsolidated baseline ($17.88\%$).
- **H5 (Falsifier)**:
  - If mean forgetting is not reduced by at least $5\text{pp}$ vs unconsolidated baseline, H2 is FALSIFIED.
- **H6 (Controls)**:
  - Unconsolidated baseline ($\kappa_1 = \kappa_2 = 0$), Frozen weights, Random diffusion control.
- **H7 (Cost)**:
  - Exactly 1 auxiliary matrix per weight tensor ($\approx 3\text{ KB}$ for $W^{\text{in}}, W^{\text{rec}}$). Negligible CPU FLOPs.
- **H8 (Scope)**:
  - Proves continual retention in streaming without oracle task boundary supervision.

---

## 3. Pre-Registered Experimental Screen & Gates

### 3.1 Pre-Registered Seeds
- **DEV Seeds**: `140011, 140017, 140023`
- **Locked Confirmation Seeds**: `129001, 129007, 129013, 129019, 129025`

### 3.2 Evaluation Criteria & Gates
- **P1 Direct Accuracy Gate**: $\ge 65\%$ on target tasks.
- **P1 Probe Gain Gate**: $\ge +5\text{pp}$ over frozen baseline with lower 95% CI $> 0$.
- **P1 Causal Revert Drop Gate**: $\ge +5\text{pp}$ upon reverting hidden weights $(w^{\text{in}}, w^{\text{rec}})$.
- **P1 Non-Inferiority Gate**: Within $-3\text{pp}$ of R35 on all reference tasks.
- **P2 Continual Stream Gate**: Acquisition $\ge 65\%$, Final $\ge 65\%$, Significant reduction in mean forgetting.
- **Resource Budget Gate**: Peak RSS $< 2,000,000,000\text{ bytes}$ ($2\text{ GB}$).
