# SCAR-X Round 9b: Pre-Registered Research Protocol
## Causal Learning Verification, Factorial MTSG & Independent Mechanism Research

**Date:** 2026-11-12  
**Lead Agent:** Gemini (Principal SNN Research Scientist & Multi-Agent Research Coordinator)  
**Base Commit SHA:** `607ffca7c9222c5e3fd9741c3ebdf5939e10bedf`  
**Research Branch:** `research/gemini/r9b-p1-matched-control-20261112`  
**Main Branch SHA:** `e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541` (UNTOUCHED)  
**Codex P0 Handoff Status:** Accessible read-only at `scar-x/local/worktrees/scarx-r8-matched-controls-20261112/` (76/76 snapshot tests pass, read-only custody preserved, not assumed pushed to remote).  
**Multi-Agent Mode:** `SUBAGENT_UNAVAILABLE` (Tooling does not support live subagent spawning; sequential audit execution conducted across Agent A–E perspectives).

---

## 1. Executive Summary & Research Mandate

The objective of Round 9b is to determine why the 1-compartment recurrent ALIF network (H=48, fan-in=8) in SCAR-X learns specific relational tasks (e.g., binding and XOR) but suffers from high distractor vulnerability and catastrophic forgetting when exposed to continual task streams.

We enforce:
1. **P1 — Factorial Matched MTSG Exploration:** An analytical $4 \times 3 \times 2 = 24$ factorial design examining eligibility timescales, sensory gating modes, and derivative-through-reset treatments.
2. **P1 — Strict Matched Controls:** Shuffled, zero-signal, frozen-hidden, and readout-only controls that share 100% identical trace dynamics, surrogate slopes, clipping, and optimizer settings, differing ONLY in the permutation of learning signals.
3. **P2 — Representation vs Direct Learning Disentanglement:** Explicitly separating direct neural accuracy, frozen-hidden probe, learned-hidden probe, and causal revert drops.
4. **P3 — Continual Learning & Consolidation Audit:** Evaluating BF-Cascade across 5 arms under strict continual learning (ZERO future-task calibration), verifying both mean forgetting ($\le 3$ pp) and worst-case forgetting ($\le 5$ pp) on a per-seed basis.
5. **P4 — Distractor Loss Attribution:** Tracing signal destruction across 6 neurocomputational stages (input encoding, subthreshold voltage, recurrent state, eligibility traces, hidden representation, readout) to test whether soma reset is the true bottleneck before positing multi-compartment dendrites.

---

## 2. Multi-Agent Sequential Audit Structure

Given `SUBAGENT_UNAVAILABLE`, execution is conducted sequentially from 5 forensic stances:
- **Agent A (Benchmark Forensic Auditor):** Audits dataset generation, guarantees 8 input channels (no channel 6 snooping), ensures no DEV/confirmation leakage, verifies no task-boundary cues, and checks codebook injectivity.
- **Agent B (Temporal Credit Mechanism Researcher):** Verifies the mathematics of subtractive vs stopped-reset Jacobians, trace filtering equations, and per-neuron credit assignment.
- **Agent C (Representation Researcher):** Measures learned representation quality via linear probe vs direct readout, distinguishing genuine hidden plasticity from probe artifacts.
- **Agent D (Continual Learning Researcher):** Audits BF-Cascade retention, testing whether consolidation preserves memories or merely suppresses learning capacity.
- **Agent E (Independent Red-Team):** Actively attempts to falsify positive claims by checking whether controls match candidate scores under low-budget or adversarial conditions.

---

## 3. Factorial MTSG Design Space (24 Configurations)

All configurations share:
- ALIF neuron model: single-compartment, leaky integrate-and-fire with adaptive threshold $\theta = v_{th} + \beta a$.
- Core topology: $H=48$, $D_{in}=8$, $K_{fan}=8$, $O=2$.
- Initial weights: identical paired seeds.
- Optimizer: sign-momentum RMS energy normalizer, learning rate $\eta=0.008$, clip $c=0.010$.
- Readout: linear softmax with pre-update credit projection $\delta_j = (W_{out}^T \text{err})_j \cdot \mathbb{I}(|\text{raw}_j| < 5) / \text{scale}_j$.

### Factors:
- **Factor A — Eligibility Decay ($\gamma$):**
  1. `uniform_fast`: $\gamma_j = 0.25$
  2. `uniform_mid`: $\gamma_j = 0.60$
  3. `uniform_slow`: $\gamma_j = 0.92$
  4. `heterogeneous`: $\gamma_j = \text{linspace}(0.25, 0.92, 48)$

- **Factor B — Sensory Gating ($G$):**
  1. `none`: $g = 1.0$ (no gating)
  2. `recurrent_only`: $g = 1.0$ if $\max(|x|) > 0.6$ else $0.15$ (applied to recurrent synapses only: $z_e[:, D:] \cdot g$)
  3. `soft_sensory`: $g = 0.15 + \frac{0.85}{1.0 + \exp(-(\max(|x|) - 0.6)/0.1)}$ (smooth continuous sigmoidal gating on recurrent synapses)

- **Factor C — Reset Derivative ($e_v$):**
  1. `own_reset`: Baseline subtractive reset $e_v = u_e - \theta z_e - z \beta a_{old}$
  2. `stopped_reset`: Reset-aware derivative $e_v = u_e$

Total configurations: $4 \times 3 \times 2 = 24$.

---

## 4. Strict Matched Control Specification

To prevent the R8 control flaw (where `shuffled_signal` altered decay and gating), all controls are executed by the identical unified engine:
- `matched_shuffle`: Exact same $\gamma_j$, exact same gating, exact same reset mode, exact same traces. ONLY $\delta$ is permuted across the 48 hidden neurons before synaptic update.
- `matched_zero`: Exact same traces, $\delta = 0$ (no hidden weight change).
- `matched_frozen`: Hidden weights frozen at initialization.
- `readout_only`: Only $W_{out}$ and $b$ are trained.

### Audit Norms Recorded:
- Eligibility norm: $\|e_f\|_F$
- Signal norm: $\|\delta\|_2$
- Weight step norm: $\|\Delta W\|_F$

---

## 5. Pre-Registered Evaluation Gates

### P1 Representation & Learning Gates:
1. `direct_acc >= 0.65`
2. `probe_gain >= +0.05` (learned probe - frozen probe)
3. `causal_drop >= +0.05` (direct - causal revert)
4. `non_inferior_vs_fast >= -0.03`
5. `shuffled_distinct >= +0.02` (direct - matched shuffle)

### P2 Retention Gates (Strict Continual Learning):
1. `acquisition >= 0.65`
2. `final >= 0.65`
3. `mean_forgetting <= 0.03` (strictly $\le 3$ percentage points)
4. `worst_forgetting <= 0.05` (strictly $\le 5$ percentage points per seed)

---

## 6. Seed Schedule

- **DEV Seeds:** `[140011, 140017, 140023]` (with exploratory extension `[140029, 140031]`).
- **Confirmation Seeds (Locked):** `[150001, 150002, 150003, 150004, 150005, 150006, 150007, 150008, 150009, 150010]` (10 fresh seeds; only unlocked if DEV passes all gates).
