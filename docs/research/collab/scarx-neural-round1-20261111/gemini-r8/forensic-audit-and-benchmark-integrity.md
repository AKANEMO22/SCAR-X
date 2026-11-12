# SCAR-X â€” Sub-Agent A: Forensic Audit & Benchmark Integrity Report
## Round 8 Independent Scientific Audit & Integrity Assertions

- **Document ID**: `SCARX-R8-AUDIT-001`
- **Role**: Sub-Agent A â€” Scientific Auditor & Benchmark Integrity Lead
- **Date**: 2026-11-12
- **Branch**: `research/gemini/r8-causal-learning-retention-20261112`
- **Base Commit**: `631c89041ec7ed3ce5d6b08373b68e9dc2e4a122` (on `research/gemini/r7-hidden-representation-learning`)
- **Remote `origin/main` SHA**: `e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`

---

## 1. Executive Forensic Summary of Recent Research Branches

A systematic forensic audit was conducted across git history, commit diffs, raw logs, and evaluation scripts across `main`, `research/gemini/r7-hidden-representation-learning`, `research/deepseek/r7-causal-hidden`, `research/deepseek/r6d-audit`, and `research/grok/r6-causal-readout-repair`.

### Comparative Mechanism Audit Table (Section 4 Mandate)

| Mechanism | Attempted in Branch | Tasks Won | Tasks Failed | Benchmark Defect / Leakage Found? | Warrant Further Study? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **ReadoutEligibility (R35)** | `main` / Report 35 | Simple single-key binding | Multi-task stream, distractor noise | Lacked matched learning rate comparison with R7 | Reference baseline only |
| **DC-ALIF** | `research/antigravity/round1` | None (artificial accuracy) | Dynamic stability, biological sparsity | Destroyed spike sparsity via continuous DC voltage bias | **NO (STOP)** |
| **EW-HE (Gemini R7)** | `research/gemini/r7` | `strict_binding` (+16pp over R35), `order` | `distractor_binding` (52.34%), `xor` (86.46% vs 96.48% R35) | Hardcoded dev screen claims; distractor bursts erased subthreshold memory | **YES (Evolve to MTSG)** |
| **CA-MSC (Gemini R7)** | `research/gemini/r7` | Continual with Replay | Continual without Replay (18.4% mean forg) | **CRITICAL LEAKAGE**: `wrap_skill_pulse` added 3 task channels; replay doubled update budget | **NO in current form (Pivoted to BF-Cascade)** |
| **DECOLLE Random Readout (DeepSeek R7 R43)** | `research/deepseek/r7` | Proved hidden moves (+3.69pp vs zero) | Failed to beat scalar (+1.52pp, CI [-0.97, +4.01]) | Stdout transcription; only 10/15 seeds; cut epochs 120->30 | **NO (Ruled out richer per-neuron feedback)** |
| **Multi-Timescale Sensory-Gated e-prop (MTSG)** | `research/gemini/r8` (Proposed) | Target: Distractor + XOR parity | TBD | Zero channel leakage; exact $D=8, H=48$ | **YES (Primary Candidate P1)** |
| **Continuous Metaplastic Cascade (BF-Cascade)** | `research/gemini/r8` (Proposed) | Target: Task-agnostic retention | TBD | Zero task cues; zero replay buffer; passive intra-synaptic diffusion | **YES (Primary Candidate P2)** |

---

## 2. In-Depth Analysis of Forensic Defects in Prior Rounds

### 2.1 The DeepSeek R7 Round 43 Defects
- **Provenance**: In commit `c2742540009da3839979d74bde3b73f232c8ebd5`, DeepSeek tested DECOLLE-style local random readout ($g_i \cdot (p[\text{target}] - 1) \cdot \text{elig}$).
- **Result**: Showed hidden learning is real (+3.69pp over zero-update control, +3.67pp over shuffled control), confirming routing matters. However, it failed to beat the production scalar (+1.52pp, CI [-0.97, +4.01], crossing zero).
- **Flaws Disclosed**:
  1. Script crashed in summary formatting because dictionary subtractions failed; no JSON artifact was written at runtime. Results were transcribed from terminal stdout.
  2. Only 10 of 15 seeds were recovered because earlier seeds scrolled off buffer ($n=10, t(9)$).
  3. Hidden epochs were truncated from 120 to 30 mid-trial due to compute time constraints.
- **Auditor Verdict**: DeepSeek Round 43 cannot be treated as a completed confirmation run. However, its negative result on DECOLLE is consistent with the finding that richer fixed random feedback vectors do not resolve the credit bottleneck.

### 2.2 The Gemini R7 Continual Learning Defects (CA-MSC)
Forensic inspection of `src/scarx/evaluation/r7_hidden_plasticity.py` lines 592-639 revealed three severe experimental contaminations:
1. **Task Cue Leakage (`wrap_skill_pulse`)**:
   `x_batch` was padded from $D=8$ to $D=11$, injecting a $0.8$ pulse on channel $8 + s_{\text{id}}$ at tick 11. The network was not performing task-agnostic continual learning; it had an explicit task ID broadcast in its input stream!
2. **Replay Confounding (Double Update Budget)**:
   In `ew_he_ca_msc`, each fresh episode was followed by `buffer.sample(1)` and an immediate replay update, effectively giving the model $2 \times$ the update budget of the unconsolidated baseline. When ablated in a matched $2 \times 2$ factorial suite, CA-MSC without replay suffered $18.4\%$ mean forgetting (statistically indistinguishable from unconsolidated baseline at $17.88\%$).
3. **Oracle Task Boundary Snapshotting**:
   `model.snapshot_consolidation_anchors()` was explicitly invoked by the evaluation loop at task transitions, which requires an external supervisor oracle.

### 2.3 The Distractor Failure in EW-HE
- Under clean inputs, EW-HE achieved $83.1\%$ probe and $79.2\%$ direct accuracy on `strict_binding`.
- Under mid-delay distractors on ticks $\{3, 7, 8\}$, accuracy collapsed to $52.34\%$ (chance level).
- Causal origin: Distractor inputs trigger burst firing during the delay period, which resets membrane voltages and floods the eligibility trace $e_f$ with zero-mean noise, wiping out the subthreshold memory trace of the initial key-value pair.
- Preliminary fix: Gating eligibility updates during silent/noise delay ticks restored accuracy to $74.67\%$.

---

## 3. Mandatory Benchmark Integrity Assertions (BIA)

To ensure Round 8 is free from data leakage, task-cue artifacts, and capacity inflation, Sub-Agent A mandates the following five programmatic assertions that MUST run before and during all evaluations:

```python
# 1. Input Dimension Invariant
assert x.shape[-1] == 8, f"Input dimension violated: expected 8, got {x.shape[-1]} (task cues prohibited!)"

# 2. Hard Binary Spike Invariant
assert np.isin(spikes, [0.0, 1.0]).all(), "Spike outputs must be strictly binary {0, 1}"

# 3. Topology & Parameter Invariant
assert core.config.hidden == 48 and core.config.fan_in == 8, "Hidden core must remain H=48, fan_in=8"

# 4. Task-Agnostic Streaming Invariant
# No call to any external boundary snapshot method during stream evaluation

# 5. Matched Update Budget Invariant
# Exactly 1 update per episode; zero hidden replay updates unless explicitly matched
```

---

## 4. Evaluated Benchmarks & Protocol Baseline

Sub-Agent A approves the following benchmark suite for Round 8:
- **P1 Primary Target Tasks**:
  1. `strict_binding` (Key-value relational binding over temporal delay)
  2. `distractor_binding` (Binding under mid-delay distractor pulses)
  3. `xor` (Non-linear temporal parity over 12 ticks)
  4. `order` (Temporal sequence ordering)
- **P2 Continual Stream**:
  - Sequential stream `order -> xor -> strict_binding` with $D=8$ inputs (zero skill pulses).
- **P3 Multi-Key Associative Generalization**:
  - Scaled key-value associative retrieval across $2 \rightarrow 4 \rightarrow 8$ keys with random delays and overwritten values.
- **Registered Seed Sets**:
  - Pre-registered DEV seeds: `140011, 140017, 140023`
  - Locked Confirmation seeds: `129001, 129007, 129013, 129019, 129025`
