# SCAR-X ROUND 12: PROTOCOL & EXPERIMENTAL REGISTRATION
## Continual Memory & Replay-Lite Investigation for ALIF H48

**Date:** 2026-11-12  
**Author:** Gemini (Independent Continual Memory & Cross-Verification)  
**Branch:** `research/gemini/r12-replay-lite-continual-memory-20261112`  
**Base Commit:** `0116fa28beaa292914f061a203ae592f319fc66c`  
**Scope & Boundary:**
- ALIF H48, D8, Fan-in 8, hard spikes, single shared output head (outputs=2).
- Local online learning (own-reset baseline, no global BPTT).
- Task stream: Task 1 (Binding) -> Task 2 (XOR) -> Task 3 (Distractor) -> Task 4 (Order).
- Zero task-ID oracle given to learner or replay buffer.
- No future examples, no test data in replay buffer.
- RAM limit: < 2 GB; CPU-first.
- Confirmation bank `150001..150010`: **STRICTLY LOCKED**.

---

## 1. Audit & Pre-registration of Corrections (Phase 0)

1. **R11 Distractor RLS Value Correction:**  
   The raw JSON artifact `artifacts/r11_shared_learning_readout_results.json` records:
   `arm_c_joint_4096_rls.summary_per_task.distractor_binding.mean_direct = 0.645182` (**64.52%**).
   The text typo (65.69%) in the previous summary is officially corrected to **64.52%**.
2. **Train / Test Isolation:**  
   Confirmed 0/512 overlap between training sets (`seed + 200`), refit sets (`seed + 300`), ID test sets (`seed + 400`), and OOD test sets (`seed + 500`). Zero data leakage.
3. **Data Exposure Budget:**  
   Arm C in R11 verified to have 4096 updates = 1024 episodes/task * 4 tasks.
4. **Checkpoint Restoration:**  
   Confirmed bit-exact serialization and deserialization of ALIFCore and IndependentLearningALIF state.

---

## 2. Hypotheses Under Investigation

- **H1 (Rehearsal Necessity):** Catastrophic forgetting is primarily driven by the absence of past data reminders during sequential learning.
- **H2 (Hidden Synaptic Overwrite):** Recurrent synapses $W_{rec}$ and feedforward synapses $W_{in}$ are destructively altered by subsequent tasks.
- **H3 (Readout Substrate Overwrite):** Hidden representations still preserve task-separable features, but the shared linear readout $W_{out}$ is overwritten.
- **H4 (Gradient Interference):** Simultaneous gradients for new tasks destructively project onto subspaces critical for prior tasks.
- **H5 (Ultra-Lite Replay Efficacy):** An ultra-compact Reservoir Replay Buffer ($M \in \{16, 64\}$) is sufficient to stabilize both hidden dynamics and readout accuracy across a 4096-episode stream without task boundaries.
- **H6 (Synaptic Protection Requirement):** Reservoir replay alone is insufficient without structural weight protection (e.g. Synaptic Intelligence or orthogonal projection).

---

## 3. Matched-Compute Experimental Design (Phase 2)

All sequential arms observe the incoming stream of 4096 episodes (1024 episodes per task in strict sequence).

| Arm ID | Description | Incoming Updates | Replay Updates | Total Updates | Replay Buffer Policy |
|---|---|:---:|:---:|:---:|---|
| **Arm A** | Sequential Baseline (No Replay) | 4096 | 0 | **4096** | None |
| **Arm B** | Equal-Update Control (Recent-Replay) | 4096 | 1024 | **5120** | Extra update on most recent incoming sample |
| **Arm C** | Reservoir-16 Replay | 4096 | 1024 | **5120** | 1 sample drawn from Reservoir($M=16$) every 4 incoming examples |
| **Arm D** | Reservoir-64 Replay | 4096 | 1024 | **5120** | 1 sample drawn from Reservoir($M=64$) every 4 incoming examples |
| **Arm E** | Interleaved Upper Bound (from R11) | 4096 (mixed) | 0 | **4096** | Fully interleaved stream (oracle mixing) |

### Strict Fairness Controls:
1. **Update Parity:** Arms B, C, and D execute EXACTLY 5120 updates. If Reservoir-16/64 outperforms Arm B, the gain is strictly attributed to *retaining diverse historical experience*, NOT to extra compute.
2. **Replay Timing:** Replay sample is drawn and trained *before* inserting the new incoming example into the buffer.
3. **Task Agnosticism:** Reservoir sampling does not observe task phase boundaries or task IDs.

---

## 4. Evaluation Metrics & Thresholds (Phase 3 & 4)

Let $A_{p, t}$ be the accuracy on Task $t$ evaluated immediately after completing Phase $p \in \{1, 2, 3, 4\}$.
- **Acquisition Accuracy ($A_{t, t}$):** Performance on Task $t$ immediately after learning it.
- **Final Accuracy ($A_{4, t}$):** Performance on Task $t$ at the end of the entire stream.
- **Forgetting ($F_t$):** $F_t = \max_{p \le 4}(A_{p, t}) - A_{4, t}$.
- **Mean Forgetting:** $\bar{F} = \frac{1}{3} \sum_{t=1}^3 F_t$.
- **Worst Forgetting:** $F_{worst} = \max_{t \in \{1, 2, 3\}} F_t$.
- **Disentanglement:**
  - Frozen Hidden Probe ($P_{p, t}$): Linear probe refit on frozen hidden state representations.
  - Counterfactual Head Swapping: Swapping Phase 1 $W_{out}$ into Phase 4 network to distinguish readout overwrite from hidden feature degradation.

### Success Criteria (Exploratory DEV Thresholds):
- Acquisition $\ge 65\%$
- Final $\ge 65\%$
- Mean Forgetting $\le 3.0$ pp
- Worst Forgetting $\le 5.0$ pp

---

## 5. Candidate Scientific Verdicts

- `REPLAY_DEV_ADVANCE`: Reservoir replay (16 or 64) achieves Mean Forgetting $\le 3$ pp and Final $\ge 65\%$ across all tasks while matching compute in Arm B.
- `REPLAY_SPECIALIZED_GAIN`: Replay significantly improves retention over Arm B but fails the strict $\le 3$ pp / $\le 5$ pp forgetting threshold.
- `REPLAY_NO_RETENTION_GAIN`: Replay achieves no statistically significant retention benefit over Arm B (Equal-Update Control).
- `READOUT_FORGETTING_DOMINANT`: Hidden probes maintain high accuracy ($>80\%$) and counterfactual head restoration recovers performance, proving readout overwrite is the primary failure mode.
- `HIDDEN_FORGETTING_DOMINANT`: Hidden probe accuracy degrades significantly alongside direct readout, proving recurrent synaptic wipeout.
- `MIXED_INTERFERENCE`: Both hidden representation and readout experience substantial degradation.
- `BLOCKED_INVALID_CONTROLS`: Failure of integrity checks or compute parity.
