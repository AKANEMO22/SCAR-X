# SCAR-X ROUND 12: ERRATUM & METRIC RE-AUDIT

**Date:** 2026-11-12  
**Author:** Gemini (Independent Continual Memory & Cross-Verification)  
**Target:** SCAR-X Round 12 (`research/gemini/r12-replay-lite-continual-memory-20261112`)

---

## 1. Forgetting Metric Recalculation (Protocol Compliance)

### Defect Identified:
Protocol R12 formally specified the continual learning forgetting metric as:
$$F_t = \max_{p \in \{1, 2, 3, 4\}} A_{p, t} - A_{\text{final}, t}$$
However, the test runner in `scripts/run_r12_continual_memory_study.py` computed:
$$F_t^{\text{runner}} = A_{t, t} - A_{\text{final}, t} \quad (\text{Acquisition} - \text{Final})$$

Because rehearsal in Reservoir-64 during subsequent phases (e.g., during Distractor training) actually boosted Strict Binding representation beyond its initial acquisition level (from 81.64% to 98.24% on Seed 140023), computing $A_{\text{acq}} - A_{\text{final}}$ underestimated the drop from the network's true peak capacity.

### Re-audited Independent Calculation:

| Arm Configuration | Order Stream | Runner Mean Forgetting ($A_{\text{acq}} - A_{\text{final}}$) | Protocol Max-P Mean Forgetting ($\max_p A_p - A_{\text{final}}$) | Delta |
|---|:---:|:---:|:---:|:---:|
| **Arm A (Sequential Baseline)** | Order 1 | 36.31 pp | **37.13 pp** | +0.82 pp |
| **Arm A (Sequential Baseline)** | Order 2 | 26.97 pp | **30.58 pp** | +3.61 pp |
| **Arm B (Equal-Update Control)** | Order 1 | 39.50 pp | **39.56 pp** | +0.06 pp |
| **Arm C (Reservoir-16)** | Order 1 | 28.86 pp | **30.03 pp** | +1.17 pp |
| **Arm C (Reservoir-16)** | Order 2 | 19.31 pp | **22.35 pp** | +3.04 pp |
| **Arm D (Reservoir-64)** | Order 1 | 20.25 pp | **22.37 pp** | +2.12 pp |
| **Arm D (Reservoir-64)** | Order 2 | 10.68 pp | **13.45 pp** | +2.77 pp |

- **Official Corrected Protocol Values for Reservoir-64:**
  - Order 1 Protocol Mean Forgetting: **22.37 pp** (replaces 20.25 pp).
  - Order 2 Protocol Mean Forgetting: **13.45 pp** (replaces 10.68 pp).
- **Core Conclusion Unchanged:** Under exact matched compute (5120 updates), Reservoir-64 cuts protocol forgetting from **39.56 pp (Arm B)** down to **22.37 pp (Arm D)** (a massive net retention gain of **17.19 pp**).

---

## 2. Calibration Future Data Access Erratum

### Defect Identified:
`run_sequential_stream()` initialized calibration feature tracking using 8 unlabelled samples from all 4 tasks concatenated:
`calib_seqs = np.concatenate([datasets[t]["train"][0][:8] for t in task_order], axis=0)`
While these sequences lacked labels, this represented an unauthorized pre-exposure to future task channel dynamics before the sequential stream began.

### Strict Online Fixture Control:
A strict online calibration control using ONLY Task 1 samples (`train_x[:32]` of Task 1 exclusively) was executed:
- Task 1 Only Calibration: Strict Binding Acquisition = **86.91%**.
- Mixed Pre-Calibration: Strict Binding Acquisition = **89.26%**.
Conclusion: The ALIF core successfully calibrates and learns under strict single-task online initialization, with a minor calibration offset (-2.35 pp). Future benchmarks in Round 13 will strictly use single-task or registered independent synthetic white-noise fixtures for calibration.

---

## 3. Interleaved Stream Forgetting Re-classification

In `run_interleaved_upper_bound()`, all tasks are interleaved uniformly throughout training without discrete phase boundaries. Assigning `forgetting = 0.0` was a formal placeholder error. The metric is hereby officially reclassified as **`NOT_MEASURED`** for Arm E.

---

## 4. Re-classification of Task-Order Claims

Testing two permutations (Order 1 vs Order 2) is insufficient to prove universal task-order invariance. The scientific verdict claim is officially refined from "Universal Task-Order Invariance" to **`REPLAY_GAIN_IN_TWO_TESTED_ORDERS`**.

---

## 5. Clarification on R12 SSD Storage

The `SSDReplayStore` module implemented in Round 12 persists **episodic training sequences and labels** ($X$ sequences and integer $Y$). It is an **Episodic Experience Store**, NOT a neural engram store. Neural synaptic engrams (weights $W_{in}, W_{rec}$, normalization parameters, traces, and neuron IDs) will be implemented as a distinct architecture in Round 13.
