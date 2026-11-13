# SCAR-X CONTINUAL LEARNING INDEPENDENT AUDIT REPORT (REPAIRED)
**Evaluated Model**: `Qwen/Qwen2.5-0.5B-Instruct` (Unquantized bfloat16, 100% Frozen)
**Model Weight Size**: `988.1 MB` (Weights alone well within 2,000 MB)
**Peak Process RSS**: `1121.1 MB` decimal (`1069.1 MiB` binary)
**Strict 2.0GB Gate Result**: `PASS`
**Random Seeds**: `[140011]` | **Device**: `cpu`
**Oracle Trace Discrepancies**: `0` (Audit re-evaluated all traces directly from private metadata)

## 1. Master Comparative Summary: End-to-End Correctness vs Tool Routing

| Variant | Description | EXP-01 E2E Acc [95% CI] | EXP-01 Tool Routing | EXP-02 E2E Recovery | EXP-03 Gen E2E Acc | EXP-04 Forgetting | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Frozen LLM Only | **20.0%** [3.6%, 62.4%] | 60.0% | 0.0% (0/4) | **20.0%** | 0.0% | 9833.9 ms |
| **B** | Static RAG / SQLite | **20.0%** [3.6%, 62.4%] | 60.0% | 50.0% (2/4) | **40.0%** | 0.0% | 11219.4 ms |
| **C** | Plastic Associative Memory | **60.0%** [23.1%, 88.2%] | 80.0% | 0.0% (0/2) | **40.0%** | 0.0% | 11164.2 ms |
| **D** | Plastic Memory + Idle Rehearsal | **60.0%** [23.1%, 88.2%] | 80.0% | 0.0% (0/2) | **60.0%** | 0.0% | 6899.3 ms |
| **E** | Static RAG + Idle Rehearsal | **20.0%** [3.6%, 62.4%] | 60.0% | 50.0% (2/4) | **20.0%** | 0.0% | 6802.8 ms |
| **F** | Simple Feedback Ranking (EMA) | **20.0%** [3.6%, 62.4%] | 20.0% | 50.0% (2/4) | **40.0%** | 0.0% | 6358.4 ms |
| **G** | Contextual Linear Router (Fair Baseline) | **60.0%** [23.1%, 88.2%] | 80.0% | 0.0% (0/2) | **60.0%** | 0.0% | 6573.4 ms |

## 2. Hypothesis Verification Ledger (Strict Empirical Falsification)

- **H1 (Knowledge & Skill Acquisition)**: [CONFIRMED] (Variant C E2E: 60.0% vs Variant A: 20.0%)
- **H2 (Rapid Mistake Adaptation on 2nd Trial)**: [REFUTED] (Variant C Recovery: 0.0% vs Variant B: 50.0%)
- **H3 (Autonomous Idle Learning Improvement)**: [CONFIRMED] (Variant D Gen: 60.0% vs Variant C: 40.0%)
- **H4 (Zero Catastrophic Forgetting & Stability $\ge 90\%$)**: [CONFIRMED] (Variant C Stability: 100.0%, Forgetting: 0.0%)
- **H5 (Strict RAM Limit $\le 2.0\text{ GB}$ = 2,000,000,000 bytes)**: [CONFIRMED] (Total Process Peak RSS: 1121.1 MB decimal vs 2,000.0 MB limit; Weights alone: 988.1 MB)

## 3. Plastic Associative Memory (C) vs Contextual Linear Router (G)

- Variant C (Plastic Associative Memory) Routing Accuracy: **80.0%** (E2E: 60.0%)
- Variant G (Contextual Linear Baseline) Routing Accuracy: **80.0%** (E2E: 60.0%)
- Routing Difference (C - G): **+0.0 pp**

## 4. Plastic Associative Memory (C) vs Simple Feedback Ranking (F)

- Variant C Routing Accuracy: **80.0%**
- Variant F Routing Accuracy: **20.0%**
- Context-specific plasticity routing advantage: **+60.0 pp**

## 5. Autonomous Idle Learning Impact (D vs C and E vs B)

- Variant D (Plastic + Idle) vs Variant C (Plastic Only) Unseen Gen: **+20.0 pp**
- Variant E (RAG + Idle) vs Variant B (RAG Only) Unseen Gen: **-20.0 pp**