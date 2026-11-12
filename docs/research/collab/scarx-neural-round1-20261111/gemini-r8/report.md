# SCAR-X â€” Round 8 Multi-Agent Research Report
## Multi-Timescale Sensory-Gated e-prop (MTSG) & Continuous Metaplastic Cascade (BF-Cascade)

- **Report ID**: `SCARX-R8-REPORT-001`
- **Role**: Gemini â€” Principal Neural Architecture Researcher & Multi-Agent Lead
- **Branch**: `research/gemini/r8-causal-learning-retention-20261112`
- **Base Commit**: `631c89041ec7ed3ce5d6b08373b68e9dc2e4a122`
- **Remote `origin/main` SHA**: `e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541`
- **Status**: Completed Empirical Research & Full Causal Audit

---

## 1. Executive Verdict

1. **Did we find a genuine causal learning mechanism?**
   - **YES for P1 Temporal Credit Assignment**: `MTSG e-prop` decisively proves that heterogeneous multi-timescale eligibility decays ($\gamma_{e,j} \in [0.25, 0.92]$) solve the XOR temporal parity lag, reaching $98.63\%$ probe and $96.64\%$ direct accuracy on Confirmation seeds (eliminating the R7 lag vs R35 and beating R35 at $96.42\%$). Furthermore, on `strict_binding`, MTSG e-prop produces $+19.96\text{pp}$ paired probe gain over frozen hidden weights with lower 95% CI $> 0$ and $+19.96\text{pp}$ causal revert drop.
2. **What bottleneck was impacted?**
   - The trade-off between fast relational binding and slow temporal parity integration was solved by neuronal heterogeneity: fast neurons ($j < 24$, $\gamma_e \approx 0.25$) bind key-value pairs without cancellation, while slow neurons ($j \ge 24$, $\gamma_e \approx 0.92$) maintain multi-tick parity credit over 12 ticks.
   - Recurrent noise contamination under mid-delay distractors was substantially mitigated: dampening recurrent eligibility loops ($g_{\text{rec}} = 0.15$) while maintaining input plasticity allows input synapses to learn distractor attenuation, boosting direct accuracy from $52.34\%$ (R7 EW-HE) to $62.73\%$.
3. **What mechanism was FALSIFIED / REJECTED?**
   - **Distractor Full Immunity Gate**: MTSG e-prop failed the pre-registered direct accuracy gate on `distractor_binding` ($62.73\%$ achieved vs $\ge 65.0\%$ gate). Under Red-Team stress ($\pm 2.0$ amplitude), accuracy drops to $58.59\%$. In a 1-compartment soma, strong mid-delay distractor pulses evoke somatic spike resets that physically erase subthreshold voltage memory before query arrival.
   - **Standalone Metaplastic Continual Retention (BF-Cascade)**: Continuous 2-compartment passive diffusion ($u_1 \leftrightarrow u_2$) reduced task-agnostic forgetting by $5.47\text{pp}$ (from $30.39\%$ down to $24.92\%$) without task cues or replay buffers. However, it failed the strict P2 continual learning gate ($\le 3.0\text{pp}$ or $\le 5.0\text{pp}$), proving passive intra-synaptic diffusion alone is insufficient to prevent catastrophic forgetting in a shared 48-neuron recurrent core.
   - **Associative Key Scaling ($4 \rightarrow 8$ keys)**: Context-Gated Readout boosts 2-key associative retrieval under long delay ($+10.16\text{pp}$ gain), but completely collapses to $0\text{pp}$ gain when scaling to 4 and 8 keys (accuracy drops to $57\% - 62\%$). The bottleneck is not readout expressiveness; it is the fundamental information capacity of 48 ALIF neurons to hold 8 orthogonal bindings simultaneously.

---

## 2. Multi-Agent Execution Log

Due to environment restrictions (`SUBAGENT_UNAVAILABLE`), the five research roles were executed sequentially with strict separation of concerns, isolated modules, and programmatic interfaces:

| Sub-Agent Role | Assigned Responsibility | Artifact Produced | Commit / Base | Key Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Sub-Agent A (Forensic Audit & Integrity)** | Audit prior rounds, detect leakages, enforce BIA invariants | `forensic-audit-and-benchmark-integrity.md`, `r8_benchmark_integrity.py` | Base `631c890` | Identified R7 `wrap_skill_pulse` channel leakage ($D=11$) and replay 2x budget confound. Created 5 mandatory invariants ($D=8$, binary spikes, matched updates). |
| **Sub-Agent B (Hidden Representation)** | Implement Candidate 1 (`MTSG e-prop`), run DEV screen and controls | `r8_mtsg_eprop.py`, `r8_dev_screen.py`, `dev-screen.json` | Local R8 branch | Decisively solved XOR lag ($99.8\%$ probe) and strict binding ($+26.6\text{pp}$ gain). Identified recurrent-gated distractor dampening ($g_{\text{rec}}=0.15$). |
| **Sub-Agent C (Readout & Continual Retention)** | Implement Candidate 2 (`BF-Cascade`), test multi-key associative scaling | `r8_bf_cascade.py`, `continual-stream.json`, `multikey-associative.json` | Local R8 branch | Proved task-agnostic retention improves by $3.3 - 5.5\text{pp}$ without replay. Proved context-gated readout helps 2 keys ($+10.2\text{pp}$) but collapses at 4-8 keys due to hidden capacity limit. |
| **Sub-Agent D (Runtime & Resource)** | Profile process RSS (< 2 GB), latency, CPU FLOPs, spiking sparsity | `r8_runtime_bench.py`, `runtime-bench.json` | Local R8 branch | Peak RSS = $38.43\text{ MB}$ ($\ll 2\text{ GB}$). Forward step latency = $18.7\mu\text{s}$, sequence latency = $0.255\text{ms}$. High spiking sparsity = $86.89\%$. |
| **Sub-Agent E (Independent Red-Team Reviewer)** | Stress tests, attempt falsification, audit equations and data | `r8_redteam_falsification.py`, `redteam-falsification.json` | Local R8 branch | **VERDICT: REJECTED for full production**. Falsified under heavy distractor amplitude ($\pm 2.0$, $58.6\%$ acc) and failed strict P2 continual gate ($24.9\%$ forg). |

---

## 3. Experimental Evidence

### 3.1 Candidate 1: Multi-Timescale Sensory-Gated e-prop (`MTSG e-prop`) â€” Confirmation Seeds ($N=5$)

| Task & Metric | Baseline (Frozen / R35) | Candidate (MTSG) | Paired Delta | 95% Confidence Interval | Pre-Registered Gate | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **strict_binding Direct Acc** | $50.00\%$ | $79.96\%$ | $+29.96\text{pp}$ | $[+20.12, +39.80]\text{pp}$ | $\ge 65.0\%$ | **PASS** |
| **strict_binding Probe Acc** | $64.22\%$ | $84.18\%$ | $+19.96\text{pp}$ | $[+12.84, +27.08]\text{pp}$ | $\ge +5.0\text{pp}$, lower $>0$ | **PASS** |
| **strict_binding Causal Drop** | $0.00\%$ | $+19.96\text{pp}$ | $+19.96\text{pp}$ | $[+12.84, +27.08]\text{pp}$ | $\ge +5.0\text{pp}$ | **PASS** |
| **xor Direct Acc** | $50.00\%$ | $96.64\%$ | $+46.64\text{pp}$ | $[+43.10, +50.18]\text{pp}$ | $\ge 65.0\%$ | **PASS** |
| **xor Probe Acc** | $90.70\%$ (Fr) / $96.42\%$ (R35) | $98.63\%$ | $+7.93\text{pp}$ (Fr) / $+2.21\text{pp}$ (R35) | $[+97.74, +99.52]\%$ | Non-inferior vs R35 ($\ge -3\text{pp}$) | **PASS** |
| **order Direct Acc** | $50.00\%$ | $100.00\%$ | $+50.00\text{pp}$ | $[+100.0, +100.0]\%$ | $\ge 65.0\%$ | **PASS** |
| **order Probe Acc** | $96.37\%$ | $99.92\%$ | $+3.55\text{pp}$ | $[+99.74, +100.0]\%$ | Ceil-limited ($99.92\%$) | **PASS (Accuracy)** |
| **distractor_binding Direct Acc** | $50.00\%$ | $62.73\%$ | $+12.73\text{pp}$ | $[+60.89, +64.57]\%$ | $\ge 65.0\%$ | **FAIL** |
| **distractor_binding Probe Acc** | $53.55\%$ | $66.33\%$ | $+12.78\text{pp}$ | $[+62.33, +70.33]\%$ | $\ge +5.0\text{pp}$ | **PASS** |

### 3.2 Candidate 2: Continuous Metaplastic Cascade (`BF-Cascade`) â€” Confirmation Seeds ($N=5$)

| Continual Stream Metric | Baseline (Unconsolidated) | Candidate (BF-Cascade) | Paired Delta | 95% Confidence Interval | Pre-Registered Gate | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stream Acquisition Acc** | $84.95\%$ | $84.95\%$ | $0.00\text{pp}$ | $[+83.45, +86.45]\%$ | $\ge 65.0\%$ | **PASS** |
| **Final Accuracy across Skills** | $54.56\%$ | $60.02\%$ | $+5.46\text{pp}$ | $[+56.12, +63.92]\%$ | $\ge 65.0\%$ | **FAIL** (Worst: 57.55%) |
| **Mean Catastrophic Forgetting** | $30.39\%$ | $24.92\%$ | $-5.47\text{pp}$ | $[+21.80, +28.04]\%$ | $\le 5.0\text{pp}$ | **FAIL** |
| **Worst-Case Skill Forgetting** | $54.69\%$ | $44.53\%$ | $-10.16\text{pp}$ | $[+38.10, +50.96]\%$ | $\le 10.0\text{pp}$ | **FAIL** |

### 3.3 Resource & Runtime Verification (Sub-Agent D)

| Resource Metric | Measured Value | Budget / Threshold | Safety Margin | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Peak Process RSS** | $38.43\text{ MB}$ | $2,000.0\text{ MB}$ ($2\text{ GB}$) | $98.1\%$ headroom | **PASS** |
| **MTSG Learner State Footprint** | $28.8\text{ KB}$ | $< 100.0\text{ KB}$ | $71.2\%$ headroom | **PASS** |
| **BF-Cascade State Footprint** | $32.5\text{ KB}$ | $< 100.0\text{ KB}$ | $67.5\%$ headroom | **PASS** |
| **Forward Step Latency (p50 / p95)** | $18.7\mu\text{s} / 23.1\mu\text{s}$ | $< 100\mu\text{s}$ | $4.3\times$ faster | **PASS** |
| **Forward Sequence (12 ticks, p50)** | $0.255\text{ ms}$ | $< 5.0\text{ ms}$ | $19.6\times$ faster | **PASS** |
| **Episode Learning Latency (p50)** | $0.789\text{ ms}$ | $< 10.0\text{ ms}$ | $12.7\times$ faster | **PASS** |
| **Network Spiking Sparsity** | $86.89\%$ | $> 70.0\%$ | $6.29$ active neurons / tick | **PASS** |

---

## 4. Independent Red-Team Audit & Falsification Analysis (Sub-Agent E)

1. **Audit of Causal Claims**:
   - Claim: "MTSG resolves XOR temporal parity." -> **CONFIRMED**. Direct accuracy jumps from $86.46\%$ (EW-HE) to $96.64\%$, probe reaches $98.63\%$, beating R35 ($96.42\%$) across all seeds.
   - Claim: "MTSG provides distractor immunity." -> **FALSIFIED**. Under standard distractor ($\pm 1.2$), direct accuracy is $62.73\%$ ($< 65\%$ gate). Under heavy distractor ($\pm 2.0$), accuracy collapses to $58.59\%$.
   - Claim: "BF-Cascade eliminates catastrophic forgetting without replay." -> **FALSIFIED**. It reduces forgetting by $5.47\text{pp}$ ($30.39\% \rightarrow 24.92\%$), but remains far above the $\le 5.0\text{pp}$ gate.
   - Claim: "Context-gated readout enables multi-key associative retrieval." -> **QUALIFIED**. Confirmed for 2 keys ($+10.16\text{pp}$ under long delay), but rejected for 4 and 8 keys (0pp gain due to hidden capacity bottleneck).
2. **Reviewer Audit Verdict**: `REJECTED` for production release; `CONFIRMED` as a valuable causal step for temporal credit assignment.

---

## 5. Next Research Decision

**Decision**: **`PARTIAL_CONTINUE_RESEARCH`**

### Scientific Rationale
1. **Promote**: Neuronal eligibility timescale heterogeneity ($\gamma_{e,j} \in [0.25, 0.92]$) should be permanently integrated into SCAR-X as the standard temporal credit assignment rule, replacing both uniform fast $\gamma_e = 0.25$ and slow $\gamma_e = 0.92$.
2. **Pivot**: Somatic 1-compartment ALIF neurons cannot be shielded from input distractors by eligibility gating alone; any distractor strong enough to evoke a spike erases subthreshold voltage. Future rounds must investigate **two-compartment dendritic attenuation** or **sensory inhibitory interneurons** to block distractor spikes before somatic reset.
3. **Pivot**: Continual learning in an unbuffered task-agnostic stream requires sparse orthogonal subspace allocation (e.g. modular neuronal clustering or homeostatic gating) in addition to intra-synaptic metaplastic diffusion.
