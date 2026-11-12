# SCAR-X ROUND 13: PROTOCOL & REGISTRATION
## Modular Neural Engrams & Zero-Shot Neural Composition

**Date:** 2026-11-12  
**Author:** Gemini (Independent Continual Memory & Cross-Verification)  
**Branch:** `research/gemini/r13-neural-engram-composition-20261112`  
**Base Commit:** `b40d774854059f24212e0832725c81bf85c1523c`  
**Confirmation Bank (`150001..150010`):** **STRICTLY LOCKED**.  
**Production Checkpoints & DB:** Strictly untouched.

---

## 1. Research Objectives & Hypotheses

- **Primary Goal:** Verify whether two independently trained ALIF SNN modules (Module A learning $X \rightarrow Y$ and Module B learning $Y \rightarrow Z$) can be persisted as neural engrams on SSD, unloaded from RAM, restored on demand, and composed via a neural spike interface to solve $X \rightarrow Z$ ($C$) in a zero-shot manner (without direct training on $X \rightarrow Z$).
- **Core Hypotheses:**
  - **H1 (Engram Restorability):** An ALIF synaptic engram persisted on SSD can be restored with 100% execution equivalence, maintaining post-reload retention drop $\le 2$ pp.
  - **H2 (Zero-Shot Neural Composition):** Connecting the output of Module A to the input of Module B via a neural spike interface enables solving composite task $C$ ($X \rightarrow Z$) with $\ge 60\%$ accuracy (chance = 12.5%) without prior direct supervised exposure to task $C$.
  - **H3 (Ablation Specificity):** Severing either module, shuffling weights, or restoring an engram trained on an alternative mapping destroys performance on $C$, proving causal dependence on the specific learned synapses.
  - **H4 (Modular vs Single-Core Continual Advantage):** Modular engram isolation prevents catastrophic forgetting observed when training sequentially on a monolithic single-core SNN.

---

## 2. Experimental Task Specifications

- **Symbolic Domains:** 8 discrete symbols for $X$, 8 for $Y$, 8 for $Z$ ($K=8$, chance baseline = 12.5%).
- **Spatio-Temporal Encoding:** Each symbol is encoded as a 20-tick spatio-temporal pattern across 8 input channels with temporal burst signatures and Gaussian noise ($\sigma = 0.05$).
- **Mappings:**
  - Task A ($X \rightarrow Y$): Bijective mapping $f_A$ generated per seed.
  - Task B ($Y \rightarrow Z$): Bijective mapping $f_B$ generated per seed.
  - Task C ($X \rightarrow Z$): Composite ground truth $f_C(X) = f_B(f_A(X))$. Held-out 256 test episodes.
  - Task B' ($Y \rightarrow Z'$): Alternative mapping for causal engram substitution (Arm E).

---

## 3. Protocol & Execution Steps

1. Train Module A on $X \rightarrow Y$ for 1024 episodes.
2. Evaluate Module A acquisition on held-out test set ($A_{\text{acq}} \ge 80\%$).
3. Save Module A engram to SSD store (`engram_A`).
4. Unload Module A from active RAM (`mod_A.unload()`).
5. Train Module B on $Y \rightarrow Z$ for 1024 episodes.
6. Evaluate Module B acquisition on held-out test set ($B_{\text{acq}} \ge 80\%$).
7. Save Module B engram to SSD store (`engram_B`).
8. Train Module B' on alternative mapping $Y \rightarrow Z'$ and save to SSD (`engram_B_prime`).
9. Restore Module A and Module B from SSD into fresh uninitialized modules.
10. Evaluate post-reload retention of Module A and Module B.
11. Candidate Composition Inference ($A + B \rightarrow C$): Feed $X \rightarrow \text{Module A} \rightarrow \hat{Y} \rightarrow \text{Module B} \rightarrow \hat{Z}$.
12. Run 11 mandatory control arms.

---

## 4. The 11 Mandatory Control Arms

- **Arm A (A-only):** Load only Module A, omit Module B $\rightarrow$ evaluate Task C (chance $\sim 12.5\%$).
- **Arm B (B-only):** Load only Module B, omit Module A $\rightarrow$ evaluate Task C (chance $\sim 12.5\%$).
- **Arm C (Random modules):** Untrained A and untrained B $\rightarrow$ evaluate Task C (chance $\sim 12.5\%$).
- **Arm D (Shuffled synapses):** Shuffle synaptic weights of A or B $\rightarrow$ evaluate Task C.
- **Arm E (Wrong-memory restoration):** Pair Module A with Module B' $\rightarrow$ verify collapse on $Z$ and successful execution on $Z'$.
- **Arm F (Broken spike connection):** Sever spike transmission between modules $\rightarrow$ evaluate Task C.
- **Arm G (Same-core no-SSD baseline):** Monolithic H128 ALIF trained sequentially on A then B without modular separation.
- **Arm H (SSD-on vs always-resident):** Compare composite accuracy of SSD-reloaded models vs in-memory models (target delta $\le 0.001$).
- **Arm I (Retrieval-only control):** Raw episodic examples from store without synaptic weights $\rightarrow$ neural prediction collapses to chance.
- **Arm J (Frozen readout / frozen hidden):** Determine contribution of plastic hidden recurrent vs readout synapses.
- **Arm K (Oracle supervised ceiling):** Supervised ALIF trained directly on Task C ($X \rightarrow Z$) labeled `ORACLE_SUPERVISED_C`.

---

## 5. Decision Gates & Candidate Verdicts

- **Gates:**
  - $A_{\text{acq}} \ge 80\%$
  - $B_{\text{acq}} \ge 80\%$
  - Post-reload retention drop $\le 2.0$ pp
  - $C_{\text{direct}} \ge 60.0\%$ (8-way, chance = 12.5%)
  - $C_{\text{direct}}$ exceeds incomplete controls by $\ge 20.0$ pp
  - Consistent across 3 DEV seeds (`140011`, `140017`, `140023`).

- **Candidate Scientific Verdicts:**
  - `BLOCKED_A_B_NOT_LEARNABLE`: Module A or B fails to achieve $\ge 80\%$ acquisition.
  - `BLOCKED_INVALID_MEMORY_RESTORE`: Engram save/load fails checksum or roundtrip test.
  - `MEMORY_RESTORE_ONLY`: Engrams restore accurately, but composition $A + B \rightarrow C$ fails ($\le 60\%$).
  - `COMPOSITION_WITH_FIXED_SPIKE_ROUTER`: Modular engrams restore from SSD and successfully compose via fixed spike interface ($C \ge 60\%$, beating controls by $>20$ pp).
  - `LEARNED_NEURAL_COMPOSITION`: Inter-module routing is itself dynamically learned by a neural router population without $C$ labels.
  - `COMPOSITION_NO_GAIN`: Composition fails to exceed chance / baseline controls.
  - `INCONCLUSIVE`: Ambiguous or unverified outcome.
