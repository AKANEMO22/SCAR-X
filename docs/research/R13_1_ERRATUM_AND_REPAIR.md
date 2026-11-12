# SCAR-X ROUND 13.1: ERRATUM & CAUSAL REPAIRS
## Rigorous Neural Engram & Modular Architecture Auditing

**Date:** 2026-11-13  
**Author:** Gemini (Independent Continual Memory & Cross-Verification)  
**Target:** SCAR-X Round 13 (`research/gemini/r13-neural-engram-composition-20261112`)  
**New Working Branch:** `research/gemini/r13-1-spike-composition-vs-small-llm`

---

## 1. Audit Findings & Systematic Repairs

### Defect A: Symbolic Router vs True Direct Spike Communication
- **R13 Implementation:** The composition bridge executed:
  `pred_y = argmax(probs_a)`
  `input_b = patterns_y[pred_y]`
  This decoded Module A's output into a discrete Python integer symbol, looking up a clean synthetic pattern for Module B. While valid as a functional baseline, it did not represent pure neural spike communication.
- **R13.1 Repair:**
  1. Maintained `fixed_symbolic_router` as a reference baseline.
  2. Implemented `direct_spike_router`: Module A's full temporal spike raster $S_{A} \in \{0, 1\}^{T \times 48}$ across all 20 time ticks is directly projected via an inter-module synaptic matrix $W_{\text{inter}} \in \mathbb{R}^{8 \times 48}$ into input currents for Module B tick-by-tick.
  3. No scalar symbol conversion, no Python integer lookup in the candidate pipeline.

### Defect B: Seed Independence in Symbol Pattern Generation
- **R13 Implementation:** `generate_orthogonal_symbol_patterns()` defined `rng = np.random.default_rng(seed)` but used deterministic modulo indices `ch1 = s % dim` and `ch2 = (s + 3) % dim`. Consequently, $X$ and $Y$ patterns had identical channel active layouts across seeds.
- **R13.1 Repair:** Implemented `generate_independent_symbol_patterns()` where channel activations, onset timing, burst durations, and amplitudes are pseudo-randomly sampled per seed, guaranteeing that $X$, $Y$, and $Z$ reside in genuinely independent spatio-temporal feature spaces.

### Defect C: Frozen Hidden Control Forward Pass Fix
- **R13 Implementation:** Arm J (Frozen Hidden control) updated `w_out` repeatedly on stale `core.features` without evaluating the sequence forward pass across training iterations.
- **R13.1 Repair:** Each episode now runs a complete forward sequence pass `step(row, track=False)` through the frozen recurrent core, computing correct dynamic hidden features before applying the readout update.

### Defect D: Fair Monolithic SNN Comparison
- **R13 Implementation:** The monolithic H128 ALIF received identical raw input representations for both Task A ($X \rightarrow Y$) and Task B ($Y \rightarrow Z$) without task context, creating immediate output collision.
- **R13.1 Repair:** Provided the monolithic baseline with fair contextual awareness (an active context channel indicator identifying task phase), allowing it to attempt joint representation without artificial ambiguity.

### Defect E: Sample-Level Wrong-Memory Ground Truth
- **R13 Implementation:** Arm E computed $Z'$ using `i % NUM_SYMBOLS` instead of the exact symbol identity of each test sample.
- **R13.1 Repair:** Tracks the ground-truth input symbol index $x_s$ for every single sample, computing $Z' = f_{B'}(f_A(x_s))$ exactly.

### Defect F: Complete Mid-Episode Neural Checkpointing & Continuous RSS Profiling
- **R13 Implementation:** Saved weights, traces, and normalization, but omitted instantaneous membrane voltage $u$, adaptation state $a$, output spikes $s$, and RNG state. RSS was only sampled once at program completion.
- **R13.1 Repair:** Full state serialization now includes `core.voltage`, `core.adaptation`, `core.spikes`, `core.features`, eligibility traces $e_v, e_a, e_f$, and RNG bit-generator state. RSS memory is sampled continuously throughout training and inference phases.
