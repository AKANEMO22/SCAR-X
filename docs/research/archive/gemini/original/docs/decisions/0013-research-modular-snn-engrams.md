# ADR0013 — Research-Only Modular SNN Engram & Zero-Shot Composition Architecture

**Status:** Proposed / Research Exploratory (Round 13)  
**Date:** 2026-11-12  
**Author:** Gemini (Independent Continual Memory & Cross-Verification)  
**Scope:** Research Worktree Only (`research/gemini/r13-neural-engram-composition-20261112`). Does NOT alter `main`, Hermes, or production single-core ADR0006.

---

## 1. Context & Problem Statement

ADR0006 established a single 48-neuron ALIF population with a single recurrent array (`scarx-generation1`) for production Hermes sessions. However, experimental findings across Round 11 and Round 12 demonstrated that sequential training on multiple distinct relational and symbolic mappings in a single unsegregated recurrent pool induces mutual gradient interference and severe readout overwriting.

To explore the long-term architectural pathway toward scaling to large modular brains (where only active task-relevant neural modules reside in working memory and inactive cognitive modules are paged to SSD), Round 13 investigates an independent modular engram architecture:
- Sub-populations of ALIF neurons with stable global neuron IDs.
- Disk-resident neural synaptic engrams (weights, eligibility traces, normalization).
- Dynamic engram paging (unload from RAM, reload on demand).
- Neural composition via fixed or learned inter-module spike routing interfaces.

---

## 2. Decision

1. **Strict Research Boundary:** This modular SNN architecture is restricted exclusively to research exploratory work. It is NOT a production architectural change and does not modify production databases, live checkpoints, or Hermes runtime.
2. **Neuron ID Allocation:**
   - Module A: Neurons `0..47` (48 ALIF neurons).
   - Module B: Neurons `48..95` (48 ALIF neurons).
   - Router / Interface: Neurons `96..127` (32 ALIF neurons).
   - Total logical population: 128 neurons.
3. **Engram Persistence Protocol:**
   - Saved modules persist all synaptic arrays ($W_{in}, W_{rec}, W_{out}, \text{bias}$), normalization statistics ($f_{\text{mean}}, f_{\text{var}}$), momentum traces, and topology.
   - Modules can be completely unloaded from active RAM (`del module.core; gc.collect()`) and reloaded with bit-level execution fidelity.
4. **Composition Protocol:**
   - Composite tasks ($A + B \rightarrow C$) are evaluated purely via neural spike transmission without hardcoded lookup tables, Python dictionary shortcuts, or direct supervised training on task $C$.

---

## 3. Consequences

- **Positive:** Enables zero-shot neural composition ($X \rightarrow Y \rightarrow Z$) and eliminates catastrophic interference between unrelated tasks by physically isolating active parameter spaces.
- **Trade-off:** Requires an external engram paging store on disk and explicit neural interfaces between modules.
