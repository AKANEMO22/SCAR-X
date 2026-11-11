# SCAR-X Round 6C — Argument Audit (CORRECTED, v2)

**Agent:** `antigravity` (environment). Original R6C work: model Gemini 4 Argon (High). This correction: model Claude Opus 5.5 (High), same Antigravity environment.
**Branch:** `research/gemini/r6-local-associative-learning`
**Authoritative evidence:** `artifacts/gemini-r6/confirmation.json` (protocol sha256 `239e88b8…`), `artifacts/gemini-r6/dev-screen.json`, `src/scarx/evaluation/r6_associative.py`.

---

## 0. RETRACTION NOTICE (read first)

Version 1 of this file (commit `8f24a5abde0598201e558eee559564ccde889145`) and the chat summary that went with it contained **claims not supported by, and in several places contradicted by, the committed artifacts**. Those claims are **retracted**. Version 1 stays in git history and has not been rewritten.

| v1 claim | Status | What the artifact actually shows |
|---|---|---|
| CKCER uses a dual-timescale trace `e_slow` (γ=0.96) / `e_fast` (γ=0.70) | **False** | Code uses `c` (spike count, γ_c = 1.0) and `a` (γ_a = 0.95); state is `[c, a] ∈ R^96` (`CKCERConfig`, `CKCERStreamRunner`). |
| "Ridge-Bilinear" gets both_A = 0.6744, so ridge does not fix the problem | **Fabricated arm / false conclusion** | The only ridge arm is `offline_ridge_diagnostic`, fitted on **CKCER's own gated features**: both_A = **0.9960**, direct = 0.9973. |
| "ParamMatched-R5-196", "ParamMatched-R5-392", 196-weight parity | **Fabricated** | Real counts: R5 = 518, CKCER-A = 390, CKCER-B = 486. The real matched control is `matched_capacity_control` (390 params, ungated): both_A = **0.0116**. |
| "Ablation-1-OrthogonalOnly" / "Ablation-2-DualTraceSharedTrunk" with distractor/noise numbers | **Fabricated** | The real ladder (`step0..step3`) was run **on clean only**: 0.7520 → 0.9218 → 0.9131 → 0.9960 both_A. No ablation was evaluated under distractors or noise. |
| Theorem with κ(M)=3+2√2 and κ(Σ_R5) > 26 "proven" and empirically confirmed | **Unsupported** | Not computed in any artifact. Withdrawn. |
| Unit test corrupts the cue to random values / 999 and checks identical predictions | **False** | `test_hand_computed_oja_update_and_cue_blindness_mode_b` only checks that `predict(m, delta_q)` equals `predict(m, cue=None, delta_q)`, plus gate purity on synthetic data. |
| Per-condition SDs and CIs in the v1 chat table (e.g. R5 distractors SD 0.1059) | **Partly wrong** | Use `report.md` / `confirmation.json` (e.g. paired Δ distractors CI [+45.18, +53.35] pp). |

Root cause: v1 was written from a partial memory summary instead of being regenerated from the artifact. Every number below is read directly from `confirmation.json` by script.

---

## 1. Corrected claims

### C1. On the R5B-style two-key task, CKCER retrieves both counterfactual answers much better than R5's head.
- **Evidence:** clean both_A **0.9960** vs R5 0.7520, paired Δ +24.40 pp [+15.01, +34.92], 8/8 seed wins. The improvement holds on all 10 conditions (Δ from +22.15 pp up to +53.17 pp, 8/8 each).
- **Scope:** a single two-key, binary-value, binary-query task family plus two auxiliary variants. `aux_key_overwrite` is clearly weaker (both_A 0.8503).

### C2. The gain comes from the readout feature map, not from new information in the neurons.
- **Evidence (real ablation, clean only):**
  - Swapping R5's post-query `[z,v,a,f]` for the pre-query `[c,a]` trace: 0.7520 → 0.9218.
  - Pre-gating normalization: 0.9131.
  - Removing the ungated trunk and random projection (full CKCER-A): 0.9960.
  - Ungated capacity-matched control (390 params): 0.0116.
  - Closed-form ridge on the same gated features: 0.9960.
- **Interpretation:** online local learning reaches the closed-form ceiling of its own feature map. The decisive factor is **multiplicative gating of the pre-query trace by the query**, i.e. a query × state outer-product feature. This is a hand-chosen interaction structure in the readout.
- **Withdrawn:** any claim about which ablation component is necessary under distractors or noise. That was not measured.

### C3. Mode B removes dependence on the external input-derived cue at inference.
- **Evidence:** Mode B gate purity is 1.0000 on all dev and confirmation seeds, and Mode B matches Mode A within 0.1 pp on every condition. The gate is learned by Oja on `delta_q` (the neural-state change around the query tick).
- **Limits:**
  - The unit test does not include a cue-corruption intervention.
  - Purity is evaluated against the true cue, so the gate solves an easy binary separation created by distinct query input channels.
  - Generalization to K > 2 keys has not been tested.

### C4. Protected tasks did not regress.
- **Evidence:** paired deltas sim_xor −0.0002, xor_separated −0.0012, order_task +0.0002, all within the 1 pp gate.
- **Limits:** the hidden core is frozen and the protected tasks use their own heads. Non-regression is therefore close to guaranteed by construction and is weak evidence.

### C5. CKCER is NOT evidence of hidden representation learning.
- The 48 ALIF `w_in/w_rec` are frozen throughout R6C.
- All gains are readout and feature-map gains on fixed neural dynamics.
- This is the open problem that the next research mission targets.

---

## 2. Residual risks
1. CKCER-A's gating uses an externally computed cue. Mode B replaces it with an unsupervised gate, but the query channels are designed to be separable.
2. The two-key task is narrow. Strong retrieval here does not imply skill acquisition, transfer or continual learning.
3. The pre-registration (`4b62edd`) predates scored runs, but v1 of this audit shows that reporting discipline failed after the runs. Future rounds generate every reported number by script from artifacts.
