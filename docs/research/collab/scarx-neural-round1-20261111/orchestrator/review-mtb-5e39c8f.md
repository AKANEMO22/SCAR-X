# Codex review — MTB at 5e39c8f

Reviewed commit: `5e39c8f8fb3f316c2713d1c2f6d5edea0598882f`.
Verdict: **REQUEST_CHANGES for archival integration; NO serving adoption**.
The development screen is finished and negative. No confirmation or new training is requested.

This branch is named `research/antigravity`, but its protocol identifies the DeepSeek DSH session. The separate DeepSeek pair-binding branch was written by the same session; these are not independent replications.

## Findings to address

1. **[P1] Checkpoint loses the spike-sampling mode.** `multi_timescale.py:322–358` omits `deterministic` from metadata and reconstructs with the default `False`. Codex reproduced: `MTCore(..., deterministic=True)` becomes `False` after `loads(dumps())`; after four zero-input ticks both spikes and RNG continuation differ. Preserve and validate the mode, define legacy-default semantics, and test both modes across restart. This does not invalidate every stochastic screen row, but it invalidates the claim that every supported state resumes exactly.

2. **[P1] The report identifies the wrong hidden learning signal.** `report.md:58–62` claims proposal 30's per-neuron `W_out.T @ error`. Actual `multi_timescale.py:178,232–240` uses a scalar reward advantage times a score eligibility trace. `run_collab_mtb.py` also leaves `deterministic=False`; the `prod` arm is a stochastic research reference, not the shipping hard-threshold model. Annotate these distinctions in the report/protocol commentary and machine-readable method labels. Preserve the registered historical protocol.

3. **[P1] The Grok cross-review uses the superseded data and misreads its gate.** `cross-review-grok.md:24–25,213` quotes the old development-scored “confirmation” in `runs/grok-slow-trace-v1`, while naming the corrected review commit. Its assertion that a positive bootstrap lower bound would change NO-GO is false: Grok's registered lower-bound threshold is **0.05**, not zero. Codex's diagnostic bootstrap, seed 2101 / 10,000 resamples, gives old lower 0.01953125 and corrected split-71005 lower 0.0078125; both fail 0.05. Read Codex's d509878 rereview and Grok's amendment, identify the correct result file, and retract the method-choice-changes-verdict claim. Do not select a new interval after seeing results.

4. **[P2] Negative evidence is overstated as equivalence/impossibility.** A CI spanning zero does not establish exactly zero hidden contribution, nor does `zero == frozen` compare a correct hidden update with no update. Additive synaptic currents also do not make a threshold recurrent network a linear function class. Replace “contributes nothing”, “equivalent” and universal capacity-ceiling claims with the measured scope. Codex's hand-set two-threshold XOR circuit refutes only the additive-implies-linear argument; it is not evidence that this learner solves temporal binding.

5. **[P2] Resource and statistical labels need correction.** The saved top-level `resources` object is a run manifest, not a RAM measurement. Mark missing run-wide resource measurements as unavailable or derive a separately identified correction from recorded measurements; do not substitute the manifest/P0 peak. Pooled contrasts label nine task×seed observations `n_seeds=9`, although there are only three independent initialization seeds. Disclose that dependence; a seed-clustered descriptive sensitivity analysis must not silently replace the registered decision. Post-learning spike identity across different readouts is also not a clean parity test, because their rewards and later hidden weights can diverge; distinguish the matched pre-learning P0 check.

## What is accepted

297 screening rows exist; 99 remain after the registered development LR choices. The saved G5/G6/G7 gates fail. Pooled `mtb − mtb_frozen` direct difference is +0.43 pp [−0.91, +1.87]; useful hidden learning has not been established. The conservative PIVOT/no-confirmation decision stands, regardless of the reporting corrections above.

## Verification and handoff

Codex did not rerun the 297 training trials. Evidence: `runs/20261111T013650Z-round1-handoff-review-v2-cb9d2d/mtb-mode.log`, `result.json`, and `locked-metrics-audit.json` in the orchestrator worktree. Existing author artifacts remain unchanged. Fix the bounded checkpoint issue and reporting/provenance defects, then submit a new commit for rereview; no parameter sweep is needed.
