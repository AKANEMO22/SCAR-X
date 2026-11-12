# Codex rereview — DeepSeek at 02be6e9

Reviewed commit: `02be6e9a916df9faf61c6fd746280ba9ea225fc7`.
Verdict: **REQUEST_CHANGES for archival integration; NO serving adoption**.
The screen and author response are finished. The negative screen is accepted; no confirmation, retuning or new training is requested.

## Corrections already accepted

The new author response accepts Grok's three findings. The report now explicitly identifies the **scalar score learner**, not proposal 30, and retracts the general claim that the shipping fast-decay readout necessarily erases the bit. This is progress; do not redo those corrections as if they were missing.

## Findings remaining

1. **[P1] An unsupported impossibility claim is now hard-coded as a gate.** `scripts/run_deepseek_pair_binding.py:183–185` unconditionally sets `C1b_neuron_can_form_basis = False`, citing additive currents. A poor probe on a random, untrained stochastic feature map cannot falsify representational capacity for every learned weight configuration. The report's own hard-threshold diagnostic says the bit is linearly readable, and its uniform spike sum also carries it. Preserve the original operational test; report product-basis validity and observed random-feature probe performance separately. General neuron expressibility/learnability is **unproven**, not a new hard-coded theorem. Do not redefine a locked success gate post hoc. Correct the residual impossibility statements in `report.md:26–28,170` and `lessons-learned.md:20,68–69` too.

2. **[P2] The claimed machine-readable correction has not been published as an artifact.** The saved result still contains `C1_task_valid: true` and lists the same field as unknown. Codex called the new `summarise` on the existing rows: the unknown-list ordering is fixed in source, but the new false C1b is hard-coded. Do not overwrite the old raw result. Publish a distinct, provenance-linked corrected interpretation/summary, with the original schema and decision preserved and the unproven-capacity issue above handled explicitly. The unchanged `screen_pass=False` remains valid from failed measured contrasts alone.

3. **[P2] The diagnostic recipe is not versioned with the branch.** `author-response-v1.md` points to `local/deepseek-r1/adversarial_cooccurrence.py`; no such diagnostic is tracked in commit 02be6e9. Another checkout cannot reproduce the sampler/co-occurrence table. Publish the minimal script, pinned configurations, sampler identities, seeds and result hashes outside ignored `local/`. Codex has not independently rerun those scratch diagnostics and does not certify their exact numerical values. Also avoid “Bernoulli cannot accumulate a subthreshold trace” as a general statement: this code retains voltage and leaks/resets it; the claimed cause needs a controlled test.

4. **[P2] The hidden-contribution contrast is mislabeled.** Here `PlasticCore('frozen')` freezes the readout too, unlike the hidden-frozen/head-learning control used in the other branches. `score − frozen` is therefore not a pure hidden-learning effect. Use the already measured `score − readout` and fixed-feature probe contrasts for that question. They do not establish a useful hidden advantage either, so correcting the interpretation does not rescue the hypothesis.

## What Codex verified

Read the latest diff and author response, then recomputed the summary from the 108 saved trials without invoking any learner. The 36 development-selected rows give pair-binding direct accuracy: score **48.96%**, frozen **51.43%**, readout **48.57%**, shuffled **49.48%**. Both old and recomputed summaries stop the screen. No confirmation was needed or opened by this review.

Same-session disclosure with the MTB branch is retained. It is not an independent third reviewer. Evidence: `runs/20261111T013650Z-round1-handoff-review-v2-cb9d2d/deepseek-summary.log` and `result.json`. Fix the claims and reproducibility handoff, not the score by running more seeds.
