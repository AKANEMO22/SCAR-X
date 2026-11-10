# Knowledge-load audit / secondary probes

2026-11-09, after the unsigned primary run finished. Keep all raw runs unchanged.

1. Audit a metric implementation error: the primary field
`old_mean_forgetting_pp` was computed from the NET overall accuracy drop, which
can hide one family's loss behind another family's gain. Recompute per-family
forgetting from raw predictions using each family's maximum previous score
(baseline and earlier milestones), clip losses at0, then average8families.
Keep the same3pp/5pp thresholds. The acquisition stop rule did not use this
field, so no training or stopping decisions change. Report audited values in
new files; do not overwrite the misleading original field or historical report.

2. Qwen's numeric-output instruction was specified before its reference run.
Probe EVERY SCAR-X final checkpoint on the exact160prompt strings in that
reference data, using canonical independent state and no retrieval. This is
a secondary cross-model comparison, not the primary acquisition metric or
checkpoint selection. Different pretraining and memory budgets must be visible.

3. Exploratory rote-memory diagnostic: we have already seen that exact trained
wording sometimes works despite failed unseen wording. At each final checkpoint,
define a new set of facts correct on TRAIN wording; measure their retention after
10k quiet ticks,8new distractors*32slots, or24new/8old replay slots per distractor.
Independent generator seed = order seed+991. Load checkpoint afresh per branch,
evaluate on clones only. This measures retention of answers memorized at the
start of this secondary test, NOT immediate acquisition after original teaching
and NOT useful representation or calendar-month retention. Label this analysis
exploratory, with no pass declaration or changed primary success criterion.

The primary result remains a failure when unseen-wording acquisition fails.
No best seed/condition is deployed based on these follow-up measurements.
