# Pre-run amendment: unsigned-answer diagnostic

Registered2026-11-09 after the original v1 aborted before completing the first
learning condition. All three frozen controls finished; first full condition
raised `Target outside active readout` on a negative target. The first-position
decoder masks punctuation-only tokens, including the leading minus sign.

Keep the original run/source/protocol and add an aborted report. Do not label
this crash as catastrophic forgetting. No main algorithm/decoder is changed.

The replacement run uses `dataset(seed=8107,nonnegative_only=True)`: skip
subtractions whose result is negative BEFORE accepting a generated expression.
Same128/32/8 split, canonical deduplication, training budgets, order seeds,
conditions, stopping thresholds and historical forgetting gates as v1.
This is a NARROWER unsigned arithmetic diagnostic, not the original signed task.
Qwen reference receives exactly the same replacement questions and numeric-output
instruction. The original negative-output limitation remains unresolved.

Vocabulary is expanded from each presented lesson before the update slot for
ALL conditions, including frozen and guarded, to control output representability.
Only zero-initialized new columns/rows are added; targets are never input features.
The guarded learner's existing update/rollback is real, but this shared vocabulary
preparation differs from ordinary Hermes feedback where a rejected expansion is
also rolled back. Do not present this as an unmodified end-to-end serving trial.

Read-only diagnostic clones can reset working state; live/training state is not
reset. Persist learned checkpoints, test cold process reload and interference.
Qwen has frozen weights: no claim of online learning or synaptic retention from
its fixed-weight answers. Any scope difference and partial/time-limited run must
be reported, with no winner or live deployment selected from these probes.
