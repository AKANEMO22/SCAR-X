# Additional committed Gemini handoff — historical audit only

User explicitly requested checking Gemini's newly completed results during the
R6D/R6B audit. Pin local/remote `8f24a5abde0598201e558eee559564ccde889145`,
`research/gemini/r6-local-associative-learning`. No peer files will be edited.

Recompute the eight already-scored seed records and replay only the same
historical bank121011,121017,121023,121029,121031,121037,121041,121043 at the same
fixed settings. These are neither fresh nor a new confirmation: they already
appear in DeepSeek R6A, and Gemini itself has two confirmation runs with different
source hashes. Compare original/corrected protected-task generators and baselines
without merging their gates into a fictitious pristine confirmation.

Record actual paired prediction scores and independently compute conjunction
metrics from labels/query-pair grouping. Where a raw oracle has not been checked,
say so; labels alone do not prove event correctness. Test the actual CKCER class
for exact inactive-branch zero updates. Audit whether Mode B's Oja weights change
the recurrent hidden or only its query readout. No new candidate or fresh seed,
hyperparameter selection, teacher, architecture change or P2 pilot is permitted.

CPU one numerical thread, process-tree2GB watchdog, engineering≤120s and one
historical replay≤300s; immutable unique output. Numerical reproduction does not
repair post-hoc selection, seed reuse or protocol mismatch. Proposals remain
pending and production/main remains unchanged.
