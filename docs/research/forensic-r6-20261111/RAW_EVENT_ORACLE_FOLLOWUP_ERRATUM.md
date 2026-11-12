# Oracle follow-up erratum — preserve the ambiguous observation

First data-only run `20261111T104720Z-forensic-gemini-events-v1-dd3666` stopped
after87,552checked rows with zero mismatched labels. The next case, historical
seed121037 train/query_tick_variation, had an extra marker above.5 at row680.
The raw error is preserved. This is not evidence that the generated latent label
is wrong: move_query adds old tick10noise to tick11noise, so a threshold detector
can be ambiguous even though the pulse/latent task remains well defined.

Keep the marker threshold.5 and all original inputs/labels. Before retrying the
data-only check, extend the evaluator-only latent contract for this specific
transform: decode base events first, build their pulse tensor without using y,
then independently apply the specified linear row/tick movement to both the
base tensor and event tensor. Require the transformed raw tensor to match the
source move_query output. Parse transformed events to establish latent labels.
Do not feed those event tensors to a model or use them to change any prediction.

For both query-move and noise conditions, report raw detector ambiguity separately
and compare the source input-cue reader to the independently established latent
query. Do not interpret ambiguity as label corruption or silently increase the
threshold to make the data pass. No neural forward/training/fresh seed is involved.
