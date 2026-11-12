# R8 snapshot adapter repair P0 — engineering only

Authorized by the current request to process remaining issues and Directive V2's
engineering work allowance. This is not approval to change a learning signal,
run a quality campaign or migrate the app. Implement in forensic/evaluation only;
peer source `fe472b3b1257cdb264805bf79fd53e5647e3568d` stays unchanged.

Persist every known MTSG/BF model array and scalar, including RNG, actual sensory
threshold, w_ctx even when disabled, normalizer, recurrent working/eligibility
state and source config. Caller-provided bounded JSON controller is stored whole;
unknown learner/core fields rejected. New data-only research format, no import
selected by payload, no pickle. Explicit matching model template required.

Gate: exact state roundtrip and a fresh-process next-forward/next-local-update
fixture continuation for each model; preserve caller replay/cursor/RNG JSON.
Only one tiny source-rule update per model continuation, no quality data/score.
Unit tests may use <=8fixture updates per case. This does not implement or certify
a live streaming learner/controller, mid-episode optimizer continuation, SQLite
or automatic replay/consolidation.

Validate metadata schema/checksums, all NPY shapes/dtypes/body sizes before array
allocation, finite values, indices, decay ranges and counter/RNG types. Reject
NaN/float64/duplicate/huge declared headers, preserve template on invalid load.
Atomic save; model rollback restores the entire snapshot via replacement model.
Per-process2GB cap and sampled process-tree RSS/commit; CPU1numericalthread.

No global/backward/BPTT, no candidate equations or main/live edits, no language
training, no confirmation. Preserve failed fixture logs and old R8 artifacts.
P0PASS is state-engineering compatibility only, never representation/retention GO.
