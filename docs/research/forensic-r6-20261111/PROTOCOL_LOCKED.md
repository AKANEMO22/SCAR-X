# Locked forensic audit protocol — P0 to P3

Date: 2026-11-11. Author: Codex. This is an audit/reproduction registration,
not approval for a new candidate, fresh seed confirmation, scaling or deployment.

Sources: main `495dab2f4bac0180e4d1cd28674391f055ccf09d`;
DeepSeek R6D `4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893`;
Grok R6B `c39e9db7771582d9933af402a209917328b71119`;
DeepSeek R5B `d6c609df6f90fc8df96fb3de424dd54388f8dc59`.
Other branch snapshots are pinned in BASELINE_INVENTORY.json. Gemini's dirty
local R6 branch is not silently promoted to a final committed experiment.

## P0/P1

Record original Git blobs, local/remote heads, run manifest/source.zip hashes,
protocol chronology, package versions, protected production fingerprints and
all omissions. Read attached JSON only as claims. No editing peer worktrees,
history, main, live SQLite, Hermes or live model state.

The reference evaluator uses the public event specification independently of
agent summaries: two distinct writes on channels0–3, one query on4–5, background
on6–7. Query location is observed per sample. Pulse threshold0.5; float32
subtracted-query noise equality tolerance2e-7. Pair identity, stored facts,
query-A/query-B order, answer labels and supplied flip mask must agree. `both_A`
requires both siblings correct when values differ; `both_B` does so when equal.
Direct and needB are separate measurements. Classify missing cases as null,
not numeric zero or a PASS. Oracle correctness and common-noise counterfactual
validity are distinct checks; report both on every scored stress sample.

Negative regression cases: wrong tick, raw/wrong-scaled feature, wrong label,
pair swap, changed sibling noise, feature-no-op shift, direct-for-both_A,
unsupported True/missing G8, changed training order and duplicate/shadowed code.
Tests must exercise the real NoClip candidate, not the shadow class in its test.

## P2 R6D historical reproduction

Use only previously published historical confirmation seeds:
130001,130007,130013,130019,130023,130029,130031,130037.
Replay the existing 2×2/scaler/data grid, matched updates and stress, then the
existing controls. Preserve training arithmetic, generator, initialization,
orders, 1024/16384 budgets, calibration256, held1024 and paired256 groups.
These seeds are seen: classify all execution as HISTORICAL_REPRODUCTION,
even though the original script internally labels them FRESH_CONFIRMATION.

Redirect output only; instrument evaluation/training boundaries to save original
predictions/labels, facts, query, pair IDs, input/features/scaler/order/head
digests. Do not change predictions to satisfy the reference evaluator. Preserve
strict counterfactual violations as evidence instead of aborting before saving.
Compare every stored metric to source results and independently recompute pair
metrics. Compute seed-paired bootstrap10,000 / RNG2101 and report full precision,
worst seed, and mean vs all-seed threshold sensitivity without changing history.
Original G1/G6 means retain their registered semantics. Never combine differing
head/orders into a fictional same-head integrated result. Null/missing G4/G5/G8
prevents a global PASS; a valid failed gate remains FAIL even if others missing.
Check temporal/distractor oracle on actual tex/ptx, and actual common-noise
relation separately. Repeat/parity coverage is explicitly scoped to measured
seeds/conditions; no universal deterministic claim from one repeat.

## P3 R5B/R6B historical reproduction

Use the exact published eight R5B reproduction seeds and three R6B screen seeds,
as recorded in the pinned runner; never substitute new ones. Keep the effective
last Python definitions and real NoClip imported class for exact replay.
Record shadowing in the original file and publish an unambiguous audit wrapper
without rewriting history. Save paired predictions and original trained/ridge
feature digests; audit all registered-vs-executed conditions. No absent stress
is filled by a new experiment. Expected historical .6676172/.9858056 and noclip
gain .00982 are comparison targets, not forced constants or passing criteria.

## Limits and decisions

One numerical thread; CPU-only. Each reproduction child has a Windows2GB
per-process cap where available plus sampled process-tree RSS/commit watchdog;
kill only owned child/tree on limit. Stage timeout1200s (R6D matrix),600s
(controls),600s (R6B); any timeout leaves immutable partial evidence. No API,
new model downloads, autograd candidate, fresh data budget or new hyperparameters.
Source snapshots stay fixed; historical predictions may differ due numeric
environment and that is recorded as REPRODUCTION_FAILED, not fixed by tuning.

Claims use PASS/FAIL/INCONCLUSIVE/INVALID_HARNESS/NOT_TESTED with evidence levels.
R6D frozen hidden cannot support a learned-hidden claim regardless of head score.
No fresh confirmation will run. R6E/R7 is a separate pending proposal after audit.
