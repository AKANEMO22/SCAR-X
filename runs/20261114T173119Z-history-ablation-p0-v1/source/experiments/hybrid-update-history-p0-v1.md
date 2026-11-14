# History ablation P0 — locked execution specification

Execution steering received 15/11/2026: user answered the pending P0 question
“không cần chuyển sang mode cpu tajaj dùng antigravity cli để đẩy nhanh tốc độ”.
Proceed only with the isolated P0 scope of that question; retain GPU serving
and use Antigravity for assistance. This is not authorization for DEV, language
training, live promotion, main/R13 changes or automatic LoRA.

Parent: `hybrid-update-history-ablation-proposal-v1.md`. ALIF D64/H128/FP32,
topology, forward eligibility, readout feedback and statistics stay unchanged.
The pinned learner SHA is
`c38f7dd9412da3bc14a02a3a83611941d15aee1c558c05f4bd26671a6872111a`.
A copied method replaces exactly three hidden history/energy/delta statements;
AST comparison checks everything else. Serving imports no new research module.

## Fixed controls and engineering fixtures

Five arms: frozen, readout_only, current_hidden, current_only, matched. Seeds
260313/260337. Per seed, NumPy default_rng(seed+17) makes twenty finite FP32
normal(0,.2) sequences of shape (8,64), target step%2 in the relevance head,
reward1, identified **engineering fixture** evidence, not factual truth or a
language dataset. Save arrays before any learning. No prior/final DEV labels.

Each arm runs twenty CPU numerical-reference and twenty CUDA0 calls. For
current_hidden only, twenty additional original production-class CPU calls
verify the copied method against the pinned source. CPU is a numerical control,
not a serving-mode change. After the trajectories, per arm/seed: two same-device
next-update calls, two replay-continuation calls, one rollback trial call.
Total intended actual calls: 490; at most65 for any arm/seed. All calls charge
the same persistent append-only budget before computation, including originals.
No retries refund attempts. Global caps: **600 calls,100 per arm/seed,300 seconds**
from first charged attempt across restarts/reruns. Corrupt/partial/expired ledger
or a stale lock stops work; no reset. Pure numerical/serialization unit tests
make zero model.learn calls and are logged separately.

## Norm matching and immutable criteria

Energy/readout equations and weight bound±3 unchanged. Match the effective
hidden delta Frobenius norm **after both clips**, against hypothetical current
rule at matched arm's own state. Positive scalar≤256, at most20 bisections,
relative norm error≤1%. Target0 yields0 effective update; alpha1 is retained
when it already matches. Unreachable support/clip stops attribution, never
loosens norm/tolerance. Scalar norms use FP64 reductions; weights, direction,
trace and all synapse updates stay FP32. No additional learned-array shapes.
Diagnostic copies are reported as temporary audit memory, not new model capacity.

Required gates: copied current matches original; all arrays CPU/CUDA
rtol3e-5/atol3e-5; same-device checkpoint/replay continuation byte-exact;
input and recurrent weight deltas nonzero on learned-arm trajectories;
energy equal across rules for identical before-state/direction; zero-history
first step exact; expected zero-support mismatch explicitly rejected. Strict
arm/algorithm/encoder/checksum/shape/dtype/finiteness/topology validation,
production loader rejection, complete RNG/replay/counters and rollback exact.
No gradient/backward or new credit transport. Resource gate: physical CUDA0
free reserve≥512MiB at points checked; record host RSS/private-commit peak for
research process and label scope, not full-system proof. Stop on a gate failure.

## Evidence and interpretation

One uniquely named P0 run, source hashes before output, fixture NPZ, generated
method, charged ledger, per-step loss/deltas, raw norm arrays and isolated full
checkpoints; CPU/CUDA device placement and actual localhost events. Independent
audit recalculates norm/energy/history and ledger without new learning. Keep
the first pure-unit Windows lock-close failure and its repair; no synapse
updates occurred in that failed unit run.

Antigravity `gemini-3.8-flash-high`, effort high, completed inline mathematical
review in43.7s. Accept reachability/firststep/budget points; do not import its
Torch examples, unsupported float32-drift magnitude assertion or suggested live
loader modifications. Actual numerical tolerances are tested, not assumed.

P0 is formula/persistence engineering only. Even PASS **does not** establish
accuracy, retention, hidden representation or superiority to LLM+RAG. DEV remains
unapproved and cannot start from this execution steering.
