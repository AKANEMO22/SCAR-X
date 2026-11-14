# Feedback, synapse update and receipt now commit together

**Confirmed two real HTTP contract defects on RC8 source.** The actual shipped
`Handler` was executed unchanged against a disposable store and original ALIF
on CUDA0; only gateway/port wiring was supplied. No VLM, private conversation,
live question or live weight was used.

1. After a feedback checkpoint committed, failure writing the separate HTTP
   cache returned503. Retrying the same request then returned400 “Unknown
   answer/question identity”, although one neural update had already occurred.
2. Reusing a client ID with different feedback returned the old200 success.
   The cache did not bind an ID to the request's contents.

The earlier three UI/HTTP fixtures implemented their own receipts and did not
exercise this exact backend commit gap. Their results remain valid for those
frontend scenarios; they did not prove backend crash atomicity. Baseline JUnit
retains both expected failures.

## Repair and invariants

`HybridEngine.feedback_once` binds the request ID to a canonical hash of answer/
question ID, text, helpful rating and ordered relevance IDs. Under engine,
incident and store locks it runs the existing feedback behavior, then commits
its checkpoint, observations, incident journal and durable receipt in one SQLite
transaction. Cached identical requests return the original result; changed
content is rejected. Both `/api/feedback` and reply messages on the legacy
learning route use this path when a request ID is present.

Python's SQLite connection contexts normally commit at nested exits. A scoped
connection proxy defers those exits to the outer transaction while Store.lock
is held; simply adding an ordinary outer `with connection` would not fix it.
The underlying connection is restored in `finally`. Receipt/commit failure
rolls back SQL and restores the full pre-feedback ALIF/controller snapshot.
The lock order prevents a tool observer holding incident.lock while waiting for
Store.lock from deadlocking feedback completion. SQLite failures return503.

Receipts live outside neural checkpoints, so cold restart and intentional
neural rollback cannot turn a retry into an extra training step. Rollback still
means restoring old neural state: a cached ACK does not secretly reapply the
rolled-back update. Candidate labels are limited to128 total, with duplicate
IDs rejected before learning, bounding one transaction's work. Existing guards,
reward/local-learning equations, factual-verification rules and ALIF topology
are unchanged. A user's helpful rating is still utility feedback, not truth.

## Verification and source custody

Nine actual-loopback HTTP tests with isolated CUDA ALIF cover:

- The old post-commit cache failure and a real socket close **after commit
  before any response bytes**, followed by successful retry with no second update.
- Same ID/different content rejection and concurrent duplicates committing once.
- Cold restart and neural rollback with the durable receipt retained.
- Receipt write failure restoring full state, and an actual SQLite ABORT trigger
  leaving objects/checkpoints/events/evidence counts unchanged.
- Relevance-batch and owner-statement rollback together, plus bounded/duplicate
  candidate validation before updates.

The final release suite includes these cases, the original CPU/native/CUDA,
media, resource, checkpoint and installer checks. Final totals/activation are
recorded in [CURRENT_RELEASE](CURRENT_RELEASE.md). CuPy CUDA_PATH discovery
still warns; actual CUDA cases execute. The test harness initially omitted the
new sqlite3 import from its extracted namespace; it now uses the actual module
globals. Intermediate failure receipts remain separate, not rewritten.

`integrations/hermes/atomic_feedback_patch.py` is an exact insertion recipe over
the RC8 engine/store hashes. It leaves old methods unchanged. The release
packager checks that recipe and [current source qualification](evidence/FEEDBACK_ATOMIC_FINAL_SOURCE_20261115.json)
after the original qualified full-query/verifier patches, preserving both
provenance stages. Earlier7/8-case candidate qualifications remain historical.

Antigravity3.8Flash high/effort high was given an inline public-code review task.
It timed out without a final review, so no approval or test execution is claimed.
There was no retry/permission-bypass loop. Independent functional execution is
by Codex. No model/live state is sent to the CLI.

## Limits

This closes a delivery/persistence defect that could lose or duplicate the
meaning of feedback. It does **not** show hidden representation, causal tool
repair, long-term retention or an advantage over the same LLM+RAG. Requests
without an ID keep the legacy behavior; no receipt is invented retroactively
for replies lost before this update. Receipt history uses local SSD and does
not get loaded wholesale into neural RAM. No eight-hour test is restarted.
History-ablation P0 remains stopped at177 attempts; these are separate product
fixtures using the unchanged production learner, not extra history experiments.
