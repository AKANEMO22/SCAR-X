# Hybrid pipeline engineering audit v1 — 2026-11-13

Request: check whether the current pipelines operate effectively. No training,
architecture change, model compression, main merge or peer/R13 intervention.
Ponytail tools/context were not present in the available inventory.

Use the unchanged serving code and original-precision assets at 16,384 tokens,
4.5 GB whole-system cap. Stop only the owned normal gateway gracefully, run the
actual gateway/HTTP/backend in a separate local data directory on the same port,
then restore the normal gateway. Retain its original database and weights.
Native constructors may be observed by the runner to record stage measurements;
no model response, inference or synaptic update is replaced/stubbed.

Predeclared engineering scenarios:

1. Real streamed chat: 17+25, exact answer42; measure first content and total time.
2. Actual function-call/tool-result round trip using local lookup_number(ALPHA)
   fixture17, clearly an engineering tool contract, not general agent competence.
3. Inline real PNG: circle color blue on an independently authored synthetic
   image; no fabricated answer. Failures remain recorded.
4. Upload the existing 21s FLEURS Vietnamese speech/synthetic-shape fixture.
   Observe capture/ASR/vision independently, backlog/coverage/errors and latency.
   Score circle colors at the start blue/end yellow. Compare normal agent route
   (raw-frame reinspection) with debug recall route (text evidence only).
   Report normalized word edit distance against the independently provided
   FLEURS transcript; this is one sample, not a corpus WER benchmark.
5. Actual capture of our generated labelled fixture window, microphone and
   system loopback OFF. Require actual frame and vision evidence; do not capture
   the user's desktop/app. System audio loopback remains untested in this audit.
6. Save/rollback in the isolated store. Snapshot/model/RNG/controller restoration
   checked against byte hashes; retain append-only evidence/audit.
7. Let the actual background scheduler review real answers for <=75s without
   hand-writing a model critique or pretending it is a factual label. Record
   review, questions/goals, verified replay and actual synaptic updates separately.
8. Exit the test process, restart against that same isolated database, compare
   persisted checkpoint/evidence counts and repeat the video start-color query.

Structural regressions/known failure paths are evaluated with controlled local
fixtures separately from real inference; they are never called quality evidence.
Use original weights/hashes and report every exception; request limit120s,
video completion limit150s, screen limit35s, overall work bounded15min excluding
source inspection. No retries on wrong answers, no tuning to these outputs.

Pass of plumbing is distinct from semantic correctness, real-time throughput,
useful autonomous learning, hidden-synapse advantage or lifelong retention.
Record actual RAM/private commit/shared GPU, native precision and all preemptions.
Historical protocols, thresholds and results remain unchanged.
