# Bind actual web evidence and the original task to review

Actual RC10 post-turn review did not receive the web results used to answer.
The answer record contained a global conversation-FTS hit unrelated to JoJo,
and a bare follow-up was recorded only as `tìm kiếm đi`. A reviewer without
the original task and actual sources proposed irrelevant gaps or called a
source-supplied date hallucinated. No automatic truth label or weight update
was created; the existing unverified-review boundary held.

## Repair

- Pair only current-turn `tool` messages with their `web_search` call IDs and
  query. Reject failed/unpaired/prior-turn results and unsuitable source URLs.
  Public facts no longer get unrelated global personal-memory FTS context.
- Persist search snippets, FTS rows and provenance atomically. IDs bind session,
  tool call, query, URL and content view. Identical retries reuse records;
  a shorter host-compression view is a new immutable observation, not an
  overwrite or a fatal conflict. Retain up to12 snippets/36KB per collection.
  Time means gateway observation time, not source publication or fact-validity.
- All entries stay `observed`, including input that supplies `verified:true`.
  No page is fetched here, no extraction provider invoked and no source becomes
  a factual verifier. Source truncation/partial coverage is recorded explicitly.
- Keep the actual human message in the transcript and answer record. Store the
  resolved `task_question` separately; use it for review, uncertainty goals and
  explicit source-relevance feedback. Internal tool-continuation scaffolding is
  not the human question. This corrects task identity, not the hash encoder or
  learning equation; older records fall back to their original question.
- Supply explicit, nonfactual interpretation of a lowercase colloquial `bn`
  in a Vietnamese quantity question. Original text is retained; uppercase `BN`
  is not rewritten. The metadata is not evidence or proof of learned language.
  Context remains bounded and marks long-question excerpts.
- Review now sees the bound task, raw follow-up and five source previews with
  modalities/coverage. `idle_review_started` logs their IDs and task for trace.
  Finalization without schemas can retain source/task context without restoring
  tool execution permission. Native output and critiques remain unverified.

## Verification and limits

332 release Python checks passed, including source pairing, cold SQLite reads,
idempotent observations, shorter content views, actual SQLite abort rollback,
source-status guards, task/follow-up preservation, review/relevance binding and
the original CUDA/CPU/native/checkpoint/feedback/installer regressions. CuPy
CUDA_PATH discovery warns; CUDA cases actually execute.

A single private actual-GPU review on the owned RC10 web receipt confirmed
five supplied sources reached the reviewer and produced zero ALIF updates.
Its output **still misinterpreted `bn` as title/date** before the explicit task
interpretation was supplied. That failed semantic output is retained privately;
input-contract success is not review-quality success. Actual Hermes acceptance
and natural idle review after installation are recorded separately, not assumed
from fixture tests.

The existing Source/feedback patch recipe was extended over exact pinned bytes;
the prior qualified recipes/results remain. [Final source qualification](evidence/WEB_CONTEXT_RELEASE_SOURCE_20261115.json)
anchors code and regression receipts. No upstream framework was installed.

This is an application information-path repair, not a new learning-rule study.
[Literature-first intake](../research/literature/REUSE_DECISIONS_20261115.md)
separates reusable experience/reflection patterns from unproven neural benefit.
All SNN-vs-RAG, hidden contribution and retention gates remain unchanged;
history P0 stays stopped at177 attempts and the cancelled eight-hour run stays
cancelled. Original precision, ALIF and the GPU profile are retained.
