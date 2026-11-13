# Hybrid context engineering benchmark v1 — 2026-11-13

User requested context-window comparison. Isolated evaluation only, no training,
main changes, production-context change, precision change or peer interruption.
Ponytail was absent from the available tool inventory.

## Locked comparison

- Original native Qwen3.5-0.8B BF16/F32 and vision projector, same llama.cpp
  b12360, CPU threads=4, batch=128/ubatch=64, one slot, native BF16 KV.
- Sequential contexts 16,384 / 24,576 / 32,768. Disable context shifting and
  prefix reuse for evaluation; temperature=0, seed=20261113, output<=96 tokens.
- Four short exact-answer tasks (revision, arithmetic, code, absent evidence).
  Two independent 12K-token synthetic long logs (relational and temporal).
  One 22K and one 30K log measure admission and longer-input behavior only.
  All prompts, target labels, measured templated token counts and source hashes
  written and hashed BEFORE generation. No selection on answers, no retries.
- Same six inputs are the paired comparison. A request too large for an arm
  is **not admitted**, not an incorrect answer. No silent truncation, retrieval,
  context compaction, feedback or SNN learning. Extra input is not a fair
  same-input quality comparison and is reported separately.
- JSON exact keys/values and type-sensitive scoring. Parse failure, truncated
  output or error stays in the denominator for admitted tasks. Record all raw
  stream events, answer, reference, timings, first-content latency and resources.

## Resource and stopping rules

Total cap 4,000,000,000 bytes, including Hermes process trees, sampled shared
GPU allocation, evaluation controller and native worker. Sample RSS/private
commit every 50 ms; GPU counter every 2 s (conservative addition). Positive
difference between prior gateway RSS and benchmark-controller RSS is reserved
and reported separately, never disguised as measured RSS. Do not change the
OS pagefile or exploit weight paging/quantization. ASR/capture not run: passing
this text pilot cannot qualify concurrent video, audio, save or rollback.

Stop the **current arm** at the first cap breach; stop its own worker only.
Try remaining arms once, even if a preceding arm failed (background client
usage may fluctuate). No extra attempts or relaxed cap. Per-request 180 s,
startup 120 s, total inference window <=1,200 s. No fresh training.
Checkpoint and stop only this owned gateway gracefully; temporary observer
serves actual benchmark trace at the same localhost:8775, then restore the
unchanged normal service in a finally block. Never touch active Gemini/R13.

## Decision before seeing results

Do not increase the default unless: no resource failure, >=150 MB headroom
(peak accounted <=3.85 GB), all six common tasks complete, no fewer common
correct answers than 16K, at least one extra-length task correct, and median
common-input first-content latency <=1.25x baseline. Tiny synthetic cases are
engineering DEV only, not statistical proof of general model superiority.
If baseline itself breaches RAM, first repair runtime headroom; no expansion.
Full quality/lifelong gates and historical thresholds remain unchanged.

Publish RAM, load time, TTFT/prefill/decode speed, admitted/completed/correct
separately. Context capacity is a temporary working buffer, not persistent
neural learning or video retention. No online model comparison is implied.
