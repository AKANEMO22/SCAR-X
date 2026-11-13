# SCAR-X Hybrid v1 — locked protocol 2026-11-13

Approved by user: original-precision Qwen3.5-0.8B + Whisper-base FP32 + ALIF H128,
CPU-first whole-system≤3,000,000,000bytes; no LoRA, quantization, pruning or BPTT.
This registration precedes any formal hybrid quality benchmark. Old protocols,
fresh confirmation banks and active Gemini runs are not development data.

## Stages and stopping

P0: exact source hashes, native conversion audit (no lossy intermediate), numerical
parity, actual image/speech/text inference, full-state save/cold-load/rollback,
process-tree RSS/private-commit under limit. Include observer and any dedicated
Hermes/browser processes in integrated receipt. A resource or integrity failure
blocks P1–P3 and live promotion. Do not switch model/dtype or relax limits.

P1: licensed public Vi/En videos totaling1–2h. Freeze a manifest with license,
video split, event timestamps, hashes and independent answer annotations before
scoring.100DEV and100TEST questions; video-disjoint splits. Include dialogue,
visual, temporal, multimodal and absent/future-evidence cases. Engineering clips
and unit fixtures are not this benchmark. TEST is scored once, no tuning on it.
MVP gates:≥75% answerable QA,≥90% valid evidence citations,≥95% correct abstention.
Report counts/denominators and uncertainty rather than point estimates alone.

P2: independent seeds260113,260137,260171. Conditions: strongest fixed/rule policy,
frozen neural, readout-only, learned-hidden; shuffled feedback diagnostic.
Same input/features/candidate pool/source access/reply budget/wall-clock budget;
no task-ID, future calibration or private oracle passed into learner.
Primary autonomous metric: verified gap-resolution rate; at least5pp gain over
strongest control with lower95% CI>0. Primary memory metric: heldout grounded QA,
same5pp/CI gate. Learned hidden must beat readout-only and lose benefit after
hidden-reset/ablation; otherwise no hidden-representation claim. CI must account
for source-video/seed dependence; individual updates are not independent seeds.
Continuous20stage sequence without task-boundary inputs. Forgetting mean≤3pp,
no old skill group>5pp; acquisition and rejected updates also reported.
Freeze all gap/QA annotations and budget before P2; no final-test labels as rewards.

P3:8h mixed ingest/QA/review/feedback/rollback soak. Same limit, no OOM/corruption,
bounded queues. Separate capture/ASR/vision frontiers; report analyzed coverage
and backlog, including skipped analysis frames. Initial speed targets5tokens/s,
warm QA p95TTFT≤5s and p95semantic lag≤15s are unproven engineering gates.

At most two preregistered DEV adjustment rounds per hypothesis; no progress→PIVOT.
Integrity/resource failure→stop downstream measurement. Passing unit tests or
recall fixtures cannot establish MVP/lifelong/hidden/self-learning success.

## Provenance and outputs

Runs under unique `runs/<UTC>-hybrid-*`; never overwrite results. Record source
SHA+dirty-source hashes, config, all prompts/answers/evidence, action origins,
model/runtime revisions, sample predictions, resources and all failed attempts.
One OpenAI-compatible loopback gateway shared by Hermes and passive trace.
Model/ASR workers remain internal children; no cloud teacher, no exec/shell tools
for model-generated content. Local checkpoint state is data-only (no pickle).
