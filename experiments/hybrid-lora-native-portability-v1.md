# LoRA native portability and bounded old-skill regression — registered before calls

User authorized train/create LoRA and continued work. This verifies the existing
adapter, no training/hyperparameter search or auto promotion. Reuse original
pilot TRAIN-excluded20repair probes (4DEV/16TEST), plus12new deterministic general
controls locked in source: arithmetic4, JSON formatting4, source-reading4.
Same isolated pinned llama.cppb12360 process/native0.8B model/dtypes, adapter
request scale0 vs1. No live server/global scale/config change, model merge,
quantization or precision reduction. Original HF pilot claims stay historical.

Read local llama.cpp server README: --lora-init-without-apply, per-request lora
id/scale and global-vs-request semantics; converter F32 metadata/norms already
checked. Portability may differ numerically or semantically from Transformers.
Not a repeat of the original backend test; no retraining to fit native results.

Max64generation requests,64newtokens each, temperature0/repeat_penalty1.05,
same prompts and budgets per scale; no response_format to mask model errors.
Use existing public structured API data; TEST not used for model selection.
Calls32×2=64 includes repair20 and general12. Resource/time cap15minutes,
physical GPU reserve512MiB at sampled points; process owner PID/ctime verified,
new native port18786 on loopback only, no unknown process termination. Launch
hidden isolated worker only; graceful owned cleanup. On failure STOP, preserve
partial records, no automatic retry/seed/scale sweep. Pause if foreground serving.

Independent exact/replayed tool effects for repairs; math/JSON/source controls
exact gold. Require repair improvement>=5pp/lower95%CI>0 and mean old-skill
regression<=3pp/no family>5pp; no reducing sample criteria after seeing results.
If uncertain/fails any gate, keep candidate unqualified. General controls12 and
single seed are limited; this is not20stage lifelong retention, vision or full
production validation. All actual prompt/output/latency/resources and failures
on localhost, clearly native adapter research rather than SNN updates.
