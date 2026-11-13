# CPU / RTX4050 speed comparison — 2026-11-13

User explicitly requested cpu/gpu config modes, switching to GPU, matched speed
measurement, native SearXNG without Docker, then removed the GPU RAM/commit cap.
CPU still4,500,000,000bytes. GPU uncapped by SCAR-X; physical VRAM/OOM is real.
Original weight precision, ALIF/H128/local rule and16Kcontext unchanged.

Three cases: Vietnamese generation, 1K public synthetic log + bounded answer,
one synthetic PNG.3repeats/case/mode, greedy seed260113/max128tokens. Same
checkpoint/template, b12360 source, context16K, batch128/ubatch64, no dummywarmup,
cache_promptfalse. CPU uses original CPU binary, GPU official matching CUDA12.4
binary/deviceCUDA0/all layers + projector. Thus this compares deployed profiles,
not bit-identical binaries. Do not claim that precision/knowledge changed.

Lock prompts/source hashes before measuring; no retries on wrong/slow answers.
Report individual answer, usage, TTFT, prefill/decode tokens/s, characters/s,
median per-case across3runs, RAM/RSS/privatecommit/sharedGPU and dedicatedVRAM.
Report startup/parity separately. Native numerical3case top1/Δprob<=.03 gate
remains unchanged. Generated language has no new quality PASS threshold.

Sequential one model only. Owned normal gateway is checkpointed/stopped, same
localhost serves read-only actual trace (all normalPOSTs503maintenance). The
benchmark owns one SearXNG child in both arms, includes Hermes trees in RAM,
does no learning or private retrieval. CPU cap sampled50ms; mixed-profile
runner has lifecycle-only WindowsJob because an ancestor CPU memory limit
cannot be removed by a later child. Report this difference from serving gate.
Per-call90s and total600s. Original serving config must be restored toGPU in
finally. Never stop/modify active peer/R13 or other GPU workloads; contention
is a limitation, not a reason to terminate somebody else's process.
