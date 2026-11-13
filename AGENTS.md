# SCAR-X working rules

- User clarification14/11: the roughly3GB remaining capacity is discreteGPU VRAM, not hostRAM. Use it for measured useful native-model/vision/ASR buffers, never dummy allocations or silent weight precision reduction. Keep physical mixed-workload reserve and honest failed candidate results; GPUmemory occupancy is not model quality.

- Latest mode amendment14/11: GPUmode routes VLM/vision, ASR and ALIF forward/local updates to CUDA; CPUmode uses no discrete GPU for neural compute (iGPU remains optional/unqualified). Host I/O/SQLite/tokenization/serialization/RNG stay CPU. Preserve native precision/local rules; GPU auto-rest keeps VLM warm, explicit sleep releases it. Device parity is not learned-skill/retention qualification. See docs/hybrid/WORKSPACE_GPU_RELIABILITY_20261114.md.

- Latest GPU amendment2026-11-13: user explicitly requested cpu/gpu modes and removed the SCAR-X RAM/private-commit cap for GPU mode only. Keep CPU4.5GB and immutable historical failures. GPU mode records RAM/shared/dedicatedVRAM separately; never hide physical OOM or lower native precision. Use native Windows SearXNG; user rejected Docker. Adapter teaching/data task remains pending behind search/GPU/speed priorities.

- Latest user approval2026-11-13 raises only the hybrid whole-system maximum to4,500,000,000decimalbytes, soft idle threshold4,250,000,000bytes. Context stays16,384 unless separately qualified. The completed context benchmark remains a4GB historical run; do not reinterpret it as a4.5GB PASS or change its frozen protocol/source/results.

- Later on2026-11-13 user explicitly raised only the hybrid whole-system RAM cap to4,000,000,000bytes. New-run resource amendment overrides the3GB line below; historical receipts and scientific thresholds stay unchanged.

- User approval 2026-11-13: isolated SCAR-X Hybrid may serve original-precision Qwen3.5-0.8B plus Whisper base FP32 and an ALIF local learner under a 3,000,000,000-byte whole-system budget. No quantization, pruning, precision reduction or LoRA. This exception applies only to `scarx.hybrid`/`run_hybrid.py`; original serving/neural modules and active peer experiments remain unchanged. See ADR0015 and the hybrid protocol. Historical 2GB results/gates remain historical.

- Load ponytail rules/tools/context first when available. Record unavailability honestly.
- Keep all SCAR-X changes inside this project; the parent contains unrelated font work.
- Start at README.md, then docs/PROJECT_MAP.md and docs/STATUS.md.
- Production neural code in src/scarx/neural must not import torch/autograd or use backward gradients.
- Non-negotiable user constraint: SCAR-X remains one Spiking Recurrent Neural Network with LIF/ALIF neurons and local synaptic learning. Never replace its core with a Transformer, pretrained LLM, or LLM proxy to improve chat/vision scores. Qwen and other architectures are isolated evaluation controls only, never serving/training components of the SCAR-X core. Preserve the CPU-first 2 GB target. Research changes must preserve this identity; do not silently change the architecture.
- Backprop controls live only in src/scarx/evaluation/controls.py and run as separate experiments.
- User approved proposal30 research on2026-11-10: a separate ALIF prototype may combine neuron-local forward eligibility with one-step learned readout-error feedback (W_out.T @ error). This explicitly relaxes signal locality; it is not permission for Transformer, whole-network/time backward, candidate autograd/RTRL, scaling or deployment. Production core/weights stay unchanged. Older pending notices describe their historical time.
- Register protocol and seeds before formal research benchmarks; never select a winner on final test data. Product-first MVP development and normal interactive teaching are authorized without research gates; record acceptance scenarios and limits honestly.
- Results go under runs/<unique-run-id>; do not rewrite previous run results.
- Trained checkpoints go in artifacts/, source fixtures in data/fixtures, user data in local/.
- Update docs/STATUS.md after a milestone; report failures and unmeasured claims.
- Do not expand to language pretraining or bigger models when the preceding gate has failed.
- Product entrypoint: run_app.py and README.md. Master plan: docs/RESEARCH_PLAN.md. Preserve historical research protocols/results; research gates do not block building a weak but real local MVP.
- Run pytest and CPU/native parity tests for neural/runtime changes.
- The localhost observer is passive: never require the user to select a model/source or click start to see agent work. Publish actual edits/checks through runtime.work_trace.activity with timestamps/evidence, and publish actual answers/learning measurements through the registered runner. Keep technical work separate from neural activity; never invent measurements or launch training merely to fill the page.

