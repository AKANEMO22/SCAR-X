# Native2B VRAM utilization probe — registered14/11 before full download/load

User clarified remaining~3GB means RTX4050 VRAM and asks to use the discreteGPU.
Current serving all-neuralCUDA is verified; this probe concerns the frozen VLM,
not growing ALIF, changing synaptic rules or replacing the SNN architecture.
Do not use a larger VLM to claim an SNN advantage. Keep same-model RAG/SNN
comparison separate. No quantization/pruning/precision reduction or adapters.

Candidate official `Qwen/Qwen3.5-2B`, pinned revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`.
Safetensors bytes4,548,221,488, SHA256
`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.
Header inspected before this registration: language3,763,655,360bytes,
vision662,833,152bytes, MTP121,656,320bytes; BF16 with originalF32 tensors.
Header metadata is not a runtime fit or quality PASS.

Bounded technical probe: official source, native-precision GGUF repacking using
existing pinned converter, dtype/coverage readback; actual all-layerCUDA load at
same16K/BF16KV. Same-machine observed globalVRAM and owned-PIDdedicatedVRAM,
hostRSS/commit, concurrent/sequential FP32ASR and ALIFCUDA. No CPU/SSD weight
offload or dtype lowering to claim it fits. Stop/recover on OOM or native error;
restore0.8B service from outer shell. Do not kill other agents/experiments.

Three locked engineering inputs (exact arithmetic, latest supplied setting and
image-color), plus the previously failed workspace/context-use cases as DEV
regressions (not fresh test). Freeze prompts before inference; report alloutputs,
load/TTFT/decode and peak-memory. No repeat-until-pass. For candidate deployment,
need native source numericparity, no integrity/resource failure, and at least
512MiB globalVRAM headroom in mixed inputs; if unavailable mark constrained,
do not silently deploy a full-VRAM unstable model. Three-seed neural parameters
stay unchanged. A larger-model improvement is VLMcapacity, not SNNlearning.

Downloads local/private only, <=12GBdisk includingexports. This is a feasibility
probe, not permission for language pretraining, LoRA, H256 or formal hybridP2.
