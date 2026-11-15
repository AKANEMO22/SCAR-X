# Experimental API-repair LoRA — not installed in Hermes

Base: Qwen/Qwen3.5-0.8B at2fc06364715b967f1860aea9cf38778875588b17.
Rank4/alpha8,159,744 trainable FP32 parameters;32steps on32 verified owned
alarm API corrections. Base original BF16/FP32 weights and unused MTP tensors
retained unchanged. No SNN/ALIF weight changes or training through its layers.

-`adapter/adapter_model.safetensors` + `adapter/adapter_config.json`: PEFT candidate.
-`adapter-f32.gguf`: llama.cppb12360 conversion;24F32 tensors, no quantization.
-`CANDIDATE_MANIFEST.json`: hashes, scope and qualification limits.

Pilot Transformers TEST exact/effect repair6/16→16/16, one seed, limited
templates/API; no general skill,20stage retention or production qualification.
Do not compare its absolute score directly with llama.cpp RAG scores from
another backend. GGUF conversion is not native inference acceptance.

Not merged into base, not loaded automatically, no new adapter routing policy
or live promotion. The separate untrained full-base save from loader failure
552928 is not this adapter and remains local/ignored by Git.

See docs/hybrid/P1B_AND_LORA_RESULT_20261115.md and the immutable run receipts.
