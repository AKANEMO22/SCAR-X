# One shared experimental API-repair adapter,20-stage research version

This folder holds20versioned checkpoints of ONE rank4FP32LoRA, not20product
models/adapters. BaseQwen3.5-0.8B revision2fc06364715b967f1860aea9cf38778875588b17
is unchanged at nativeBF16/FP32. ALIF core/weights are not trained by these files.

-Latest PEFT: `stage-19/adapter_model.safetensors` + `stage-19/adapter_config.json`.
-Latest F32GGUF: `shared-adapter-final-f32.gguf` (format conversion only, not
 native generation acceptance for these new weights).
-`retention20-and-loss.png`: actual training/known-task curves, not SNN score.
-`CANDIDATE_MANIFEST.json`: hashes and evidence/qualification limits.

20stages/480optimizer updates,68unique TRAIN cases and repeated exposures.
KnownTEST40:29→40correct, finalforget0pp, transientAddReminder−12.5pp recovered.
Whole original experiment STOP/NOT-PASS at a comparator token-bound failure.
Independent diagnostic verifies the completed trajectory; separately corrected
RAG view is weak/copy-prone and cannot establish superiority. One seed, four
knownAPIs; not general/lifelong retention or production qualification.

Checkpoint `.pt` files contain trusted-created research optimizer/RNG state.
Use only safe weights-only loading for audit; serving never loads them. No
untrusted pickle inputs accepted. Data-valid RNG/optimizer storage doesn't prove
next-gradient-step numerical continuation. Adapter artifact reload logits were
checked separately in the corrected comparator run; no retraining there.

Not merged with base or installed inHermes. See
`docs/hybrid/SHARED_LORA_RETENTION20_RESULT_20261115.md` for raw receipts/limits.
