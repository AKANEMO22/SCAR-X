# Native-precision LoRA error-repair pilot — separately attributed from ALIF

Approved15/11/2026: user explicitly permits training and creating LoRA. One
small adapter candidate for local Qwen3.5-0.8B; no automatic serving promotion,
adapter merge, hidden-SNN credit claim or change to ALIF learning equations.

## Literature/code and provenance

LoRA original arxiv2106.09685 + PEFT0.19.1 lora/layer.py source were read; use
standard rank decomposition through installed PEFT rather than inventing an
optimizer. Inspect local Transformers5.22.2 Qwen3_5ForConditionalGeneration:
logits_to_keep supports bounded output-logits allocation; adapter calls are
normal VLM backprop. This exception does not permit backward in SNN.
Reuse original pinned model2fc06364715b967f1860aea9cf38778875588b17 and local
ToolTalk e05f4ce6 actual API errors/valid repairs, independent exact transition
checker. No quantization, pruning or lowered precision. No paid teacher API.

Source safetensors have452 BF16 tensors and36 FP32 tensors (873,436,192 BF16
parameters and2,592 FP32). Every loaded tensor must retain its original dtype;
if loader auto casting changes FP32 tensors, explicitly restore those tensors
from source before use. Frozen base weights checked by hashes/samples and
requires_grad false. New LoRA matrices may be FP32; no reduction of base dtype.

## Data / target

Fresh32TRAIN/8DEV/16TEST owned bilingual alarm-repair requests, partitioned
wording/time values before inference. For every case execute original faulty
AddAlarm request and a valid corrected request on an independent snapshot.
Correction target compact JSON {"api_name":"AddAlarm","parameters":{"time":...}}
is teacher/executable contract from public requested time, not model output or
self-verified hallucination. Independent effect checker must prove exactly one
new matching alarm and unchanged other state. No credentials in public prompt.
TEST gold not accessible to optimizer/retrieval; no checkpoint selection onTEST.
Record template limits and raw tool/source/data manifests; not a user-chat dataset.

## Single pilot budget locked before model training

Seed260115, rank4/alpha8/dropout0, target_modules q_proj/v_proj in text attention
layers only; don't target vision or lm_head. AdamW lr1e-4, no weight decay,
one pass32steps, batch1, max256 input+target tokens, gradient checkpointing and
no KVcache. Corrected completion-only cross entropy; logits only target+1
positions, exact causal shift checked before backward. No hyperparameter grid.
Stop after first32attempts/20minutes; attempts precharged/no refunds. Max20
heldout generation probes (4DEV+16TEST) at baseline and final, greedy/max64new
tokens; no repeated prompt tuning or selection. Validation NLL all8DEV optional
within forward budget128. Save candidate even if quality fails; label it unqualified.

Check E0/source/dtype/device/adapter trainable count before backward; independent
source check, save/reload fixed logits equality and base weights unchanged.
Reserve512MiB physical GPU at samples, worker memory host<=8GB. On OOM/SDK
training failure stop, save failure receipt; no silent CPU fallback/dtype reduction.
Runtime may be put into explicit sleep only if needed for shared GPU capacity,
after checking no user task/capture, then restore previous intent; do not kill
foreign processes or modify source/config/weights. User foreground wins.

Train and native runtime may coexist only if reserve holds. Adapter output goes
artifacts/<run>/adapter; checkpoint/data/predictions/log loss and resources visible
on localhost. No claims that32steps learn a general skill or retain20stages.
Pilot gate: independently executed repair success improvement>=5pp, lower95%CI>0
and old/general control regression<=3pp with no family>5pp, same input/tool budget.
If retention/general controls absent, NOT production or retention qualified.
Single seed/smallTEST is exploratory; don't claim three-seed P1B conclusion.
