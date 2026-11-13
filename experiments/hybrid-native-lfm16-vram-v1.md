# Optional native1.6B VLM feasibility —14/11, before download/inference

Qwen2B original and3bufferprofiles failed512MiB mixedreserve; preserve them.
Evaluate ONE alternative frozenVLM, never scale/replace ALIF or attribute a
VLMcapacity change to SNNlearning. Same one-assistantHermes interface. Candidate
does not enter live before qualification. No quantization/pruning/dtype reduction.

Official`LiquidAI/LFM2.5-VL-1.6B`, pinned revision
`919fde3d022e3f90a4716006f993938ee8c2eb97`, originalallBF16 tensors,
source3,193,334,216bytes,SHA256
`7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d`.
Read official LFMOpenLicense1.0; localpersonalresearch, no weight redistribution.
Keep LICENSE/NOTICE with localexports. Publicmodelcard targets8languages but not
Vietnamese; do not assume Viquality. Reject SmolVLM2-2.2B originalF32(~8.99GB)
without download, because dtype lowering is prohibited and storage exceedsVRAM.

Use existing pinnedb12360converter and CUDAworker, same16Kcontext/BF16KV and
ALIF/WhisperCUDA. Nativeexportdtype/coverage audit and originalCPUTransformers
reference3probabilitycases (helloVi/En,7+8); allsourceweights exact, no remote
Pythonmodelcode. Match top1/Δprob<=.03. No reusingQwenreference/tokenizer.

Lock12closedengineeringcases before inference: mathVi/En, directViinstruction,
finalsettingVi/En,distractorlatestsetting, strictJSON, oldlongcompactcorrection,
missing-evidenceabstention, colorsmall/1080image, two-eventtemporalquestion.
Sameprompts/inputbytes/budget96outputs/temp0 on current0.8B andcandidate, no
testfeedback/retry-until-pass. These are engineeringDEV, not formalhybridP1.
Require no fewer validanswers than0.8B overall or onVi subset and no unsafe
toolactivation; explicittoolcall transport tested separately in read-onlysandbox.

Resourcegate unchanged: sampledglobalfree>=512MiB, noOOM/crash/large sharedGPU
spill, warmTTFTp95<=5s; actualGPUASR plusVRAMguard/componentrelease as needed,
same sourceprecision/context.3repeatmixedshortcycles and fullstate persistence.
If preload fails/parity fails stop; do not silently fall backCPU or quantize.
8hsoak and generalquality remain separate/open. Maxtechnicalprobe15minutes,
at most1candidateconfiguration (defaultbatch128/64/FAauto/graphon). No further
search in this registration after seeing results. Allfailedoutputs retained.

Technicalreference repair before any candidate inference: initialreference falsely
classified vision keysunused because Transformers5.22 flattened the loader prefix
`model.vision_tower.vision_model.` to`model.vision_tower.`. Initialmismatchlist
empty but unusedvisionkeys causedP0FAIL. Preserve reference.json and original
sourcebytes. V2 permits ONLY this deterministic prefixrename and checks original
dtype/shape/everytensorbyte; no value rewrite/ignoredtensor exception. Write a
new reference-keymap-v2.json; if any remaining tensor/byte mismatch, stop.
