# Native FP32 Whisper-small candidate —14/11/2026 before download/inference

Reason: VLM2B failed physicalVRAM reserve; nativeLFM1.6B failed locked quality.
Current Whisper-base Vi transcript had68.42%WER on one engineering utterance.
Test whether available discreteGPUVRAM improves perception before any live swap.
This is a frozenASR candidate, not SNNsize/training/LLM-LoRA or hiddengain.

Source official openai/whisper-small, revision
973afd24965f72e36ca33b3055d56a652f456b4d.
model.safetensors966,995,080bytes, SHA256
1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b.
Require originalallFP32 tensor inventory. Same existing direct F32 GGML writer
in scripts/hybrid_convert.py, everytensor exactreadback and no unused/missing/
renamed-collision tensor. Sourcecard/license metadata retained locally; do not
redistributeweights. Same pinned whisper.cppb5454 CUDA12.4 worker, no quantization,
pruning, FP16conversion or CPU fallback. Source/convert areCPU I/O, not serving.

Lock before inference:32new speech samples,16Vi/16En from FLEURS publicdev,
CC-BY4.0, separate author-provided transcript. Stream officiallanguagearchive;
select first16eligible audio/dev WAVs≤2MB, matching dev.tsv entry, unique normalized
transcript, excluding priorengineering14135486895615497173.wav. No modelscore
selection; lock paths/reference/audiohash before nativecalls. Record durations,
gender if available and attribution. Existingknown engineeringclip reported
separately, never counted newheldoutcase. No claim FLEURS absent from pretraining.

Compare same32samples once per model in counterbalanced indexorder, language auto
(same currentpipeline), t2, defaultbeam/search/sourceprecision. No utteranceprompt,
forcedlanguage, retryuntilpass or decodingretune. Store full transcripts,
autodetectedlanguage and independent NFCcasefold/regex-word edit metrics. Primary
microWER perlanguage; pairedutterancebootstrap95%CI (not token/seed replicates).
Whisper deterministic frozeninference does not supply3independent learningseeds.

Technical selection gate: source/readbackPASS, noOOM/crash, globalphysicalfree
≥512MiB sampled while both VLM andASR active; no sharedGPUspike>256MiB vsbaseline.
Do not closeotherapps, unloadliveVLM, change16K/precision or hide resourcefailures.
Quality: ViWER≥10%relative reduction and95%upper(deltaWERsmall-base)<0; EnpointWER
notworseby>2pp. RTFp95≤1.0 and no more than2×baselinemedianlatency. If candidate
fails stop/retainbase.32samples areDEV, not fullvideoP1/productionconfirmation;
no livepromotion onstorageprobe alone. Any nativeCPUreference must be explicitly
separateevaluationcontrol, never silentGPUmodefallback.

Maxdownload1.2GBmodel plus boundedstreamaudio≤150MB/network≤500MB perlanguage/
10minutes; same sourceURL/TSV+extracthash. Totalmeasure≤20minutes,≤64newnative
calls + existingengineering2calls, stop onintegrity/resourceerror. During active
8hsharedworker test, leave its frozenfiles/workers/state untouched; candidateASR
ownedchild sharesGPU only if reservepreflight permits. Yield beforelaunch when
Hermes isserving. No newHTTPservice, daemon, gateway, tokenizer/model cache inlive.

Artifacts unique runs and ignoredlocalcandidate dirs. Publish actual downloads,
references/transcripts/WER/RAM/VRAM/latency to8775 actororchestrator; not liveSNN
learning. Lossy or resourcefailed candidates never get imported into main/Hermes.
