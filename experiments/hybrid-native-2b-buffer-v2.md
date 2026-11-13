# Native2B buffer/scheduling DEV —14/11/2026, locked before tests

Follow-up to failed mixedheadroom probe77a5fb. Keep original failure,512MiB
reserve,16Kcontext,all-neuralCUDA and originalBF16/F32weights/BF16KV. No model
quantization/pruning, neuron growth, CPU weight offload or new learning rule.
Test3fixed runtime profiles, not a hyperparameter search after results:

| Profile | logicalbatch | microbatch | FlashAttention | CUDAgraphs |
|---|---:|---:|---|---|
|baseline|128|64|auto|on|
|small_flash|64|32|on|on|
|small_no_graph|64|32|on|off|

Use pinnedllama.cppb12360. Graphs flag verified in localpinnedggml-cuda/graph.cuh;
only childprocess environment changes. FA/kernel choices may change floating
reduction order; require existing3nativeprobability cases top1same/Δprob<=.03.
Storage/config/modelsourceSHA stay unchanged. No precision reduction.

Same realGateway/ALIFH128 and originalWhisperFP32CUDA, own isolateddatastore,
all8775mutations nonce-gated. All3profiles independent processes, owner-checked
normalservice stopped/restored by outer shell, no peer/model deletion.

Lock6questions:17+25,currentportshort,longprojectportupdate(known distractors),
smallimagecirclecolor,1920x1080syntheticcirclecolor,strictschemaJSONfinalport.
Three repeats each, nativeparity first, output96tokens, temperature0. Same input
bytes and prompts across profiles. No retry-until-pass, no testfeedbacklabels.
Print actualresponses/measurements automatically atlocalhost. Max12minutes total,
90s/request. A profile failing toload/parity ends that arm; preserve failures.

Nvidiasampling100ms across boot/text/images/ASR; ownedRSS/commit/dedicated/shared
VRAM separately. Primaryresourcegate: minimumglobalfree>=512MiB during active
work, noOOM/workercrash/no extra sharedGPUspike>256MiB from post-load baseline.
BeforeASR reserve conservative538MiB scratch+512MiBheadroom; unload entire VLM
when needed, keep ALIF/evidence. ASR andVLM computations stayCUDA; measure cold
reload latency and backlog instead of pretending concurrent residency.

Candidate technicalpromotion requires parity,sourceintegrity,6caseoutputchecks,
minreserve, p95warmTTFT<=5s and scheduledASR successful. Existingformalquality/
retention and8hsoak gates remain unpassed until measured. If none qualifies,
keep current0.8Blive; report2B constrained, do not waive memorygate.
This is buffer/schedulingengineering, not proof of SNNadvantage or generalAI.

Preflight source audit found Gateway dropped clientresponse_format on native
calls. Forward it unchanged on streaming and nonstreaming paths; prove behavior
red on originalsource/green on repair. StrictJSON case uses an explicit schema,
not a post-hoc parser that accepts Markdown. Oldprobe strictJSON failure stays.
An arm with <512MiBfree immediately after load stops before potentially larger
imageallocations, recorded as resourceFAIL rather than a quality score.
