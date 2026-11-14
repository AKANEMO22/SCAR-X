# Whisper-small nativeFP32 paced integration soak — before execution,14/11/2026

Goal: qualify repeatedcandidateASR activation over time with the existing local
nativeVLM stillresident/usable. Prior32speechDEV/resources showed componentgain;
this is engineering durability, not new quality/learning samples or livepromotion.
No new localrule, LoRA, backward, modelprecisionchange, gateway or paidservice.

Same officialsmallF32export SHAfcfe5ac3405eb920581f41444238ef9b5d86a6c57356c96cd1e486269c60072e,
whisper.cppb5454CUDA12.4 executablehash from68e3c7; originalsmalltensor/source
parity remains prerequisite. Same32ownedFLEURSDEVinputfiles/lockedhashes from
ddf1fd, fixedsourceorder, alternating repeats over345.76seconds of uniqueaudio.
Do not call repetitions new heldout samples, language seeds or retention proof.

Nativecommand same existing CLI:modelsmallF32,languageauto,t2,CUDAdevice0,JSONfull.
One ownedprocess perchunk, no ASRHTTPserver or streamingmodelweightpaging.
Require actuallog usingCUDA0/ftype0, source/nativehash stable before/after.
Keep one existing Qwen0.8B/16K/nativeGPU worker identity; never unload/kill it.
No image/video recapture, no private conversation/audio goes to any service.

Two horizons:30s mechanicalpreflight, then1800s pacedsoak only if same-source
preflight qualifies. Pace nextstart by cumulativeoriginalaudio durations, not
run as fast as possible. Report latency/RTF and any lag; this is repeated
utteranceASR, not continuous1–2hvideo or P3fullassistantqualification.
Gatepreflight:≥1completeCUDAFP32call/noerrors/reserve≥512MiB.
Gate30min:actualelapsed≥1800s,≥80completecalls,≥1000audioseconds processed,
RTFp95≤1,noOOM/error/CPUfallback,nativeidentity unchanged, sampledfreeVRAM≥512MiB,
and correctedexplicit+dynamicclient root coverage present. GPUhostRAMuncapped;
RSS/commit/shared/dedicated report separately, no historicalpeakbackfill.

Budget≤256attemptedASRprocesses TOTAL including preflight/reruns; local ledger
charges before spawn, atomicsave, no reset by newrunID. No retry-to-pass. Errors,
source/device/integrity/reserve failures stop ownedASR and preserve partialrun.
No completedcall may be silently dropped from transcript/latency tables.

Foreground production has priority: wait before newchunk while serving; if
foreground begins during an ASRcall, terminate only that ownedcall, record the
preemption and consume its attempt budget. Elapsedclock continues; insufficient
coverage is notPASS. Otherapps untouched. Physicalglobalfree samples at most
every~250ms while active plus heartbeats; actualtimestamps/resolution retained,
not a guarantee of unobserved peaks. No nativegeneration just to fillwork.

Use uninstalled correctedResourceMonitor overlay8774db6c6f91adb8bcb2f990341aa086eebe88f57b047521837603829e0af2c8,
explicitliveSCAR-Xowner (and validated8hlabroot if present) plus dynamicHermes.
Monitor root union means group measurements, not isolatedsmall allocation or
proof of RAM reduction. P0root membership0c0835 passed. Active8hsource remains
byte-exact; no resources.py install until its finalaudit. ASRsoak is external
mixedGPUworkload; record coexistence with base8h, do not hide that confound.

Write-once unique results/raw percall/model/data/sourcehashes. Heartbeats may
advance; job handle/PID/create-time must be verified. Detached hidden worker
for30min, no automaticrestart if observationtimeout. On8775publish actualASR
text, latency, attempted/completed counts, measuredRAM/VRAM and ownership;
actororchestrator/ASRcomponent, no liveALIF update or fake neuralthought.

Even a completed engineeringsoak cannot promoteASR toHermes without verifying
the real configuredmodel identity/selection/integrity path. That sourcechange
waits for frozen8hcompletion. Whole-system/video/skill quality gates remain.
