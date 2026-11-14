# Startup timeline and storage reads — retrospective/source diagnostic

The existing57a681 activationevents have already been inspected to locate
the criticalpath; this is posthoc descriptive evidence, not a preregistered
coldstart speedbenchmark. One actual gateway/native owner remains live; no
restart/startup/nativeinference/ASR/learning during this audit. Active32cd13
source/worker untouched; all production neural placements remainGPU.

Timeline: use knownactivationstartupcursor, first subsequentnativegate and
native_storage_verified/neural_execution_device/worker_started/search_ready/
threeparitycases between them. Verify actualPID/create-time/executable paths
and startedworkerPID. Publish reducedtechnicalevents only, no workerargv/token
or privateconversation. Intervals are walltimestampdifferences and mixedstage
bounds, not exclusive CPUtime; no separaration of VLMload from SearXNG without
an intermediatehealthready marker. The whole21.25s activation also includes
stop/sourcebackup/restore/probe; do not label it modelstartuptime.

Hash measurement in a separate localPythonprocess, original files read only,
same selected smallGPUconfig. Real verify_native_storage hashes baseline3files
plus selectedASR; a fresh realNativeBackends.ensure_asr_storage hashes selected
small again; secondensure call should use its in-processstat-signaturecache.
Never call start/generate/transcribe/to_device. Reject if unexpected method
requires modelworker/storeactions. No GPUallocation, modelconstructor or
liveinstance mutation. Hashlib instrumentation wraps realSHA256/update only
in this isolatedprocess and counts bytes/object without changing digest.

Exactlythree timedstages, max30s whole audit; no repetitions/cacheflush/rerun.
Cache state is uncontrolled while servicealreadyruns, not strictcoldI/O or
performanceconfirmation. Record hashbytes/count/time, sampled ownRSS and
processCPU; don't attribute a mixed historicalstage entirely to hashing or
claim actual userspeedup. Any futurecachehandoff must retain realintegrity
verification before ready and filechange invalidation, never trust a verified
flag/filemetadata alone. No optimization is applied by this diagnostic.

Source/run manifest before timedhashes; realsha restored before trace/storage
publication even on failure. Verify expectedbytes using pinnedfilemetadata,
all returned proofs/pins and finalfile/sourceidentity. Preserve partialfailure,
stop on mismatch/time/source/processerrors. Dataaudit recomputes timeline and
bytecounts from receipts/cases; no extra timedstages or neuralcalls. Source8h,
main/protectedR13, privatecheckpoint, configuration and noquant/noLoRA unchanged.
