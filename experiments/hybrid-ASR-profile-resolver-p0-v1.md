# ASR profile resolver P0 — mechanical preparation before frozen-source unlock

No changes/imports into activeHybridconfig/backends/qualification/resources until
original554c19terminalaudit. This P0 only adds an unimported profilelibrary and
unit/data-only storage checks, as planned in ASR-profile-integration plan.
No ASR/VLM calls in this preparation, no model/download/training or liveconfig.

Whitelist officialbase/small names and already-pinned revisions/exportSHA/byte
counts. Base: nativeF32export290960250bytes/6a15df05602c21d513a0912bb02725280927ff72a32e150d1a053a2d8ec62cd1;
small:967535936bytes/fcfe5ac3405eb920581f41444238ef9b5d86a6c57356c96cd1e486269c60072e.
Base245/small479tensor readback provenance from existingaudit; do not rerun
conversion or pretend metadataflags alone prove tensorquality.

Resolve fixedrelativepaths under explicitlyowned dataroot, reject symlinkescape,
unknown identity/revision, cpuSmall(unqualified), missing/tamperednativefile or
conflicting native-storage receipt. Correctexportpin establishes exactknownbytes;
qualityqualification remainsfalse/independent, never inferred from checkbox.
Do not rename/replace basefile to impersonate small or download on missingasset.

Tests: explicit field/type checks, profilebytesandmetadata match, wrongSHA/type/
dtype/tensorcount/source or missingreceipt rejected; base/default resolution,
smallGPU resolution, cpuSmall and wrongrevision rejected. Filesystem failures
tested with tinyinvalidfiles; puremetadata validcase is clearly a unitfixture,
not an inference or tensorreadback. A data-only actualknownexports hashcheck may
run once after unitchecks, ≤30seconds, outputs/source/identity archived before
any later runtimeintegration. Source safetensors alreadyindependentlyaudited;
this check does not reload a neuralmodel or make newqualitysamples.

Expose exactfile and sourceidentity to future runtime callers. Keep original
3exportVLM/baseintegrity report unchanged; futureintegration will separately
validate selectedASRprofile and be preregistered after currentfrozenwork stops.
P0library success does not fulfill actualruntimeconfig/device/guard integration.

Shadowintegration preparation, before candidate tests: generate reviewedcopies
of config/backends/qualification/gateway from the exactfrozenSHA without editing
their paths. Import candidateconfig/backend only under isolatedmodule names in
the testprocess; no modelstart/ASRspawn unless fakefixture explicitly intercepts
spawn. Candidateboundarytests inspect actualcommand/result routing with a fake
worker transport, clearlynot actualGPUinference. Actual4ASR/4VLMcallbudget from
integrationplan remainsunused until8hterminal/unlock. Do not claim source being
generated/compiled means currentHermes has changed. Defaultbase+CPU behavior
must remain, smallCPUrejects, smallGPUresourceguard mustnotunloadVLM onpressure.

Before finalshadowtests, require current-callnative log: capture byteoffset
before ownedASRspawn, read only freshbounded≤256KiBlogslice after completion,
confirmftype0 plus requesteduseGPU and actualusingCUDAdevice; GPUrequestflag
alone or stalepriorCUDAlog cannot qualify. CPU requiresuseGPU0/noCUDAusage.
Test this with declared fakeworkertransports and archivedrealsoaklogs(dataonly),
no newASRcalls before8hunlock. Mismatches keep failure/draft, no silentfallback.

Before finalizedshadow tests, replace blockingASRwait with boundedpolling of
the ownedprocess:120sdeadline, monitorcheck, GPUreserve512MiB and foreground
preemption. The caller's existingfinally stops only its ownASRprocess on error;
VLM must remain resident forsmall. Tests use interceptedworker/metadata only;
realprofile/pollinglatency still needs actualintegrationcall after8hunlock.
