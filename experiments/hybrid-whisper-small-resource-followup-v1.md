# Whisper-small resource instrumentation follow-up —14/11/2026

Paired DEV68e3c7 completed64coldprocesscalls on32locked clips; quality/speed
gates passed, ViWER45.95→21.23 and En9.64→6.02. Preserveall outputs/protocol.
Source audit found final sharedGPUgrowth uses lastsample after monitor.close,
not sampledpeak; final PDHavailable=false is shutdown status. Thus original
locked_resource_gate=true is insufficient evidence. Add review receipt, no
rewriting original result/promotion. No model/decoding/quality threshold retune.

Register a resource-only followup before execution. Same nativeFP32 models/
whisper.cppCUDA0/t2/defaultcontext/search, existingVLM16K/nativeprecision keeps
running. One nativeCLI process per model receives all32locked audiofiles as
documented positionalarguments; it remains alive long enough for2sPDH snapshots.
This is sustainedresidency instrumentation, not the per-chunk production mode
or a new quality/speed replicate. Fulltext logs kept; no scoring these repeated
outputs as new test samples or changing the original coldprocess latency.

Use ResourceMonitor.sample/close return correctly. Collect active resource rows
while ownedchildPID exists, uniquePDHcollectiontimes, sampled shared/dedicated
allocation and globalfreeVRAM every≤.25s. Require≥2freshactivePDHreadings/arm
withavailable flags; compare max sampledsharedusage to prelaunchbaseline
(≤256MiBgrowth) and globalfree≥512MiB. Report2sPDH/≤.25sNvidia sampling limits.
Missing/unavailable/onlypost-exit readings cannot pass. Stop ownedASRchild on
resource/integrity/ownerchange, neverstopliveVLM or8hworker.

Max5minutes/2nativebatchcalls, no extra download/privateaudio/learning/LoRA.
Same32datahashes/modelhashes/runtimerevision, no CPUoffload or preprocessing
change.8hsourcefrozen untouched. Publish actualresourceprobe at8775; only a
passedfollowup can close this specific missing resourceevidence. It does not
qualify64newTEST, fullvideoP1 or8hstability of a changed ASRprofile.
