# Independent ASR paced-soak receipt audit — before derived conclusions

Read only existing30s/1800s candidate runs and chargedbudgetledger. No new
ASR/VLMcall, data generation, training, threshold, source edit or promotion.
Verify actualworker PID/create-time and absence only via recorded handle; a
finalreceipt while workerlive is pendingexit, not a terminalengineeringPASS.

Recompute originalmechanicalgates:30s≥1completecall;1800s≥80calls,≥1000processed
audioseconds,RTFp95≤1,elapsedhorizonmet,noerrors,activeCUDAFP32logs for every
call,physicalfreeVRAMsamples≥512MiB,rootcoverage and unchangednativeidentity.
Require at least1activeASRfreeVRAMsample to avoid interpreting idleonlyresources
as mixedworkload qualification. Source/nativeexport/executable/corpus/overlay
hashes and aggregateattempts≤256 must match. Failed67183e chargedattempt1 is
retained despite zeroASRspawn, not silently refunded or recast as success.

Rebuild transcript text/digest and per-callRTF from storedraw/seconds/duration,
count actualuniqueattempts and first/lastsource-order sequence. Skipindices
only if recordedpreemptions account for them. Compare percallJSON and rawlog;
outputdirectoryrawWhisperJSON may remainlocal but copiedrunrecords must suffice.
Never derive extraindependentqualitysamples or CIs from repeatedaudio/calls.
Report uniqueIDs/audioseconds separately from sumrepeatedaudioseconds.

Read exactrunobserverheartbeats/transcripts in one readonlytransaction, archive
with row/bytebound and compare counts. DynamicHermesroots can change: compare
recordedsample memberships to samebeatreportedroots, not currentrootretroactively.
Finalresources cumulativeRSS/commit/shared/dedicated reported separately;
physicalfree min from active+heartbeat rawsamples, not post-exitPDH alone.
SourcePVLM/8hlab coexistence remainsmixedworkloadconfound; noclaimisolatedASRcost.

Report first/last20callmedians, repeatedtranscriptvariants per32sourceIDs, not
statisticalproof of leak/drift. FrozenASR may vary without implying learning.
No language/QA/retention/proactiveSNNclaim. AnengineeringPASS only qualifies
this componentprofile, not livebackendmodelselection or full8hproduction.

≤30secondsaudit,≤5000observerevents/20MBJSON. Write-once supplemental audit
inside completedrun; originalmanifest/result/heartbeat untouched. Failed or
live snapshots cannot be labeledPASS. Dataonly tests must reject NaN/negative/
boolmetrics and idleonlyVRAMsamples. Publish actual summary8775 asorchestrator.
