# Independent data audit of mixed endurance —14/11 before derived metrics

No new inference, training, model, worker restart, threshold or active-source
change. Audit original8h554c19 and smoke/failed runs only, using owned local
trace/SQLite/checkpoint/process state. New source files must not overlap the
active manifest. Results are uniquely named snapshots; do not overwrite any
heartbeat/result or interpret unfinished observation as a completed PASS.

Recompute original engineering gates directly (do not call its qualifies):
elapsed≥28800,≥80completedvideo/QAcycles,≥8checkpointrollbackchecks,updates>0,
updates>replay_seen,devicegpu,criticalfailures0,noactivecapture/cleanupfailure,
same nativeworker identity, frozen source integrity and sampled512MiBreserve.
Require originalfinalreceipt plus actualworker terminal before a final audit
PASS; source/receipt absence or monitoring timeout cannot trigger job restart.
For a liveprocess produce PRELIMINARY_ONLY regardless counts/time.

Read a transaction-bounded event snapshot from liveobserver DB filtered by the
exactrun_id and owned shared_endurance event kinds; archive those events/hash.
Separately read privateDB in read-only transaction. Cross-database snapshots
are not atomic: provisional count differences at a liveboundary are reported,
not retroactively rewritten or called a failure. At terminal compare complete
receipt/trace/private row counts and evidence/stream/checkpoint consistency.

Report video wall/capture length separately: sum completed stream durations,
unique original source path/hash, repeatedstreamcount, analyzed/skippedranges,
capture→answer latency percycle and any trend (descriptive only). One21sfixture
repeated is not8hcontinuous video,1–2hdiverseP1,semanticquality or lifelongskill.
Leave primaryanswer_quality_scoredfalse and preserve actualrawanswers.

Resources: derive peaks and sampled minima from rawheartbeats, do not add RSS
and privatecommit as independent memory. Separate physicalglobalfreeVRAM from
owned allocation/sharedVRAM and hostGPUmodeuncapped budget. Sampling can miss
short peaks; cumulative reportedpeaks may cover periods between heartbeats,
but audit cannot infer unobserved physicalminima. Never call partial process
coverage whole-system peak. Compare current validatedHermesgatewayroot against
lastresource sampled PIDset; if a reload changed coverage record the gap, not
a fabricated historical correction or a whole-CPUbudget PASS.

Private state: count actual local_learning_update, criticalquantum/ASRfailures,
arithmetic goals/decisions and checkpointpins. Parse saved checkpointmetadata
data-only, hash/replay/RNG/updates constraints; no neural forward or restoring
state into live. One arithmetic family/20goals does not become20skills.

Budget60seconds/snapshot,≤5000observerevents/≤20MBarchivedJSON,≤1000private
stream/evidence rows per summary or a clear count-only truncation notice. One
snapshot per meaningful milestone/terminal, not high-frequency query polling.
Use finite/typed metric checks; forged8hcount/negative/nonfinite numbers must
not pass unit regression. Publish measuredcoverage/resources on8775 as auditor
work, never as a newly completed training job or invented model thought.
