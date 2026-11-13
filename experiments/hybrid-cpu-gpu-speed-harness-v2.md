# Speed harness lifecycle repair v2

The measured18requests in1ab607 completed; all raw answers/timings and source
hashes are retained. V1 assigned the benchmark parent to a lifecycle-only
KILL_ON_JOB_CLOSE job. Its finally block restarted the real gateway as a child;
when the runner exited, that job killed the restored gateway too. This is a
restore failure, not a speed/accuracy measurement failure. Append a recovery
receipt after starting the service from the outside parent. Do not alter old
receipt's transient "started" response or label it a successful persistent restore.

V2 keeps explicit child termination/reaping and sampled50msCPUguard, without
assigning a job to the mixed-mode benchmark parent. Serving continues to use
its own real lifecycle job, with CPUmemorylimit or explicit uncappedGPUmode.
No repeat18request run or changed speed thresholds is justified by this fix.
