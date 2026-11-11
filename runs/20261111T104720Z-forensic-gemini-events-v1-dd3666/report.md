# Gemini raw event follow-up

{
  "decision": "RAW_EVENT_AUDIT_FAILED",
  "rows_checked": 87552,
  "wrong_labels": 0,
  "neural_forward_calls": 0,
  "learning_updates": 0,
  "fresh_seeds_used": false,
  "resources": {
    "peak_rss_bytes": 53854208,
    "peak_commit_or_vms_bytes": 39362560,
    "os_process_peak_working_set_bytes": 55398400,
    "os_process_peak_pagefile_commit_bytes": 40931328,
    "samples": 249,
    "interval_seconds": 0.02,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 5.263090899999952,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  },
  "cap": {
    "enforced": true,
    "limit_bytes": 2000000000,
    "kind": "Windows per-process committed memory; shared pages monitored separately"
  },
  "error": "EventOracleError: row680: marker count ambiguous: writes=2, query=2",
  "scope": "Latent generated-event/label consistency. Seeded noise removed only by evaluator, never fed to model; not a guarantee every noisy observation has an unambiguous threshold decoding.",
  "delivered_24453bd_package_overwritten": false
}
