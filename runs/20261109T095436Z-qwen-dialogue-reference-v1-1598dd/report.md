# Qwen dialogue reference

Same24public context prefixes, no added system prompt/tools/retrieval. Exact matching to one stylistic target is not a sufficient conversational quality metric.

{
  "run": "runs/20261109T095436Z-qwen-dialogue-reference-v1-1598dd",
  "source_run": "runs/20261109T094940Z-event-dialogue-prototype-v1-285565",
  "data": "data/candidates/antigravity-events-curated-20261109T094142Z",
  "completed": 24,
  "expected": 24,
  "exact_to_single_target": 0.0,
  "truncated": 11,
  "error": null,
  "server_stopped": true,
  "resources": {
    "peak_rss_bytes": 3070279680,
    "peak_commit_or_vms_bytes": 3753009152,
    "os_process_peak_working_set_bytes": 44339200,
    "os_process_peak_pagefile_commit_bytes": 670470144,
    "samples": 673,
    "interval_seconds": 0.1,
    "limit_bytes": 4500000000,
    "observed_over_limit": false,
    "elapsed_seconds": 75.00418150000041,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  },
  "cap": {
    "enforced": true,
    "limit_bytes": 4500000000,
    "kind": "Windows per-process committed memory; shared pages monitored separately"
  },
  "matched_memory_or_pretraining": false,
  "rubric_review": "pending; source-known exploratory only",
  "hermes_configuration_changed": false,
  "scarx_uses_qwen_proxy": false,
  "general_chat_equivalence_proven": false
}
