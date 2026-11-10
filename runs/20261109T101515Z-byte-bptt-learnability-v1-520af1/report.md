# Isolated byte learnability controls

These weights cannot be deployed as SCAR-X local-learning evidence. Same forward model/input but different optimizer, temporal gradients and step aggregation.

{
  "run": "runs/20261109T101515Z-byte-bptt-learnability-v1-520af1",
  "resources": {
    "peak_rss_bytes": 737263616,
    "peak_commit_or_vms_bytes": 1256800256,
    "os_process_peak_working_set_bytes": 738852864,
    "os_process_peak_pagefile_commit_bytes": 1257553920,
    "samples": 1416,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 160.9708873999989,
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
  "control_only": true,
  "cuda_allocated": false,
  "weights_imported_into_scarx": false,
  "main_local_learning_success": false,
  "normal_chat_or_vision_proven": false,
  "scope": "Different optimizer/all-synapse backward vs local per-byte updates, not matched compute/updates; negative result inconclusive under limited budget"
}
