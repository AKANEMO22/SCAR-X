# Adaptive generation engineering

{
  "measurements": [
    {
      "allocated": 128,
      "computed": 48,
      "seconds_per_1000": [
        0.04134140000132902,
        0.042293300000892486,
        0.052985500000431784
      ],
      "median_seconds": 0.042293300000892486
    },
    {
      "allocated": 128,
      "computed": 128,
      "seconds_per_1000": [
        0.05259300000034273,
        0.05743359999905806,
        0.04583889999958046
      ],
      "median_seconds": 0.05259300000034273
    }
  ],
  "same_core_identity": true,
  "maximum_allocated": 128,
  "dormancy_preserves_weights": true,
  "restart_continuation_equal": true,
  "new_token_training_point": {
    "prompt": "1+1",
    "target": "2",
    "verifier": "integer arithmetic 1+1, expected2",
    "before": "Có. học. học.",
    "after": "2",
    "weights_changed": true,
    "generalization_proof": false
  },
  "resources": {
    "peak_rss_bytes": 42713088,
    "peak_commit_or_vms_bytes": 28372992,
    "os_process_peak_working_set_bytes": 43720704,
    "os_process_peak_pagefile_commit_bytes": 29159424,
    "samples": 10,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 0.4367686000005051,
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
  "capacity_quality_advantage_proven": false
}

Timings include Python selection overhead; no energy or general intelligence claim. Learning1+1 is a training-point check, not arithmeticgeneralization.
