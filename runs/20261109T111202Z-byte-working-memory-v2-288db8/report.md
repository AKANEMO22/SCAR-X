# Held-neural working-memory byte trial

Same48ALIF population with48held activation values and extra readout synapses; no hidden updates. A snapshot is not learned long-term memory.

{
  "run": "runs/20261109T111202Z-byte-working-memory-v2-288db8",
  "summary": {
    "no_snapshot": {
      "development_exact": 0.0,
      "train_exact": 0.0,
      "byte_accuracy": 0.23006535947712417,
      "zero_snapshot_exact": 0.0,
      "swapped_snapshot_exact": 0.0,
      "gain_vs_no_snapshot": 0.0
    },
    "snapshot_raw": {
      "development_exact": 0.0,
      "train_exact": 0.0,
      "byte_accuracy": 0.19869281045751633,
      "zero_snapshot_exact": 0.0,
      "swapped_snapshot_exact": 0.0,
      "gain_vs_no_snapshot": 0.0
    },
    "snapshot_normalized": {
      "development_exact": 0.0,
      "train_exact": 0.0,
      "byte_accuracy": 0.10980392156862746,
      "zero_snapshot_exact": 0.0,
      "swapped_snapshot_exact": 0.0,
      "gain_vs_no_snapshot": 0.0
    }
  },
  "resources": {
    "peak_rss_bytes": 101072896,
    "peak_commit_or_vms_bytes": 131952640,
    "os_process_peak_working_set_bytes": 54304768,
    "os_process_peak_pagefile_commit_bytes": 76046336,
    "samples": 603,
    "interval_seconds": 0.25,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 151.8515658000033,
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
  "deployed": false,
  "normal_chat_vision_proven": false,
  "retention_proven": false,
  "hidden_representation_gain": false,
  "useful_screen": false,
  "decision": "STOP_BYTE_DIALOGUE_FAMILY_AFTER_V2"
}
