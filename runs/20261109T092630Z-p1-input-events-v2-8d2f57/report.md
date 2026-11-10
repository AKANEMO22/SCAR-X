# Ordered/RGB input events

{
  "run": "runs/20261109T092630Z-p1-input-events-v2-8d2f57",
  "pairs": 384,
  "ordered_collisions": 0,
  "pooled_collisions": 384,
  "minimum_sequence_difference": 1.7853569984436035,
  "fresh_confirmation_collisions": 0,
  "fresh_confirmation_pairs": 256,
  "color_events_equal": false,
  "color_l2_difference": 9.182515144348145,
  "same_channel_layout_as_g1": true,
  "input_channels": 222,
  "no_target_features": true,
  "text_maximum_events": 256,
  "image_events": 18,
  "resources": {
    "peak_rss_bytes": 43388928,
    "peak_commit_or_vms_bytes": 28864512,
    "os_process_peak_working_set_bytes": 44302336,
    "os_process_peak_pagefile_commit_bytes": 29593600,
    "samples": 3,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 0.2941852000003564,
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
  "training_updates": 0,
  "normal_chat_or_vision_proven": false,
  "interpretation": "Chronological input preserves the tested order/color distinctions. The learner still must use them; pooled decoding would discard order again."
}

This fixes a tested information-loss precondition in an isolated frontend, not neural task learning. No live weights changed.
