# Actual live inference on48reviewed public utterances

Raw private-model answers shown locally/ontrace, no target labels learned; archivedQwen matched only by exactpublicprompt. No new capability certification.

{
  "run": "runs/20261109T171819Z-g1-live-new-prompts-observation-a9c6a1",
  "decision": "ACTUAL_LIVE_INFERENCE_OBSERVED_ONLY",
  "completed": 48,
  "expected": 48,
  "updates_before": 241363,
  "updates_after": 241363,
  "new_training": false,
  "live_model_id": "scarx-generation1",
  "cores_loaded": 1,
  "normal_chat_or_vision_proven": false,
  "input_scope": "SCAR-X continuous recurrent state, Qwen archived single-prompt context; not a matched-state/training benchmark",
  "private_raw_answers_saved_locally": true,
  "resources": {
    "peak_rss_bytes": 70524928,
    "peak_commit_or_vms_bytes": 43991040,
    "os_process_peak_working_set_bytes": 63762432,
    "os_process_peak_pagefile_commit_bytes": 47788032,
    "samples": 113,
    "interval_seconds": 0.5,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 6.300683599998592,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  }
}
