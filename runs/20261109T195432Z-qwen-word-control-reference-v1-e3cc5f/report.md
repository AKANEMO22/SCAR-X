# Actual Qwen reference on47missing publicprompts

Paired with archived isolatedALIF-BPTT outputs, not mainSCAR-X. Same publicsingletext, different training/RAM/decoderbudget. Rawresponses not truthlabels, truncations retained; ownedserver stopped.

{
  "run": "runs/20261109T195432Z-qwen-word-control-reference-v1-e3cc5f",
  "scarx_control_run": "runs/20261109T192850Z-g1-word-learnability-control-v1-d74480",
  "decision": "QWEN_REFERENCE_MEASURED",
  "completed": 47,
  "expected": 47,
  "truncated": 24,
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "model_sha256": "8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc",
  "max_tokens": 64,
  "context": 1024,
  "temperature": 0,
  "seed": 919,
  "server_stopped": true,
  "error": null,
  "resources": {
    "peak_rss_bytes": 3236315136,
    "peak_commit_or_vms_bytes": 3281039360,
    "os_process_peak_working_set_bytes": 64733184,
    "os_process_peak_pagefile_commit_bytes": 50561024,
    "samples": 2345,
    "interval_seconds": 0.1,
    "limit_bytes": 4500000000,
    "observed_over_limit": false,
    "elapsed_seconds": 257.0255285000021,
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
  "new_training": false,
  "scarx_uses_qwen_proxy": false,
  "hermes_configuration_changed": false,
  "private_input_sent": false,
  "normal_chat_or_vision_equivalence": false
}
