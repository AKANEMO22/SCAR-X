# ActualQwen4B64publicpromptreference

Rawoutputs/truncationretained; not traininglabels, exactstylegrade or equalcost. Ownedserver stopped.

{
  "run": "runs/20261110T062343Z-qwen-direct-word-reference-v1-5e4d7d",
  "source_run": "runs/20261110T062226Z-direct-word-bridge-v1-a6a908",
  "decision": "QWEN_DIRECT_WORD_REFERENCE_MEASURED",
  "completed": 64,
  "expected": 64,
  "truncated": 14,
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "model_sha256": "8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc",
  "temperature": 0,
  "seed": 919,
  "context": 1024,
  "max_tokens": 64,
  "server_stopped": true,
  "error": null,
  "resources": {
    "peak_rss_bytes": 3362549760,
    "peak_commit_or_vms_bytes": 4046532608,
    "os_process_peak_working_set_bytes": 80990208,
    "os_process_peak_pagefile_commit_bytes": 706678784,
    "samples": 2910,
    "interval_seconds": 0.1,
    "limit_bytes": 4500000000,
    "observed_over_limit": false,
    "elapsed_seconds": 320.98723840000457,
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
  "Qwen_outputs_used_as_training_labels": false,
  "hermes_configuration_changed": false,
  "normal_chat_or_vision_equivalence": false,
  "private_input_sent": false
}
