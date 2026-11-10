# Actual Qwen4B reference on48public utterances

No private prompts, tools, retrieval or SCAR-Xproxy. Qwenmemory measured separately aboveSCAR-Xtarget. Exact matching toone style target is not a conversationqualitygrade.

{
  "run": "runs/20261109T162356Z-qwen-g1-state-reference-v1-360a02",
  "data": "data/candidates/20261109T161256Z-antigravity-g1-state-paraphrases-a4d648/reviewed.json",
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "model_sha256": "8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc",
  "decision": "QWEN_REFERENCE_MEASURED",
  "completed": 48,
  "expected": 48,
  "truncated": 17,
  "server_stopped": true,
  "error": null,
  "resources": {
    "peak_rss_bytes": 3196772352,
    "peak_commit_or_vms_bytes": 3881074688,
    "os_process_peak_working_set_bytes": 66080768,
    "os_process_peak_pagefile_commit_bytes": 691290112,
    "samples": 2058,
    "interval_seconds": 0.1,
    "limit_bytes": 4500000000,
    "observed_over_limit": false,
    "elapsed_seconds": 228.08286509999743,
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
  "scarx_uses_qwen_proxy": false,
  "matched_pretraining_or_ram": false,
  "hermes_configuration_changed": false,
  "private_input_sent": false,
  "new_training": false,
  "normal_chat_or_vision_equivalence": false
}
