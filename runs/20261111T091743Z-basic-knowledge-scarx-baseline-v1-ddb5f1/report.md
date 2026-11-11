# Public G1 development knowledge baseline

{
  "decision": "NOT_PRODUCTION_READY_BASELINE_ONLY",
  "summary": {
    "known_questions": 64,
    "unknown_questions": 32,
    "known_exact_accuracy": 0.0,
    "known_typed_semantic_accuracy": 0.0,
    "unknown_registered_abstention_rate": 0.0,
    "generalization_gate": "NOT_MEASURED_WITH_REQUIRED_TRAINED_CONTROLS",
    "retention_gate": "NOT_MEASURED",
    "causal_grounding_gate": "NOT_MEASURED",
    "production_ready": false
  },
  "checkpoint": {
    "format": "scarx-generation1-release",
    "model_id": "scarx-generation1",
    "checkpoint": "artifacts/generation1/20261109T000625Z-generation1-joint-bd6c92/seed-17.checkpoint.zip",
    "sha256": "0c1e8bab2c4ca2ec10578bd4ee0e745366ce1f0b483b4b72b418e09fc5fe10a1",
    "run": "runs/20261109T000625Z-generation1-joint-bd6c92",
    "seed": 17,
    "hidden": 48,
    "shared_recurrent_core": true,
    "general_chat": false,
    "general_vision": false,
    "lifelong_gates_passed": false
  },
  "resources": {
    "peak_rss_bytes": 48549888,
    "peak_commit_or_vms_bytes": 33271808,
    "os_process_peak_working_set_bytes": 49598464,
    "os_process_peak_pagefile_commit_bytes": 34103296,
    "samples": 159,
    "interval_seconds": 0.02,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 2.0626718000021356,
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
  "model_unchanged": true,
  "source_sha256": "13bb051467b27caa10bf8d16039d52fa830baf77a5e87bd63e3520d8b96ed062",
  "new_learning_updates": 0,
  "final_test_opened": false,
  "live_state_read_or_written": false,
  "qwen_used_as_labels_or_proxy": false,
  "error": null,
  "seconds": 2.0755500000013853
}
