# Qwen4B numeric reference

Frozen quantized model on unsigned arithmetic only. Same raw question/format instruction is available for secondary SCAR-X probes; no claim about general intelligence or Qwen online learning.

{
  "run": "runs/20261109T072118Z-qwen4b-numeric-reference-bda43a",
  "reference_download": {
    "base_model": "Qwen/Qwen3-4B-Instruct-2507",
    "quantizer": "lmstudio-community",
    "revision": "4edb920b6f14e3b9284d4502a6485103d72cde05",
    "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "expected_bytes": 2497280448,
    "expected_sha256": "8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc",
    "purpose": "isolated frozen comparison only; no imported weights in SCAR-X",
    "available": true,
    "sha256_verified": true,
    "bytes": 2497280448,
    "seconds": 82.57900000000154
  },
  "exact": 1.0,
  "curriculum_exact": 1.0,
  "unseen_exact": 1.0,
  "completed": 160,
  "expected": 160,
  "truncated": 0,
  "error": null,
  "server_stopped": true,
  "resources": {
    "peak_rss_bytes": 2923950080,
    "peak_commit_or_vms_bytes": 3604324352,
    "os_process_peak_working_set_bytes": 45826048,
    "os_process_peak_pagefile_commit_bytes": 670593024,
    "samples": 1001,
    "interval_seconds": 0.1,
    "limit_bytes": 4500000000,
    "observed_over_limit": false,
    "elapsed_seconds": 96.11857099999906,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  },
  "hard_cap": {
    "enforced": true,
    "limit_bytes": 4500000000,
    "kind": "Windows per-process committed memory; shared pages monitored separately"
  },
  "matched_scarx_memory_budget": false,
  "elapsed_seconds": 96.11000000000058,
  "general_knowledge_equivalence": false,
  "reference_online_learning_tested": false,
  "hermes_config_changed": false
}
