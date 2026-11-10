# Event dialogue end-to-end prototype

One48-ALIF core, real word decoding from ordered events, raw/normalized/hidden controls. No serving weights replaced; no natural-photo skill claim.

{
  "run": "runs/20261109T094940Z-event-dialogue-prototype-v1-285565",
  "data": "data/candidates/antigravity-events-curated-20261109T094142Z",
  "training_responses": 96,
  "development_responses": 24,
  "training_vocabulary": 166,
  "effects": {
    "normalization": {
      "mean": 0.0,
      "lower95": 0.0,
      "upper95": 0.0,
      "independent_seeds": 3,
      "method": "paired seed-cluster bootstrap; exploratory with only three clusters"
    },
    "hidden": {
      "mean": 0.0,
      "lower95": 0.0,
      "upper95": 0.0,
      "independent_seeds": 3,
      "method": "paired seed-cluster bootstrap; exploratory with only three clusters"
    },
    "order": {
      "mean": 0.0,
      "lower95": 0.0,
      "upper95": 0.0,
      "independent_seeds": 3,
      "method": "paired seed-cluster bootstrap; exploratory with only three clusters"
    }
  },
  "resources": {
    "peak_rss_bytes": 45256704,
    "peak_commit_or_vms_bytes": 31772672,
    "os_process_peak_working_set_bytes": 46206976,
    "os_process_peak_pagefile_commit_bytes": 32505856,
    "samples": 699,
    "interval_seconds": 0.25,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 161.36482490000344,
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
  "live_model_replaced": false,
  "normal_chat_vision_proven": false,
  "retention_gate_still_failed": true,
  "decision": "STOP_EVENT_DIALOGUE_V1"
}
