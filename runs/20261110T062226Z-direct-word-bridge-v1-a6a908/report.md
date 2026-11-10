# Directwordbridgepilot

STOP_DIRECT_WORD_BRIDGE_V1

Neuralfreewordoutputs, no targetlookup. No main/P1/P2/generalchatclaim.

{
  "run": "runs/20261110T062226Z-direct-word-bridge-v1-a6a908",
  "decision": "STOP_DIRECT_WORD_BRIDGE_V1",
  "summary": {
    "correct": {
      "macro_development": 0.359375,
      "macro_train": 0.9947916666666666,
      "probe_gain": 0.140625,
      "restore_initial_hidden_dev": 0.17708333333333334
    },
    "shuffled": {
      "macro_development": 0.171875,
      "macro_train": 0.765625,
      "probe_gain": 0.0625,
      "restore_initial_hidden_dev": 0.11458333333333333
    },
    "zero": {
      "macro_development": 0.16666666666666666,
      "macro_train": 0.9270833333333334,
      "probe_gain": 0.0,
      "restore_initial_hidden_dev": 0.16666666666666666
    },
    "frozen": {
      "macro_development": 0.16666666666666666,
      "macro_train": 0.9270833333333334,
      "probe_gain": 0.0,
      "restore_initial_hidden_dev": 0.16666666666666666
    },
    "bptt_control": {
      "macro_development": 0.3645833333333333,
      "macro_train": 1.0,
      "probe_gain": 0.11458333333333333,
      "restore_initial_hidden_dev": 0.171875
    }
  },
  "gates": {
    "macro65_eachseed": false,
    "family50_eachseed": false,
    "gain10_allpositive_lower95": true,
    "resethead_probe_gain5_lower95": false
  },
  "paired_hidden_dev_gain": {
    "mean": 0.19270833333333334,
    "lower95": 0.1875,
    "upper95": 0.203125,
    "n_seeds": 3
  },
  "paired_resethead_probe_gain": {
    "mean": 0.140625,
    "lower95": 0.0,
    "upper95": 0.25,
    "n_seeds": 3
  },
  "proposal30_approved": true,
  "all_cold_exact": true,
  "zero_frozen_exact": true,
  "dataset": "data/candidates/20261110T061343Z-direct-word-semantic-review-v1-921842/reviewed.json",
  "dataset_sha256": "585fd541cdc2651e554e7cd9b7a0ae6d04d0dd5fc0acc1104add773a92864d0b",
  "deployed": false,
  "main_weights_changed": false,
  "normal_chat_or_vision_proven": false,
  "retention_proven": false,
  "resources": {
    "peak_rss_bytes": 627220480,
    "peak_commit_or_vms_bytes": 1024737280,
    "os_process_peak_working_set_bytes": 46419968,
    "os_process_peak_pagefile_commit_bytes": 32784384,
    "samples": 878,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 68.11508920000051,
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
  "seconds": 68.28158979999716,
  "preflight_run": "runs/20261110T061535Z-direct-word-preflight-v1-6591af",
  "limits": [
    "Smallclosed8style/no multi-turn/generalchat/vision/lifelong;3pairedseeds/inspectedstyles/3historicaloverlaps.",
    "Qwen64textreference separatecost/notlabels; no mainimport/Transformer/proxy.",
    "P1/P2failedgates unchanged. No automatic tuning/data/encoder/LR/epoch round after pilotfailure."
  ]
}
