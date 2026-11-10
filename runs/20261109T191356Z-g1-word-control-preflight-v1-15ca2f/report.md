# Word learnability control preflight

Sixfrozen NumPy baselines match exactstatebytes. Training-only timing, no heldout selection.

{
  "run": "runs/20261109T191356Z-g1-word-control-preflight-v1-15ca2f",
  "source_run": "runs/20261109T184028Z-g1-word-score-input-transfer-v2-c7a277",
  "decision": "CONTROL_PREFLIGHT_FEASIBLE",
  "all_matched": true,
  "frozen_baselines": [
    {
      "seed": 28001,
      "encoding": "legacy",
      "64_response_statebytes_exact": true,
      "source_frozen_calibration_exact": true
    },
    {
      "seed": 28001,
      "encoding": "known-digest128-v1",
      "64_response_statebytes_exact": true,
      "source_frozen_calibration_exact": true
    },
    {
      "seed": 28019,
      "encoding": "legacy",
      "64_response_statebytes_exact": true,
      "source_frozen_calibration_exact": true
    },
    {
      "seed": 28019,
      "encoding": "known-digest128-v1",
      "64_response_statebytes_exact": true,
      "source_frozen_calibration_exact": true
    },
    {
      "seed": 28027,
      "encoding": "legacy",
      "64_response_statebytes_exact": true,
      "source_frozen_calibration_exact": true
    },
    {
      "seed": 28027,
      "encoding": "known-digest128-v1",
      "64_response_statebytes_exact": true,
      "source_frozen_calibration_exact": true
    }
  ],
  "timing": {
    "legacy": {
      "responses": 64,
      "training_seconds": 11.545754599996144,
      "setup_seconds": 0.48620489999302663,
      "forecast_trial_with25pct_headroom": 250.27284812491416,
      "numpy_forward_ticks": 60624,
      "surrogate_forward_ticks": 60624,
      "backward_passes": 1372,
      "max_graph_state_error": 1.9073486328125e-06,
      "max_graph_feature_error": 0.0
    },
    "known-digest128-v1": {
      "responses": 64,
      "training_seconds": 11.208648200001335,
      "setup_seconds": 0.4809928000031505,
      "forecast_trial_with25pct_headroom": 243.52420500003063,
      "numpy_forward_ticks": 60624,
      "surrogate_forward_ticks": 60624,
      "backward_passes": 1372,
      "max_graph_state_error": 1.9073486328125e-06,
      "max_graph_feature_error": 0.0
    }
  },
  "forecast_round_seconds": 1511.3911593748344,
  "formal_training_permitted": true,
  "control_only": true,
  "new_training": "64trainingresponse timing copies only, no development/prior evaluation",
  "deployed": false,
  "normal_chat_or_vision_proven": false,
  "resources": {
    "peak_rss_bytes": 560476160,
    "peak_commit_or_vms_bytes": 981331968,
    "os_process_peak_working_set_bytes": 561463296,
    "os_process_peak_pagefile_commit_bytes": 982102016,
    "samples": 624,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 67.83123039999919,
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
  "torch_version": "2.13.0+cu130"
}
