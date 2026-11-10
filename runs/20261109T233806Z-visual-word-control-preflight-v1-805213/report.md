# Isolated visual ALIF CONTROL preflight

No devquality evaluated; mainALIF/locallearning unchanged.

{
  "run": "runs/20261109T233806Z-visual-word-control-preflight-v1-805213",
  "source_run": "runs/20261109T222354Z-visual-word-readout-v1-a11424",
  "decision": "VISUAL_CONTROL_PREFLIGHT_FEASIBLE",
  "all_matched": true,
  "frozen_baselines": [
    {
      "seed": 33113,
      "64_trainresponses_fullstatebytes_exact": true,
      "source_initialhidden_calibration_exact": true
    },
    {
      "seed": 33119,
      "64_trainresponses_fullstatebytes_exact": true,
      "source_initialhidden_calibration_exact": true
    },
    {
      "seed": 33133,
      "64_trainresponses_fullstatebytes_exact": true,
      "source_initialhidden_calibration_exact": true
    }
  ],
  "timing": {
    "seed": 34103,
    "responses": 64,
    "training_seconds": 0.8462284999986878,
    "setup_seconds": 0.17384709999896586,
    "forecast_trial_with25pct_headroom": 75.60416387495934,
    "forecast_round_seconds": 316.812491624878,
    "max_graph_state_error": 3.814697265625e-06,
    "max_graph_feature_error": 5.960464477539063e-08,
    "max_graph_loss_error": 6.915539656482395e-07,
    "backward_passes_timing_only": 64,
    "codec_nextupdate_generation_exact": true
  },
  "formal_training_permitted": true,
  "control_only": true,
  "deployed": false,
  "normal_chat_or_vision_proven": false,
  "dataset": "data/candidates/20261109T221711Z-visual-word-dataset-v1-f93af5/dataset.json",
  "dataset_sha256": "0fb04abefe67451952a2d73634fded9c86243d5874356d68299ff047deecf6a2",
  "resources": {
    "peak_rss_bytes": 565088256,
    "peak_commit_or_vms_bytes": 986120192,
    "os_process_peak_working_set_bytes": 566042624,
    "os_process_peak_pagefile_commit_bytes": 986865664,
    "samples": 42,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 3.816577799996594,
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
