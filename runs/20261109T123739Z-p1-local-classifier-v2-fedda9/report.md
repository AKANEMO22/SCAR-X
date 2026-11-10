# Final local-classifier adjustment

{
  "run": "runs/20261109T123739Z-p1-local-classifier-v2-fedda9",
  "selection": {
    "selected_condition": "norm-last4",
    "selected_lr": 0.002,
    "tasks": {
      "order": {
        "probe_gain": 0.026041666666666668,
        "direct": 0.9661458333333334,
        "initial_probe": 0.9270833333333334
      },
      "xor": {
        "probe_gain": 0.037760416666666664,
        "direct": 0.9192708333333334,
        "initial_probe": 0.8619791666666666
      },
      "strict_binding": {
        "probe_gain": 0.0026041666666666665,
        "direct": 0.6549479166666666,
        "initial_probe": 0.5833333333333334
      }
    },
    "screen_pass": false,
    "confirmation_opened": false
  },
  "confirmation_statistics": {},
  "primary_representation_gate": false,
  "screen_trials": 108,
  "confirmation_trials": 0,
  "control_trials": 18,
  "control_summary": {
    "order": {
      "bptt-readout": 0.9375,
      "bptt-full": 0.9986979166666666
    },
    "xor": {
      "bptt-readout": 0.88671875,
      "bptt-full": 1.0
    },
    "strict_binding": {
      "bptt-readout": 0.6236979166666666,
      "bptt-full": 0.9166666666666666
    }
  },
  "resources": {
    "peak_rss_bytes": 750137344,
    "peak_commit_or_vms_bytes": 1250435072,
    "os_process_peak_working_set_bytes": 755302400,
    "os_process_peak_pagefile_commit_bytes": 1255788544,
    "samples": 1989,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 186.38411070000438,
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
  "deployed": false,
  "normal_chat_or_vision_proven": false,
  "retention_proven": false,
  "decision": "NO_GO_LOCAL_CLASSIFIER_FAMILY",
  "family_rounds_exhausted": true,
  "scope": "Same ALIF48, no deployment; actual controls use backward only in evaluation/controls.py; no equal-time efficiency claim"
}
