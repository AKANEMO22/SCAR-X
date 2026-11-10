# P1 neuron-local target eligibility

Same deterministic48-neuron ALIF, three temporal tasks. No core scale-up or live weights change.

| Task | SelectedLR | Development gain | Direct | Screen task pass |
|---|---:|---:|---:|---|
| order | 0.002 | -1.43pp | 72.92% | False |
| xor | 0.002 | 0.91pp | 50.00% | False |
| strict_binding | 0.002 | 3.78pp | 54.56% | False |

{
  "run": "runs/20261109T092029Z-p1-local-target-v1-d4efd9",
  "selection": {
    "selected_lr": 0.002,
    "development_mean_gain": {
      "0.002": 0.010850694444444444,
      "0.02": -0.022135416666666668
    },
    "tasks": {
      "order": {
        "gain": -0.014322916666666666,
        "direct": 0.7291666666666666
      },
      "xor": {
        "gain": 0.009114583333333334,
        "direct": 0.5
      },
      "strict_binding": {
        "gain": 0.037760416666666664,
        "direct": 0.5455729166666666
      }
    },
    "screen_pass": false,
    "confirmation_opened": false
  },
  "confirmation_statistics": {},
  "screen_trials": 54,
  "confirmation_trials": 0,
  "primary_representation_gate": false,
  "decision": "PIVOT_LOCAL_TARGET_V1",
  "resources": {
    "peak_rss_bytes": 135757824,
    "peak_commit_or_vms_bytes": 161685504,
    "os_process_peak_working_set_bytes": 41795584,
    "os_process_peak_pagefile_commit_bytes": 28823552,
    "samples": 560,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 48.773089000002074,
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
  "elapsed_seconds": 48.85899999999674,
  "deployed": false,
  "retention_tested": false,
  "normal_chat_vision_proven": false,
  "learnability_control_limitation": "No new BPTT control is executed; existing task controls differ in budget/dynamics. Failure does not establish impossibility."
}

This rule uses an explicit supervised target-code third factor and local pseudo-derivative, not unsupervised STDP, exact e-prop, or whole-network backprop. Missing temporal learnability control limits negative conclusions. P2 retention and chat/vision quality are not established.
