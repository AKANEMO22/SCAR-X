# P1 neuron-local target eligibility

Same deterministic48-neuron ALIF, three temporal tasks. No core scale-up or live weights change.

| Task | SelectedLR | Development gain | Direct | Screen task pass |
|---|---:|---:|---:|---|
| order | 0.02 | 1.43pp | 66.80% | False |
| xor | 0.02 | -1.56pp | 55.21% | False |
| strict_binding | 0.02 | 1.95pp | 55.99% | False |

{
  "run": "runs/20261109T092905Z-p1-random-feedback-v2-68c153",
  "selection": {
    "selected_lr": 0.02,
    "development_mean_gain": {
      "0.002": 0.0,
      "0.02": 0.006076388888888889
    },
    "tasks": {
      "order": {
        "gain": 0.014322916666666666,
        "direct": 0.66796875
      },
      "xor": {
        "gain": -0.015625,
        "direct": 0.5520833333333334
      },
      "strict_binding": {
        "gain": 0.01953125,
        "direct": 0.5598958333333334
      }
    },
    "screen_pass": false,
    "confirmation_opened": false
  },
  "confirmation_statistics": {},
  "screen_trials": 54,
  "confirmation_trials": 0,
  "primary_representation_gate": false,
  "decision": "NO_GO_LOCAL_TARGET_FAMILY",
  "resources": {
    "peak_rss_bytes": 136151040,
    "peak_commit_or_vms_bytes": 160940032,
    "os_process_peak_working_set_bytes": 42369024,
    "os_process_peak_pagefile_commit_bytes": 29880320,
    "samples": 561,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 48.50091610000527,
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
  "elapsed_seconds": 48.578999999997905,
  "deployed": false,
  "retention_tested": false,
  "normal_chat_vision_proven": false,
  "learnability_control_limitation": "No new BPTT control is executed; existing task controls differ in budget/dynamics. Failure does not establish impossibility."
}

This rule uses an explicit supervised target-code third factor and local pseudo-derivative, not unsupervised STDP, exact e-prop, or whole-network backprop. Missing temporal learnability control limits negative conclusions. P2 retention and chat/vision quality are not established.
