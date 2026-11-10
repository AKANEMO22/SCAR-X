# Final paired-score adjustment

Local scalar return / synaptic eligibility, no network backward; quality gate unchanged.

{
  "run": "runs/20261109T131841Z-p1-paired-score-v2-0dc401",
  "decision": "GO_P2_RESEARCH_ONLY",
  "selection": {
    "selected_condition": "raw-independent",
    "selected_lr": 0.002,
    "tasks": {
      "order": {
        "probe_gain": 0.18229166666666666,
        "direct": 0.90234375,
        "initial_probe": 0.7213541666666666
      },
      "xor": {
        "probe_gain": 0.25390625,
        "direct": 0.7630208333333334,
        "initial_probe": 0.5494791666666666
      },
      "strict_binding": {
        "probe_gain": 0.010416666666666666,
        "direct": 0.4986979166666667,
        "initial_probe": 0.4752604166666667
      }
    },
    "screen_pass": true,
    "confirmation_opened": true
  },
  "confirmation_statistics": {
    "order": {
      "probe_gain": {
        "mean": 0.183349609375,
        "lower95": 0.146484375,
        "upper95": 0.225341796875,
        "n_seeds": 8
      },
      "direct": {
        "mean": 0.919921875,
        "lower95": 0.88232421875,
        "upper95": 0.956298828125,
        "n_seeds": 8
      },
      "vs_frozen": {
        "mean": 0.183349609375,
        "lower95": 0.146484375,
        "upper95": 0.225341796875,
        "n_seeds": 8
      },
      "gain_p": 0.0078125,
      "vs_frozen_p": 0.0078125,
      "holm_gain_p": 0.0234375,
      "holm_vs_frozen_p": 0.0234375,
      "pass": true
    },
    "xor": {
      "probe_gain": {
        "mean": 0.199951171875,
        "lower95": 0.150634765625,
        "upper95": 0.246826171875,
        "n_seeds": 8
      },
      "direct": {
        "mean": 0.74560546875,
        "lower95": 0.69775390625,
        "upper95": 0.793701171875,
        "n_seeds": 8
      },
      "vs_frozen": {
        "mean": 0.199951171875,
        "lower95": 0.150634765625,
        "upper95": 0.246826171875,
        "n_seeds": 8
      },
      "gain_p": 0.0078125,
      "vs_frozen_p": 0.0078125,
      "holm_gain_p": 0.0234375,
      "holm_vs_frozen_p": 0.0234375,
      "pass": true
    },
    "strict_binding": {
      "probe_gain": {
        "mean": -0.012939453125,
        "lower95": -0.023681640625,
        "upper95": -0.0009765625,
        "n_seeds": 8
      },
      "direct": {
        "mean": 0.49560546875,
        "lower95": 0.484375,
        "upper95": 0.50732421875,
        "n_seeds": 8
      },
      "vs_frozen": {
        "mean": -0.012939453125,
        "lower95": -0.023681640625,
        "upper95": -0.0009765625,
        "n_seeds": 8
      },
      "gain_p": 0.09375,
      "vs_frozen_p": 0.09375,
      "holm_gain_p": 0.09375,
      "holm_vs_frozen_p": 0.09375,
      "pass": false
    }
  },
  "primary_representation_gate": true,
  "screen_trials": 108,
  "confirmation_trials": 120,
  "resources": {
    "peak_rss_bytes": 165040128,
    "peak_commit_or_vms_bytes": 187883520,
    "os_process_peak_working_set_bytes": 51671040,
    "os_process_peak_pagefile_commit_bytes": 38653952,
    "samples": 14236,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 1292.7726668000032,
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
  "network_backward": false,
  "limitations": [
    "Eight rollouts vs historical one; no equal-cost advantage to old scores or deterministic BPTT.",
    "No new matched-stochastic BPTT control; negative result does not prove universal local-learning impossibility.",
    "Raw-estimator math does not establish unbiased floating-point clipped Adam learner or useful representation."
  ]
}
