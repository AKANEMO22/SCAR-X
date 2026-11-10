# P1 local auxiliary classifier adaptation

Manual local-loss adaptation under unchanged48ALIF topology; not a faithful DECOLLE reproduction, deployment or normalchat/vision claim.

| Task | Initial probe | Selected gain | Direct |
|---|---:|---:|---:|
| order | 0.9180 | 2.73pp | 0.9479 |
| xor | 0.8099 | -0.52pp | 0.7839 |
| strict_binding | 0.5560 | -0.65pp | 0.6250 |

{
  "run": "runs/20261109T114551Z-p1-local-classifier-v1-7384dd",
  "selection": {
    "selected_lr": 0.002,
    "scores": {
      "0.0002": 0.0,
      "0.002": 0.005208333333333333
    },
    "tasks": {
      "order": {
        "probe_gain": 0.02734375,
        "direct": 0.9479166666666666,
        "initial_probe": 0.91796875
      },
      "xor": {
        "probe_gain": -0.005208333333333333,
        "direct": 0.7838541666666666,
        "initial_probe": 0.8098958333333334
      },
      "strict_binding": {
        "probe_gain": -0.006510416666666667,
        "direct": 0.625,
        "initial_probe": 0.5559895833333334
      }
    },
    "screen_pass": false,
    "confirmation_opened": false
  },
  "confirmation_statistics": {},
  "primary_representation_gate": false,
  "screen_trials": 54,
  "confirmation_trials": 0,
  "resources": {
    "peak_rss_bytes": 137854976,
    "peak_commit_or_vms_bytes": 163057664,
    "os_process_peak_working_set_bytes": 42610688,
    "os_process_peak_pagefile_commit_bytes": 29396992,
    "samples": 486,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 41.851057600004424,
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
  "published_paper_reproduction": false,
  "decision": "PIVOT_LOCAL_CLASSIFIER_V1",
  "learnability_limit": "No new matched BPTT control for these tasks; failure does not refute published DECOLLE or every SNN."
}

Main output updates matched; auxiliary hidden updates are4pertrainingepisode vs scalar1, with different compute. No equal-time superiority claim.
