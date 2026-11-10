# Directvisual37preflight

Same32forward/head, no devquality or mainchanges. Anchoredsurrogatefinite-difference is localmath, not hardspike/networkgradient.

{
  "run": "runs/20261110T064337Z-direct-visual-preflight-v1-927d70",
  "decision": "DIRECT_VISUAL_PREFLIGHT_PASS",
  "all21_forwardparity_checks": [
    {
      "seed": 37313,
      "method": "correct",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37313,
      "method": "shuffled",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37313,
      "method": "zero",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37313,
      "method": "frozen",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37313,
      "method": "no_image",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37313,
      "method": "no_question",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37313,
      "method": "bptt_control",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37321,
      "method": "correct",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37321,
      "method": "shuffled",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37321,
      "method": "zero",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37321,
      "method": "frozen",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37321,
      "method": "no_image",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37321,
      "method": "no_question",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37321,
      "method": "bptt_control",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37339,
      "method": "correct",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37339,
      "method": "shuffled",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37339,
      "method": "zero",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37339,
      "method": "frozen",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37339,
      "method": "no_image",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37339,
      "method": "no_question",
      "16responses_exact_visual32bytes": true
    },
    {
      "seed": 37339,
      "method": "bptt_control",
      "16responses_exact_visual32bytes": true
    }
  ],
  "mathematics": {
    "synapses": 64,
    "max_absolute_error": 1.777805671565602e-07,
    "cache_ticks": 16,
    "input_ticks": 36,
    "scope": "Anchoredsmooth own-neuron derivatives for SAME32 pooledfeature with fixedpresynaptictape; NOT hardspike/fullnetworkgradient."
  },
  "timing": {
    "correct": {
      "training_examples": 64,
      "train_seconds": 0.34663929999805987,
      "setup_seconds": 0.18818759999703616,
      "forecast_trial_seconds": 60.63441349993809,
      "array_bytes": 485832,
      "checkpoint_bytes": 643761,
      "nextlearn_generation_fullstate_exact": true
    },
    "bptt_control": {
      "training_examples": 64,
      "train_seconds": 0.9528012000082526,
      "setup_seconds": 0.19461130000127014,
      "forecast_trial_seconds": 78.82730012524917,
      "array_bytes": 485832,
      "checkpoint_bytes": 644435,
      "nextlearn_generation_fullstate_exact": true,
      "max_graph_state_error": 5.7220458984375e-06,
      "max_graph_feature_error": 5.960464477539063e-08,
      "max_graph_loss_error": 4.892482063922898e-07
    }
  },
  "forecast_round_seconds": 1327.9013433746331,
  "formal_training_permitted": true,
  "no_development_quality_evaluated": true,
  "deployed": false,
  "main_weights_changed": false,
  "proposal30_approved": true,
  "resources": {
    "peak_rss_bytes": 571056128,
    "peak_commit_or_vms_bytes": 987242496,
    "os_process_peak_working_set_bytes": 572100608,
    "os_process_peak_pagefile_commit_bytes": 988045312,
    "samples": 159,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 15.125742999996874,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  },
  "cap": {
    "enforced": true,
    "limit_bytes": 2000000000,
    "kind": "Windows per-process committed memory; shared pages monitored separately"
  }
}
