# Local credit alignment audit39

No neural training. Directions applied to disposable copies only.

{
  "run": "runs/20261110T080003Z-local-credit-alignment-audit-v1-d8de33",
  "source_run": "runs/20261110T062226Z-direct-word-bridge-v1-a6a908",
  "decision": "LOCAL_CREDIT_ALIGNMENT_DIAGNOSTIC_ONLY",
  "alignment_rows": [
    {
      "seed": 36313,
      "split": "train",
      "cases": 64,
      "nll_before": 0.01565547895092136,
      "delta_direction": -8.156845112418701e-05,
      "delta_signflip": -0.00023550641405924508,
      "delta_random_mean": 2.6898862967856085e-05,
      "gain_vs_random": 0.0001084673140920431,
      "gain_vs_signflip": -0.00015393796293505807,
      "alignment_fraction": 0.09375,
      "norm_match_ratio": null
    },
    {
      "seed": 36313,
      "split": "development",
      "cases": 64,
      "nll_before": 0.9068324651830587,
      "delta_direction": -0.13267975959807993,
      "delta_signflip": 0.10440250291411018,
      "delta_random_mean": 0.011258340095248684,
      "gain_vs_random": 0.1439380996933286,
      "gain_vs_signflip": 0.2370822625121901,
      "alignment_fraction": 0.671875,
      "norm_match_ratio": null
    },
    {
      "seed": 36319,
      "split": "train",
      "cases": 64,
      "nll_before": 0.012586112336278784,
      "delta_direction": -0.0016883833987306968,
      "delta_signflip": -0.0005452624699726381,
      "delta_random_mean": -3.0840110824656685e-05,
      "gain_vs_random": 0.0016575432879060402,
      "gain_vs_signflip": 0.0011431209287580586,
      "alignment_fraction": 0.109375,
      "norm_match_ratio": null
    },
    {
      "seed": 36319,
      "split": "development",
      "cases": 64,
      "nll_before": 0.936246920468397,
      "delta_direction": -0.12387295501144857,
      "delta_signflip": 0.09107243716158431,
      "delta_random_mean": 0.003579018782546158,
      "gain_vs_random": 0.12745197379399473,
      "gain_vs_signflip": 0.21494539217303288,
      "alignment_fraction": 0.6875,
      "norm_match_ratio": null
    },
    {
      "seed": 36341,
      "split": "train",
      "cases": 64,
      "nll_before": 0.010461512504322012,
      "delta_direction": 2.7033820202184745e-05,
      "delta_signflip": -0.0001711971206829037,
      "delta_random_mean": -0.00011913231755890244,
      "gain_vs_random": -0.0001461661377610873,
      "gain_vs_signflip": -0.0001982309408850884,
      "alignment_fraction": 0.109375,
      "norm_match_ratio": null
    },
    {
      "seed": 36341,
      "split": "development",
      "cases": 64,
      "nll_before": 0.7642856070909014,
      "delta_direction": -0.130827533566712,
      "delta_signflip": 0.06623511417795461,
      "delta_random_mean": 0.004773012180852412,
      "gain_vs_random": 0.1356005457475644,
      "gain_vs_signflip": 0.19706264774466659,
      "alignment_fraction": 0.546875,
      "norm_match_ratio": null
    }
  ],
  "dev_macro": {
    "alignment_fraction": 0.6354166666666666,
    "gain_vs_random": 0.1356635397449626,
    "gain_vs_signflip": 0.2163634341432965,
    "delta_direction": -0.12912674939208016,
    "delta_random_mean": 0.006536790352882418,
    "delta_signflip": 0.08723668475121638,
    "cases": 192
  },
  "dev_gain_vs_random_interval": {
    "mean": 0.1356635397449626,
    "lower95": 0.12745197379399473,
    "upper95": 0.1439380996933286,
    "n_seeds": 3
  },
  "dev_gain_vs_signflip_interval": {
    "mean": 0.2163634341432965,
    "lower95": 0.19706264774466656,
    "upper95": 0.2370822625121901,
    "n_seeds": 3
  },
  "heldout": {
    "seeds": 3,
    "mean_dev_delta": -8.430614986019037e-05,
    "interval": {
      "mean": -8.430614986019037e-05,
      "lower95": -0.00019261835523108096,
      "upper95": 1.9115740974243872e-07,
      "n_seeds": 3
    }
  },
  "flags": {
    "LOCAL_DIRECTION_ALIGNED": true,
    "DIRECTION_UNINFORMATIVE": false,
    "DIRECTION_CONFLICT_CONFIRMED": false
  },
  "model_sources": 15,
  "public_cases_per_source": 128,
  "actual_forward_segments": 384,
  "applied_counterfactual_evaluations": 7680,
  "neural_training": false,
  "neural_update_steps": 0,
  "torch_imported": false,
  "deployed": false,
  "main_weights_changed": false,
  "normal_chat_or_vision_proven": false,
  "resources": {
    "peak_rss_bytes": 52805632,
    "peak_commit_or_vms_bytes": 38191104,
    "os_process_peak_working_set_bytes": 55586816,
    "os_process_peak_pagefile_commit_bytes": 41000960,
    "samples": 196,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 22.657439999995404,
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
  "seconds": 23.008914899997762,
  "limits": [
    "15existing checkpoints / 3 paired seeds only; CIs are exploratory, not 15 independent seeds.",
    "Single-step local linear probe of ONE applied update; repeated application NOT tested.",
    "Direction is one-step readout-error x own-neuron eligibility; no gradient/backward computed or approximated.",
    "Applied to a disposable deep copy; source checkpoints and main weights unchanged.",
    "No newlearner/objective/epoch/LR/grid from this audit; any credit-rule change needs its own protocol and approval.",
    "Existing STOP35/36/37 and P1/P2 flags unchanged; this diagnoses a mechanism, it does not certify language ability."
  ]
}
