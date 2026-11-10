# Local credit alignment audit39

No neural training. Directions applied to disposable copies only.

{
  "run": "runs/20261110T080426Z-local-credit-alignment-audit-v1-db01fc",
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
      "delta_random_mean": 1.1669437311795804e-06,
      "delta_permuted_mean": 1.922125486312827e-05,
      "gain_vs_random": 8.273539485536684e-05,
      "gain_vs_permuted": 0.00010078970598731546,
      "gain_vs_signflip": -0.00015393796293505807,
      "alignment_fraction": 0.09375,
      "norm_match_ratio_random": 36.73366020196202,
      "norm_match_ratio_permuted": 15.927852194229237
    },
    {
      "seed": 36313,
      "split": "development",
      "cases": 64,
      "nll_before": 0.9068324651830587,
      "delta_direction": -0.13267975959807993,
      "delta_signflip": 0.10440250291411018,
      "delta_random_mean": 0.0069074155281230645,
      "delta_permuted_mean": 0.007085930909684903,
      "gain_vs_random": 0.139587175126203,
      "gain_vs_permuted": 0.13976569050776483,
      "gain_vs_signflip": 0.2370822625121901,
      "alignment_fraction": 0.671875,
      "norm_match_ratio_random": 4.932859415770863,
      "norm_match_ratio_permuted": 2.1307658451561973
    },
    {
      "seed": 36319,
      "split": "train",
      "cases": 64,
      "nll_before": 0.012586112336278784,
      "delta_direction": -0.0016883833987306968,
      "delta_signflip": -0.0005452624699726381,
      "delta_random_mean": 8.212609629788816e-06,
      "delta_permuted_mean": 7.424973551490313e-06,
      "gain_vs_random": 0.0016965960083604858,
      "gain_vs_permuted": 0.001695808372282187,
      "gain_vs_signflip": 0.0011431209287580586,
      "alignment_fraction": 0.109375,
      "norm_match_ratio_random": 37.383698325807515,
      "norm_match_ratio_permuted": 15.869441792126784
    },
    {
      "seed": 36319,
      "split": "development",
      "cases": 64,
      "nll_before": 0.936246920468397,
      "delta_direction": -0.12387295501144857,
      "delta_signflip": 0.09107243716158431,
      "delta_random_mean": 0.006774183007408405,
      "delta_permuted_mean": 0.002988785642827067,
      "gain_vs_random": 0.13064713801885697,
      "gain_vs_permuted": 0.12686174065427566,
      "gain_vs_signflip": 0.21494539217303288,
      "alignment_fraction": 0.6875,
      "norm_match_ratio_random": 4.715058772696321,
      "norm_match_ratio_permuted": 2.046587172976616
    },
    {
      "seed": 36341,
      "split": "train",
      "cases": 64,
      "nll_before": 0.010461512504322012,
      "delta_direction": 2.7033820202184745e-05,
      "delta_signflip": -0.0001711971206829037,
      "delta_random_mean": -8.962243674129065e-05,
      "delta_permuted_mean": -0.00012094770727693027,
      "gain_vs_random": -0.00011665625694347541,
      "gain_vs_permuted": -0.000147981527479115,
      "gain_vs_signflip": -0.0001982309408850884,
      "alignment_fraction": 0.109375,
      "norm_match_ratio_random": 30.403261741791503,
      "norm_match_ratio_permuted": 13.140922742193831
    },
    {
      "seed": 36341,
      "split": "development",
      "cases": 64,
      "nll_before": 0.7642856070909014,
      "delta_direction": -0.130827533566712,
      "delta_signflip": 0.06623511417795461,
      "delta_random_mean": 0.007521574850722084,
      "delta_permuted_mean": 0.0018296893863320582,
      "gain_vs_random": 0.13834910841743406,
      "gain_vs_permuted": 0.13265722295304405,
      "gain_vs_signflip": 0.19706264774466659,
      "alignment_fraction": 0.546875,
      "norm_match_ratio_random": 5.0856223429210035,
      "norm_match_ratio_permuted": 2.2218225893953547
    }
  ],
  "dev_macro": {
    "alignment_fraction": 0.6354166666666666,
    "gain_vs_random": 0.1361944738541647,
    "gain_vs_permuted": 0.13309488470502817,
    "gain_vs_signflip": 0.2163634341432965,
    "delta_direction": -0.12912674939208016,
    "delta_random_mean": 0.007067724462084518,
    "delta_permuted_mean": 0.003968135312948009,
    "delta_signflip": 0.08723668475121638,
    "norm_match_ratio_random": 4.9111801771293955,
    "norm_match_ratio_permuted": 2.1330585358427228,
    "cases": 192
  },
  "dev_gain_vs_random_interval": {
    "mean": 0.1361944738541647,
    "lower95": 0.13064713801885697,
    "upper95": 0.139587175126203,
    "n_seeds": 3
  },
  "dev_gain_vs_permuted_interval": {
    "mean": 0.13309488470502817,
    "lower95": 0.12686174065427566,
    "upper95": 0.13976569050776483,
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
    "DIRECTION_CONFLICT_CONFIRMED": false,
    "GAUSSIAN_CONTROL_OVERSHOOTS": true
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
    "peak_rss_bytes": 53395456,
    "peak_commit_or_vms_bytes": 38236160,
    "os_process_peak_working_set_bytes": 54870016,
    "os_process_peak_pagefile_commit_bytes": 39886848,
    "samples": 347,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 41.468109199995524,
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
  "seconds": 41.76091529999394,
  "limits": [
    "15existing checkpoints / 3 paired seeds only; CIs are exploratory, not 15 independent seeds.",
    "Single-step local linear probe of ONE applied update; repeated application NOT tested.",
    "Direction is one-step readout-error x own-neuron eligibility; no gradient/backward computed or approximated.",
    "Applied to a disposable deep copy; source checkpoints and main weights unchanged.",
    "No newlearner/objective/epoch/LR/grid from this audit; any credit-rule change needs its own protocol and approval.",
    "Existing STOP35/36/37 and P1/P2 flags unchanged; this diagnoses a mechanism, it does not certify language ability."
  ]
}
