# Conditional credit-signal noise audit

Readonly measurements, no newlearner/qualityclaim. Source/mainweights unchanged. Detailed144inputs in measurements.json.

{
  "run": "runs/20261109T211546Z-g1-credit-noise-audit-v1-941266",
  "decision": "READONLY_CREDIT_SIGNAL_DIAGNOSTIC_ONLY",
  "summary": {
    "frozen_final": {
      "existing_score": {
        "replicate_cosine": 0.26838815823383366,
        "cross_signal_to_pair_noise": 0.004428854626278168,
        "pair_noise_energy": 4017.7713457652803,
        "sample_mean_norm": 5.318040216797515
      },
      "crossfit_baseline_score": {
        "replicate_cosine": 0.3155292833824614,
        "cross_signal_to_pair_noise": 0.0054714861161265305,
        "pair_noise_energy": 1014.7512311641672,
        "sample_mean_norm": 3.778485495745835
      },
      "biased_surrogate": {
        "replicate_cosine": 0.9972777874113103,
        "cross_signal_to_pair_noise": 1.385804231336752,
        "pair_noise_energy": 3.909470974299096,
        "sample_mean_norm": 2.6434384293793314
      }
    },
    "local_final": {
      "existing_score": {
        "replicate_cosine": 0.20188094044524152,
        "cross_signal_to_pair_noise": 0.0024523568346614213,
        "pair_noise_energy": 2313.2874086254847,
        "sample_mean_norm": 4.063969185535336
      },
      "crossfit_baseline_score": {
        "replicate_cosine": 0.24425670087518836,
        "cross_signal_to_pair_noise": 0.00359947936189858,
        "pair_noise_energy": 326.2231049385536,
        "sample_mean_norm": 1.695100140285506
      },
      "biased_surrogate": {
        "replicate_cosine": 0.9986839479425998,
        "cross_signal_to_pair_noise": 0.7604853100786272,
        "pair_noise_energy": 0.8686268907759942,
        "sample_mean_norm": 0.9776323190690048
      }
    },
    "surrogate_final": {
      "existing_score": {
        "replicate_cosine": 0.05199365407693857,
        "cross_signal_to_pair_noise": 0.0006503875385750744,
        "pair_noise_energy": 3.446592714404606,
        "sample_mean_norm": 0.13685554712989542
      },
      "crossfit_baseline_score": {
        "replicate_cosine": 0.1517994874565321,
        "cross_signal_to_pair_noise": 0.0017161851771419793,
        "pair_noise_energy": 0.3458939481116203,
        "sample_mean_norm": 0.04893799998893289
      },
      "biased_surrogate": {
        "replicate_cosine": 0.9827014251593539,
        "cross_signal_to_pair_noise": 0.11062691925323125,
        "pair_noise_energy": 0.007957639059447194,
        "sample_mean_norm": 0.02436841912427251
      }
    }
  },
  "paired_statistics": {
    "frozen_final": {
      "crossfit_vs_existing_noise_ratio": {
        "mean": 0.5533975053292415,
        "lower95": 0.42516378616558387,
        "upper95": 0.7146340665529684,
        "n_seeds": 3
      },
      "surrogate_minus_existing_replicate_cosine": {
        "mean": 0.7290487196149705,
        "lower95": 0.7188902874199409,
        "upper95": 0.7375840390672668,
        "n_seeds": 3
      }
    },
    "local_final": {
      "crossfit_vs_existing_noise_ratio": {
        "mean": 0.2273971008138915,
        "lower95": 0.10350957931812345,
        "upper95": 0.3822219534195764,
        "n_seeds": 3
      },
      "surrogate_minus_existing_replicate_cosine": {
        "mean": 0.7908151450824894,
        "lower95": 0.7689535697287577,
        "upper95": 0.8267206274922937,
        "n_seeds": 3
      }
    },
    "surrogate_final": {
      "crossfit_vs_existing_noise_ratio": {
        "mean": 0.3382242465656718,
        "lower95": 0.08617375172874393,
        "upper95": 0.7828972733666748,
        "n_seeds": 3
      },
      "surrogate_minus_existing_replicate_cosine": {
        "mean": 0.8994617506190271,
        "lower95": 0.8580368537525277,
        "upper95": 0.93953018762519,
        "n_seeds": 3
      }
    }
  },
  "measurements": 144,
  "pairs_per_input": 128,
  "forecast_seconds": 123.72562140027003,
  "all_source_checkpoints_unchanged": true,
  "all_weights_learningstate_rng_unchanged": true,
  "max_graph_state_error": 3.814697265625e-06,
  "max_graph_feature_error": 0.0,
  "numpy_forward_ticks": 801792,
  "extra_surrogate_forward_ticks": 801792,
  "backward_passes": 18432,
  "first_input_raw_signals": "artifacts/credit-noise/20261109T211546Z-g1-credit-noise-audit-v1-941266/first-predefined-input-signals.npz",
  "new_training": false,
  "control_only": true,
  "deployed": false,
  "main_local_learning_success": false,
  "normal_chat_or_vision_proven": false,
  "resources": {
    "peak_rss_bytes": 594030592,
    "peak_commit_or_vms_bytes": 994041856,
    "os_process_peak_working_set_bytes": 594989056,
    "os_process_peak_pagefile_commit_bytes": 1011220480,
    "samples": 1581,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 159.04555759999494,
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
  "limits": [
    "Finite MonteCarlo128pairs,3pairedmodels; nestedcases are not independentseeds.",
    "Crossmeanenergy maynegative; splitproxy not trueSNR. Surrogate is biased and not truegradient.",
    "No neuron/optimizer/head/RNGlearningupdate. Neuron/eligibilitysimulation only on discardedcopies.",
    "Crossfitbaseline variancecontrol not a newlearner, successfulquality or reopening scorefamilySTOP."
  ]
}
