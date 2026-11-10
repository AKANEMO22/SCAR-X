# ISOLATED ALIF word learnability control

Surrogate/globaltemporal control only, never mainSCAR-X. Reused actualparitychecked frozen/local baselines.

{
  "run": "runs/20261109T192850Z-g1-word-learnability-control-v1-d74480",
  "source_run": "runs/20261109T184028Z-g1-word-score-input-transfer-v2-c7a277",
  "preflight_run": "runs/20261109T191356Z-g1-word-control-preflight-v1-15ca2f",
  "decision": "ISOLATED_WORD_LEARNABILITY_CONTROL_ONLY",
  "control_only": true,
  "summary": {
    "legacy": {
      "full": {
        "train16": {
          "strict_style_exact": 0.9375,
          "normalized_style_exact": 0.9375,
          "teacher_token_accuracy": 0.9905753968253969,
          "teacher_nll": 0.017479018389525754,
          "first_token_accuracy": 0.9583333333333334
        },
        "development": {
          "strict_style_exact": 0.3548387096774193,
          "normalized_style_exact": 0.3655913978494623,
          "teacher_token_accuracy": 0.8289042498719917,
          "teacher_nll": 0.7146516421348378,
          "first_token_accuracy": 0.5591397849462365
        },
        "prior": {
          "strict_style_exact": 0.7291666666666666,
          "normalized_style_exact": 0.7291666666666666,
          "teacher_token_accuracy": 0.9438822751322751,
          "teacher_nll": 0.2373627000773972,
          "first_token_accuracy": 0.7847222222222222
        }
      },
      "frozen": {
        "train16": {
          "strict_style_exact": 0.14583333333333334,
          "normalized_style_exact": 0.16666666666666666,
          "teacher_token_accuracy": 0.6582837301587302,
          "teacher_nll": 1.2809500707212176,
          "first_token_accuracy": 0.5416666666666666
        },
        "development": {
          "strict_style_exact": 0.02150537634408602,
          "normalized_style_exact": 0.03225806451612903,
          "teacher_token_accuracy": 0.483026113671275,
          "teacher_nll": 2.189940906038051,
          "first_token_accuracy": 0.3010752688172043
        },
        "prior": {
          "strict_style_exact": 0.027777777777777776,
          "normalized_style_exact": 0.0625,
          "teacher_token_accuracy": 0.5505621693121693,
          "teacher_nll": 1.8366171438941998,
          "first_token_accuracy": 0.34722222222222227
        }
      },
      "local_reference": {
        "train16": {
          "strict_style_exact": 0.10416666666666667,
          "normalized_style_exact": 0.10416666666666667,
          "teacher_token_accuracy": 0.5741071428571428,
          "teacher_nll": 1.5391542488762748,
          "first_token_accuracy": 0.4791666666666667
        },
        "development": {
          "strict_style_exact": 0.06451612903225806,
          "normalized_style_exact": 0.06451612903225806,
          "teacher_token_accuracy": 0.5058115719406042,
          "teacher_nll": 1.8561206871390883,
          "first_token_accuracy": 0.3333333333333333
        },
        "prior": {
          "strict_style_exact": 0.03472222222222222,
          "normalized_style_exact": 0.041666666666666664,
          "teacher_token_accuracy": 0.5248842592592592,
          "teacher_nll": 1.7057201874010133,
          "first_token_accuracy": 0.3194444444444445
        }
      }
    },
    "known-digest128-v1": {
      "full": {
        "train16": {
          "strict_style_exact": 0.9166666666666666,
          "normalized_style_exact": 0.9166666666666666,
          "teacher_token_accuracy": 0.9847222222222222,
          "teacher_nll": 0.024905437427427318,
          "first_token_accuracy": 1.0
        },
        "development": {
          "strict_style_exact": 0.4408602150537635,
          "normalized_style_exact": 0.4408602150537635,
          "teacher_token_accuracy": 0.8796466973886329,
          "teacher_nll": 0.5036540982227646,
          "first_token_accuracy": 0.5483870967741935
        },
        "prior": {
          "strict_style_exact": 0.7777777777777777,
          "normalized_style_exact": 0.7777777777777777,
          "teacher_token_accuracy": 0.9541170634920634,
          "teacher_nll": 0.14115048515143988,
          "first_token_accuracy": 0.8402777777777777
        }
      },
      "frozen": {
        "train16": {
          "strict_style_exact": 0.20833333333333334,
          "normalized_style_exact": 0.25,
          "teacher_token_accuracy": 0.6671626984126985,
          "teacher_nll": 1.245457122924914,
          "first_token_accuracy": 0.5416666666666666
        },
        "development": {
          "strict_style_exact": 0.07526881720430108,
          "normalized_style_exact": 0.08602150537634408,
          "teacher_token_accuracy": 0.5383000512032771,
          "teacher_nll": 1.958938181862699,
          "first_token_accuracy": 0.3548387096774193
        },
        "prior": {
          "strict_style_exact": 0.10416666666666669,
          "normalized_style_exact": 0.13888888888888892,
          "teacher_token_accuracy": 0.5755125661375661,
          "teacher_nll": 1.718053132321251,
          "first_token_accuracy": 0.4375
        }
      },
      "local_reference": {
        "train16": {
          "strict_style_exact": 0.125,
          "normalized_style_exact": 0.125,
          "teacher_token_accuracy": 0.624454365079365,
          "teacher_nll": 1.4162797911184484,
          "first_token_accuracy": 0.5
        },
        "development": {
          "strict_style_exact": 0.06451612903225806,
          "normalized_style_exact": 0.07526881720430108,
          "teacher_token_accuracy": 0.529800307219662,
          "teacher_nll": 1.7968085179323114,
          "first_token_accuracy": 0.3548387096774193
        },
        "prior": {
          "strict_style_exact": 0.09027777777777778,
          "normalized_style_exact": 0.09722222222222221,
          "teacher_token_accuracy": 0.6059027777777778,
          "teacher_nll": 1.4012805979391276,
          "first_token_accuracy": 0.4583333333333333
        }
      }
    }
  },
  "paired_statistics": {
    "legacy": {
      "vs_frozen": {
        "train16": {
          "mean": 0.7708333333333334,
          "lower95": 0.6875,
          "upper95": 0.8125,
          "n_seeds": 3
        },
        "development": {
          "mean": 0.3333333333333333,
          "lower95": 0.19354838709677416,
          "upper95": 0.4193548387096773,
          "n_seeds": 3
        },
        "prior": {
          "mean": 0.6666666666666666,
          "lower95": 0.6041666666666666,
          "upper95": 0.7083333333333334,
          "n_seeds": 3
        }
      },
      "vs_local_full": {
        "train16": {
          "mean": 0.8333333333333334,
          "lower95": 0.8125,
          "upper95": 0.875,
          "n_seeds": 3
        },
        "development": {
          "mean": 0.3010752688172043,
          "lower95": 0.16129032258064516,
          "upper95": 0.3870967741935483,
          "n_seeds": 3
        },
        "prior": {
          "mean": 0.6875,
          "lower95": 0.6666666666666666,
          "upper95": 0.7083333333333334,
          "n_seeds": 3
        }
      }
    },
    "known-digest128-v1": {
      "vs_frozen": {
        "train16": {
          "mean": 0.6666666666666666,
          "lower95": 0.625,
          "upper95": 0.75,
          "n_seeds": 3
        },
        "development": {
          "mean": 0.3548387096774193,
          "lower95": 0.25806451612903225,
          "upper95": 0.41935483870967744,
          "n_seeds": 3
        },
        "prior": {
          "mean": 0.6388888888888888,
          "lower95": 0.5833333333333333,
          "upper95": 0.7291666666666666,
          "n_seeds": 3
        }
      },
      "vs_local_full": {
        "train16": {
          "mean": 0.7916666666666666,
          "lower95": 0.75,
          "upper95": 0.875,
          "n_seeds": 3
        },
        "development": {
          "mean": 0.3655913978494623,
          "lower95": 0.2903225806451613,
          "upper95": 0.41935483870967744,
          "n_seeds": 3
        },
        "prior": {
          "mean": 0.6805555555555555,
          "lower95": 0.6458333333333333,
          "upper95": 0.7291666666666666,
          "n_seeds": 3
        }
      }
    }
  },
  "interpretations": {
    "legacy": "CONTROL_FITS_TRAIN_GENERALIZATION_UNRESOLVED",
    "known-digest128-v1": "CONTROL_LIMITED_LEARNABILITY_MAIN_UNPROVEN"
  },
  "reused_frozen_trials": 6,
  "reused_local_trials": 6,
  "all_cold_exact": true,
  "resources": {
    "peak_rss_bytes": 623366144,
    "peak_commit_or_vms_bytes": 1014571008,
    "os_process_peak_working_set_bytes": 45813760,
    "os_process_peak_pagefile_commit_bytes": 32038912,
    "samples": 16129,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 1485.3560125000076,
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
  "technical_retry": {
    "source_run": "runs/20261109T192017Z-g1-word-learnability-control-v1-446c16",
    "failure_kind": "windows_atomic_replace_sharing",
    "error": "PermissionError WinError5 at experiment_status._write os.replace",
    "terminal_verified": true,
    "quality_evaluated": false,
    "last_worker_responses": 448,
    "budget_consumed_seconds": 88.18021559715271,
    "technical_retry_count": 1,
    "main_model_changed": false,
    "cause_scope": "Concurrent observer reader/atomic replace sharing is tested separately; no neural failure inferred."
  },
  "total_program_wall_seconds_including_failed_attempt": 1998.716662547973,
  "deployed": false,
  "weights_imported_into_scarx": false,
  "main_local_learning_success": false,
  "normal_chat_or_vision_proven": false,
  "limits": [
    "Threepairedseeds, known8styles and inspected dev/prior; not freshfinaltest or broadlanguage/vision.",
    "SurrogateBPTT is biased, global/temporal and isolated. Extra forward/backward compute, not matchedhardwarecost.",
    "Mainword-score route remains STOP; controlweights/algorithm cannot replace locallearning core."
  ]
}
