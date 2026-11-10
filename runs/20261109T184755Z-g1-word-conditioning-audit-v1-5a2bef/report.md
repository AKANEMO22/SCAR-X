# Word conditioning audit

Readonly firstword sensitivity and usefulness, never a full model reply or learning result.

{
  "run": "runs/20261109T184755Z-g1-word-conditioning-audit-v1-5a2bef",
  "source_run": "runs/20261109T175033Z-g1-word-score-transfer-v1-1b1604",
  "decision": "READONLY_CONDITIONING_DIAGNOSTIC_ONLY",
  "summary": {
    "frozen": {
      "train16": {
        "real": {
          "first_token_accuracy": 0.6458333333333334,
          "nll": 1.3437882264455159
        },
        "zero": {
          "first_token_accuracy": 0.3125,
          "nll": 2.369577646255493,
          "argmax_flip_vs_real": 0.6875,
          "probability_l1_vs_real": 1.1828905741373699,
          "hidden_rms_vs_real": 0.312582790851593
        },
        "cross_family": {
          "first_token_accuracy": 0.20833333333333334,
          "nll": 3.5048258304595947,
          "argmax_flip_vs_real": 0.8125,
          "probability_l1_vs_real": 1.4078160127003987,
          "hidden_rms_vs_real": 0.40975870688756305
        }
      },
      "development": {
        "real": {
          "first_token_accuracy": 0.3118279569892473,
          "nll": 2.6529600620269775
        },
        "zero": {
          "first_token_accuracy": 0.3440860215053763,
          "nll": 2.4738094806671143,
          "argmax_flip_vs_real": 0.5268817204301075,
          "probability_l1_vs_real": 1.0833600759506226,
          "hidden_rms_vs_real": 0.3100413680076599
        },
        "cross_family": {
          "first_token_accuracy": 0.22580645161290325,
          "nll": 3.065194765726725,
          "argmax_flip_vs_real": 0.7634408602150539,
          "probability_l1_vs_real": 1.338358998298645,
          "hidden_rms_vs_real": 0.4000500837961833
        }
      },
      "prior": {
        "real": {
          "first_token_accuracy": 0.3680555555555556,
          "nll": 2.2944770654042563
        },
        "zero": {
          "first_token_accuracy": 0.2986111111111111,
          "nll": 2.5563605626424155,
          "argmax_flip_vs_real": 0.6388888888888888,
          "probability_l1_vs_real": 1.1363461017608643,
          "hidden_rms_vs_real": 0.31715330481529236
        },
        "cross_family": {
          "first_token_accuracy": 0.16666666666666666,
          "nll": 3.3464833895365396,
          "argmax_flip_vs_real": 0.7777777777777777,
          "probability_l1_vs_real": 1.3360661268234253,
          "hidden_rms_vs_real": 0.4001001218954722
        }
      }
    },
    "full": {
      "train16": {
        "real": {
          "first_token_accuracy": 0.6666666666666666,
          "nll": 1.629295786221822
        },
        "zero": {
          "first_token_accuracy": 0.3541666666666667,
          "nll": 2.385401725769043,
          "argmax_flip_vs_real": 0.5625,
          "probability_l1_vs_real": 1.0331681569417317,
          "hidden_rms_vs_real": 0.3191525340080261
        },
        "cross_family": {
          "first_token_accuracy": 0.3125,
          "nll": 2.62972625096639,
          "argmax_flip_vs_real": 0.7916666666666666,
          "probability_l1_vs_real": 1.150862733523051,
          "hidden_rms_vs_real": 0.4022599558035533
        }
      },
      "development": {
        "real": {
          "first_token_accuracy": 0.3440860215053763,
          "nll": 2.2461020151774087
        },
        "zero": {
          "first_token_accuracy": 0.3763440860215053,
          "nll": 2.346374193827311,
          "argmax_flip_vs_real": 0.3870967741935483,
          "probability_l1_vs_real": 0.8355846405029297,
          "hidden_rms_vs_real": 0.29995789130528766
        },
        "cross_family": {
          "first_token_accuracy": 0.3118279569892473,
          "nll": 2.4945704142252603,
          "argmax_flip_vs_real": 0.6129032258064516,
          "probability_l1_vs_real": 1.0727947155634563,
          "hidden_rms_vs_real": 0.3950621883074443
        }
      },
      "prior": {
        "real": {
          "first_token_accuracy": 0.3333333333333333,
          "nll": 2.2067875067392984
        },
        "zero": {
          "first_token_accuracy": 0.34027777777777773,
          "nll": 2.4061283270517984,
          "argmax_flip_vs_real": 0.5277777777777778,
          "probability_l1_vs_real": 0.9215993483861288,
          "hidden_rms_vs_real": 0.3026711444060008
        },
        "cross_family": {
          "first_token_accuracy": 0.18055555555555558,
          "nll": 3.1300370693206787,
          "argmax_flip_vs_real": 0.75,
          "probability_l1_vs_real": 1.137019157409668,
          "hidden_rms_vs_real": 0.39547712604204815
        }
      }
    },
    "shuffled": {
      "train16": {
        "real": {
          "first_token_accuracy": 0.6041666666666666,
          "nll": 1.585452675819397
        },
        "zero": {
          "first_token_accuracy": 0.375,
          "nll": 2.216256777445475,
          "argmax_flip_vs_real": 0.4375,
          "probability_l1_vs_real": 0.7620366414388021,
          "hidden_rms_vs_real": 0.299833079179128
        },
        "cross_family": {
          "first_token_accuracy": 0.25,
          "nll": 2.664807081222534,
          "argmax_flip_vs_real": 0.6458333333333334,
          "probability_l1_vs_real": 0.9251751701037089,
          "hidden_rms_vs_real": 0.38603784640630084
        }
      },
      "development": {
        "real": {
          "first_token_accuracy": 0.24731182795698925,
          "nll": 2.566398859024048
        },
        "zero": {
          "first_token_accuracy": 0.3440860215053763,
          "nll": 2.216498613357544,
          "argmax_flip_vs_real": 0.4408602150537635,
          "probability_l1_vs_real": 0.7739372054735819,
          "hidden_rms_vs_real": 0.28831084569295246
        },
        "cross_family": {
          "first_token_accuracy": 0.26881720430107525,
          "nll": 2.426298141479492,
          "argmax_flip_vs_real": 0.7849462365591396,
          "probability_l1_vs_real": 1.0502514044443767,
          "hidden_rms_vs_real": 0.36970168352127075
        }
      },
      "prior": {
        "real": {
          "first_token_accuracy": 0.3055555555555555,
          "nll": 2.4500479698181152
        },
        "zero": {
          "first_token_accuracy": 0.34027777777777785,
          "nll": 2.3261590003967285,
          "argmax_flip_vs_real": 0.5625,
          "probability_l1_vs_real": 0.8278201421101888,
          "hidden_rms_vs_real": 0.2907455066839854
        },
        "cross_family": {
          "first_token_accuracy": 0.09722222222222222,
          "nll": 3.013313372929891,
          "argmax_flip_vs_real": 0.8402777777777778,
          "probability_l1_vs_real": 0.9864723483721415,
          "hidden_rms_vs_real": 0.3743502199649811
        }
      }
    }
  },
  "paired_gains": {
    "frozen": {
      "train16": {
        "zero": {
          "mean": 0.3333333333333333,
          "lower95": 0.1875,
          "upper95": 0.4375,
          "n_seeds": 3
        },
        "cross_family": {
          "mean": 0.4375,
          "lower95": 0.4375,
          "upper95": 0.4375,
          "n_seeds": 3
        }
      },
      "development": {
        "zero": {
          "mean": -0.03225806451612904,
          "lower95": -0.09677419354838712,
          "upper95": 0.0,
          "n_seeds": 3
        },
        "cross_family": {
          "mean": 0.08602150537634408,
          "lower95": 0.0,
          "upper95": 0.19354838709677416,
          "n_seeds": 3
        }
      },
      "prior": {
        "zero": {
          "mean": 0.06944444444444446,
          "lower95": -0.0625,
          "upper95": 0.2916666666666667,
          "n_seeds": 3
        },
        "cross_family": {
          "mean": 0.20138888888888892,
          "lower95": 0.08333333333333331,
          "upper95": 0.3333333333333333,
          "n_seeds": 3
        }
      }
    },
    "full": {
      "train16": {
        "zero": {
          "mean": 0.3125,
          "lower95": 0.25,
          "upper95": 0.375,
          "n_seeds": 3
        },
        "cross_family": {
          "mean": 0.3541666666666667,
          "lower95": 0.1875,
          "upper95": 0.5,
          "n_seeds": 3
        }
      },
      "development": {
        "zero": {
          "mean": -0.032258064516129024,
          "lower95": -0.12903225806451613,
          "upper95": 0.06451612903225806,
          "n_seeds": 3
        },
        "cross_family": {
          "mean": 0.03225806451612905,
          "lower95": -0.12903225806451613,
          "upper95": 0.19354838709677422,
          "n_seeds": 3
        }
      },
      "prior": {
        "zero": {
          "mean": -0.006944444444444438,
          "lower95": -0.041666666666666685,
          "upper95": 0.041666666666666685,
          "n_seeds": 3
        },
        "cross_family": {
          "mean": 0.15277777777777776,
          "lower95": 0.020833333333333315,
          "upper95": 0.2916666666666667,
          "n_seeds": 3
        }
      }
    },
    "shuffled": {
      "train16": {
        "zero": {
          "mean": 0.22916666666666666,
          "lower95": 0.1875,
          "upper95": 0.25,
          "n_seeds": 3
        },
        "cross_family": {
          "mean": 0.3541666666666667,
          "lower95": 0.25,
          "upper95": 0.5,
          "n_seeds": 3
        }
      },
      "development": {
        "zero": {
          "mean": -0.09677419354838708,
          "lower95": -0.19354838709677416,
          "upper95": 0.0,
          "n_seeds": 3
        },
        "cross_family": {
          "mean": -0.021505376344086013,
          "lower95": -0.25806451612903225,
          "upper95": 0.16129032258064516,
          "n_seeds": 3
        }
      },
      "prior": {
        "zero": {
          "mean": -0.034722222222222245,
          "lower95": -0.10416666666666669,
          "upper95": 0.04166666666666663,
          "n_seeds": 3
        },
        "cross_family": {
          "mean": 0.2083333333333333,
          "lower95": 0.04166666666666666,
          "upper95": 0.29166666666666663,
          "n_seeds": 3
        }
      }
    }
  },
  "all_state_unchanged": true,
  "new_training": false,
  "deployed": false,
  "normal_chat_or_vision_proven": false,
  "resources": {
    "peak_rss_bytes": 47730688,
    "peak_commit_or_vms_bytes": 66232320,
    "os_process_peak_working_set_bytes": 48742400,
    "os_process_peak_pagefile_commit_bytes": 67194880,
    "samples": 18,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 0.988194299992756,
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
  "limitations": [
    "Firstword only; shared targetfirstwords confound accuracy, sensitivity is not useful semantics.",
    "Batched RNG is newdiagnostic, not original v1 decoder scores. Three oldfreshmodels percondition; no newvalidation."
  ]
}
