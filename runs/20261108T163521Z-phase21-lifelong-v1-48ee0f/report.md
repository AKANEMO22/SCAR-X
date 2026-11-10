# Lifelong task-agnostic stream

```json
{
  "summaries": {
    "readout_ema": {
      "final_accuracy": {
        "mean": 0.51171875,
        "lower95": 0.49788411458333337,
        "upper95": 0.5301106770833334,
        "n_seeds": 8
      },
      "forgetting": {
        "mean": 0.087890625,
        "lower95": 0.060546875,
        "upper95": 0.11572265624999999,
        "n_seeds": 8
      },
      "worst_forgetting": {
        "mean": 0.1650390625,
        "lower95": 0.11865234375,
        "upper95": 0.21192626953124982,
        "n_seeds": 8
      },
      "train_seconds": {
        "mean": 4.573400824912824,
        "lower95": 3.6471714393216645,
        "upper95": 5.106210181164215,
        "n_seeds": 8
      },
      "new_rule_gain": {
        "mean": -0.0234375,
        "lower95": -0.06689453125,
        "upper95": 0.01171875,
        "n_seeds": 8
      },
      "fresh_new_rule_gain": {
        "mean": 0.041015625,
        "lower95": -0.05712890625,
        "upper95": 0.14600830078124982,
        "n_seeds": 8
      },
      "accepted_noise": {
        "mean": 255.375,
        "lower95": 245.0,
        "upper95": 266.375,
        "n_seeds": 8
      },
      "updates": {
        "mean": 1920.0,
        "lower95": 1920.0,
        "upper95": 1920.0,
        "n_seeds": 8
      }
    },
    "score_ema": {
      "final_accuracy": {
        "mean": 0.51416015625,
        "lower95": 0.5011393229166667,
        "upper95": 0.5304361979166666,
        "n_seeds": 8
      },
      "forgetting": {
        "mean": 0.08772786458333334,
        "lower95": 0.057779947916666664,
        "upper95": 0.119140625,
        "n_seeds": 8
      },
      "worst_forgetting": {
        "mean": 0.1630859375,
        "lower95": 0.10400390625,
        "upper95": 0.2216796875,
        "n_seeds": 8
      },
      "train_seconds": {
        "mean": 5.476575237558791,
        "lower95": 5.128273425149018,
        "upper95": 5.730327636637503,
        "n_seeds": 8
      },
      "new_rule_gain": {
        "mean": -0.01806640625,
        "lower95": -0.05908203125,
        "upper95": 0.01220703125,
        "n_seeds": 8
      },
      "fresh_new_rule_gain": {
        "mean": 0.03955078125,
        "lower95": -0.0556640625,
        "upper95": 0.14111328125,
        "n_seeds": 8
      },
      "accepted_noise": {
        "mean": 255.375,
        "lower95": 245.0,
        "upper95": 266.375,
        "n_seeds": 8
      },
      "updates": {
        "mean": 1920.0,
        "lower95": 1920.0,
        "upper95": 1920.0,
        "n_seeds": 8
      }
    },
    "score_meta": {
      "final_accuracy": {
        "mean": 0.5008138020833334,
        "lower95": 0.5,
        "upper95": 0.5021158854166667,
        "n_seeds": 8
      },
      "forgetting": {
        "mean": 0.06656901041666667,
        "lower95": 0.03661702473958334,
        "upper95": 0.09977213541666667,
        "n_seeds": 8
      },
      "worst_forgetting": {
        "mean": 0.14111328125,
        "lower95": 0.0830078125,
        "upper95": 0.2001953125,
        "n_seeds": 8
      },
      "train_seconds": {
        "mean": 5.631770050100386,
        "lower95": 5.468707466021806,
        "upper95": 5.781400990824795,
        "n_seeds": 8
      },
      "new_rule_gain": {
        "mean": 0.02001953125,
        "lower95": 0.00048828125,
        "upper95": 0.044921875,
        "n_seeds": 8
      },
      "fresh_new_rule_gain": {
        "mean": 0.06298828125,
        "lower95": -0.01220703125,
        "upper95": 0.14599609375,
        "n_seeds": 8
      },
      "accepted_noise": {
        "mean": 255.375,
        "lower95": 245.0,
        "upper95": 266.375,
        "n_seeds": 8
      },
      "updates": {
        "mean": 1920.0,
        "lower95": 1920.0,
        "upper95": 1920.0,
        "n_seeds": 8
      }
    },
    "meta_decay": {
      "final_accuracy": {
        "mean": 0.5008138020833334,
        "lower95": 0.5,
        "upper95": 0.5021158854166667,
        "n_seeds": 8
      },
      "forgetting": {
        "mean": 0.06640625,
        "lower95": 0.03776041666666667,
        "upper95": 0.09766031901041661,
        "n_seeds": 8
      },
      "worst_forgetting": {
        "mean": 0.1455078125,
        "lower95": 0.0859375,
        "upper95": 0.2060546875,
        "n_seeds": 8
      },
      "train_seconds": {
        "mean": 5.5504738250328955,
        "lower95": 5.4218680250123725,
        "upper95": 5.6879937053292,
        "n_seeds": 8
      },
      "new_rule_gain": {
        "mean": 0.0146484375,
        "lower95": 0.0,
        "upper95": 0.03369140625,
        "n_seeds": 8
      },
      "fresh_new_rule_gain": {
        "mean": 0.06298828125,
        "lower95": -0.01220703125,
        "upper95": 0.14599609375,
        "n_seeds": 8
      },
      "accepted_noise": {
        "mean": 255.375,
        "lower95": 245.0,
        "upper95": 266.375,
        "n_seeds": 8
      },
      "updates": {
        "mean": 1920.0,
        "lower95": 1920.0,
        "upper95": 1920.0,
        "n_seeds": 8
      }
    },
    "selective": {
      "final_accuracy": {
        "mean": 0.5,
        "lower95": 0.49951171875,
        "upper95": 0.50048828125,
        "n_seeds": 8
      },
      "forgetting": {
        "mean": 0.07112630208333333,
        "lower95": 0.040852864583333336,
        "upper95": 0.10498046875,
        "n_seeds": 8
      },
      "worst_forgetting": {
        "mean": 0.13916015625,
        "lower95": 0.0927734375,
        "upper95": 0.189453125,
        "n_seeds": 8
      },
      "train_seconds": {
        "mean": 5.612583250085663,
        "lower95": 5.430247849993975,
        "upper95": 5.830456862312531,
        "n_seeds": 8
      },
      "new_rule_gain": {
        "mean": -0.013671875,
        "lower95": -0.037109375,
        "upper95": 0.0,
        "n_seeds": 8
      },
      "fresh_new_rule_gain": {
        "mean": 0.06298828125,
        "lower95": -0.01220703125,
        "upper95": 0.14599609375,
        "n_seeds": 8
      },
      "accepted_noise": {
        "mean": 75.625,
        "lower95": 68.625,
        "upper95": 83.125,
        "n_seeds": 8
      },
      "updates": {
        "mean": 1919.375,
        "lower95": 1918.625,
        "upper95": 1919.875,
        "n_seeds": 8
      }
    },
    "random_reject": {
      "final_accuracy": {
        "mean": 0.49560546875,
        "lower95": 0.4879557291666667,
        "upper95": 0.5009765625,
        "n_seeds": 8
      },
      "forgetting": {
        "mean": 0.068359375,
        "lower95": 0.04052734375,
        "upper95": 0.09554036458333333,
        "n_seeds": 8
      },
      "worst_forgetting": {
        "mean": 0.140625,
        "lower95": 0.07763671875,
        "upper95": 0.2021484375,
        "n_seeds": 8
      },
      "train_seconds": {
        "mean": 5.506333612498565,
        "lower95": 5.402472571026147,
        "upper95": 5.595186311832094,
        "n_seeds": 8
      },
      "new_rule_gain": {
        "mean": 0.02587890625,
        "lower95": -0.010266113281249989,
        "upper95": 0.06396484375,
        "n_seeds": 8
      },
      "fresh_new_rule_gain": {
        "mean": 0.06298828125,
        "lower95": -0.01220703125,
        "upper95": 0.14599609375,
        "n_seeds": 8
      },
      "accepted_noise": {
        "mean": 201.625,
        "lower95": 192.25,
        "upper95": 213.125,
        "n_seeds": 8
      },
      "updates": {
        "mean": 1919.875,
        "lower95": 1919.625,
        "upper95": 1920.0,
        "n_seeds": 8
      }
    },
    "si_readout": {
      "final_accuracy": {
        "mean": 0.5315755208333333,
        "lower95": 0.5104166666666666,
        "upper95": 0.5530598958333333,
        "n_seeds": 8
      },
      "forgetting": {
        "mean": 0.08658854166666667,
        "lower95": 0.057942708333333336,
        "upper95": 0.12109375,
        "n_seeds": 8
      },
      "worst_forgetting": {
        "mean": 0.13623046875,
        "lower95": 0.09178466796875001,
        "upper95": 0.18798828125,
        "n_seeds": 8
      },
      "train_seconds": {
        "mean": 4.708861624883866,
        "lower95": 4.627323739743406,
        "upper95": 4.7944380898569765,
        "n_seeds": 8
      },
      "new_rule_gain": {
        "mean": 0.0107421875,
        "lower95": -0.068359375,
        "upper95": 0.07032470703124982,
        "n_seeds": 8
      },
      "fresh_new_rule_gain": {
        "mean": 0.041015625,
        "lower95": -0.05712890625,
        "upper95": 0.14600830078124982,
        "n_seeds": 8
      },
      "accepted_noise": {
        "mean": 255.375,
        "lower95": 245.0,
        "upper95": 266.375,
        "n_seeds": 8
      },
      "updates": {
        "mean": 1920.0,
        "lower95": 1920.0,
        "upper95": 1920.0,
        "n_seeds": 8
      }
    }
  },
  "comparisons": {
    "metaplasticity": {
      "accuracy_gain": {
        "mean": -0.013346354166666657,
        "lower95": -0.029626464843749978,
        "upper95": -0.0003255208333333287,
        "n_seeds": 8
      },
      "forgetting_reduction": {
        "mean": 0.021158854166666668,
        "lower95": -0.010904947916666666,
        "upper95": 0.05338541666666667,
        "n_seeds": 8
      },
      "holm_p": 0.46875
    },
    "importance_decay": {
      "accuracy_gain": {
        "mean": 0.0,
        "lower95": 0.0,
        "upper95": 0.0,
        "n_seeds": 8
      },
      "forgetting_reduction": {
        "mean": 0.0001627604166666674,
        "lower95": -0.006510416666666666,
        "upper95": 0.007649739583333337,
        "n_seeds": 8
      },
      "holm_p": 1.0
    },
    "selective_quality": {
      "accuracy_gain": {
        "mean": 0.004394531250000014,
        "lower95": -0.0013020833333333148,
        "upper95": 0.012044270833333336,
        "n_seeds": 8
      },
      "forgetting_reduction": {
        "mean": -0.002766927083333333,
        "lower95": -0.03727213541666667,
        "upper95": 0.029134114583333336,
        "n_seeds": 8
      },
      "holm_p": 0.9375
    }
  },
  "absolute_gates": {
    "readout_ema": false,
    "score_ema": false,
    "score_meta": false,
    "meta_decay": false,
    "selective": false,
    "random_reject": false,
    "si_readout": false
  },
  "resources": {
    "peak_rss_bytes": 45588480,
    "peak_commit_or_vms_bytes": 64937984,
    "os_process_peak_working_set_bytes": 46395392,
    "os_process_peak_pagefile_commit_bytes": 65236992,
    "samples": 11397,
    "interval_seconds": 0.01,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 374.74376250000205,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  },
  "interpretation": "Finite stream, observable input families but no task IDs/boundaries. Final 64-example reversal is a plasticity diagnostic, not simultaneous retention of contradictory labels. SI readout adaptation only."
}
```
