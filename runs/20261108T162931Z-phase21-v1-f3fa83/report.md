# Phase 2.1

```json
{
  "representation": {
    "xor": {
      "probe_gain": {
        "mean": 0.0146484375,
        "lower95": -0.0078125,
        "upper95": 0.037109375,
        "n_seeds": 8
      },
      "vs_readout": {
        "mean": 0.0146484375,
        "lower95": -0.0078125,
        "upper95": 0.037109375,
        "n_seeds": 8
      },
      "vs_shuffled": {
        "mean": 0.01171875,
        "lower95": -0.0087890625,
        "upper95": 0.0302734375,
        "n_seeds": 8
      },
      "accuracy": {
        "mean": 0.49755859375,
        "lower95": 0.49267578125,
        "upper95": 0.5,
        "n_seeds": 8
      },
      "probe": {
        "mean": 0.53271484375,
        "lower95": 0.505859375,
        "upper95": 0.556640625,
        "n_seeds": 8
      },
      "frozen_gain": {
        "mean": -0.017578125,
        "lower95": -0.0263671875,
        "upper95": -0.0107421875,
        "n_seeds": 8
      },
      "ood_probe": {
        "mean": 0.50244140625,
        "lower95": 0.484375,
        "upper95": 0.5185668945312498,
        "n_seeds": 8
      },
      "permutation_drop": {
        "mean": 0.0166015625,
        "lower95": 0.0,
        "upper95": 0.03564453125,
        "n_seeds": 8
      },
      "reset_drop": {
        "mean": 0.0302734375,
        "lower95": 0.0009765625,
        "upper95": 0.0625,
        "n_seeds": 8
      },
      "holm_signflip_p": 0.546875,
      "pass": false
    },
    "order": {
      "probe_gain": {
        "mean": 0.01806640625,
        "lower95": 0.0048828125,
        "upper95": 0.03076171875,
        "n_seeds": 8
      },
      "vs_readout": {
        "mean": 0.01806640625,
        "lower95": 0.0048828125,
        "upper95": 0.03076171875,
        "n_seeds": 8
      },
      "vs_shuffled": {
        "mean": 0.01416015625,
        "lower95": -0.00537109375,
        "upper95": 0.03515625,
        "n_seeds": 8
      },
      "accuracy": {
        "mean": 0.52294921875,
        "lower95": 0.5,
        "upper95": 0.56591796875,
        "n_seeds": 8
      },
      "probe": {
        "mean": 0.61376953125,
        "lower95": 0.59228515625,
        "upper95": 0.6328125,
        "n_seeds": 8
      },
      "frozen_gain": {
        "mean": 0.037109375,
        "lower95": -0.005859375,
        "upper95": 0.09375,
        "n_seeds": 8
      },
      "ood_probe": {
        "mean": 0.68408203125,
        "lower95": 0.6474609375,
        "upper95": 0.71875,
        "n_seeds": 8
      },
      "permutation_drop": {
        "mean": -0.03076171875,
        "lower95": -0.05419921875,
        "upper95": -0.00537109375,
        "n_seeds": 8
      },
      "reset_drop": {
        "mean": 0.11328125,
        "lower95": 0.09228515625,
        "upper95": 0.1337890625,
        "n_seeds": 8
      },
      "holm_signflip_p": 0.1640625,
      "pass": false
    },
    "binding": {
      "probe_gain": {
        "mean": -0.00732421875,
        "lower95": -0.02099609375,
        "upper95": 0.00537109375,
        "n_seeds": 8
      },
      "vs_readout": {
        "mean": -0.00732421875,
        "lower95": -0.02099609375,
        "upper95": 0.00537109375,
        "n_seeds": 8
      },
      "vs_shuffled": {
        "mean": -0.01904296875,
        "lower95": -0.033203125,
        "upper95": -0.00390625,
        "n_seeds": 8
      },
      "accuracy": {
        "mean": 0.72705078125,
        "lower95": 0.6884765625,
        "upper95": 0.76025390625,
        "n_seeds": 8
      },
      "probe": {
        "mean": 0.7138671875,
        "lower95": 0.7001953125,
        "upper95": 0.7294921875,
        "n_seeds": 8
      },
      "frozen_gain": {
        "mean": 0.27197265625,
        "lower95": 0.2021484375,
        "upper95": 0.357421875,
        "n_seeds": 8
      },
      "ood_probe": {
        "mean": 0.70849609375,
        "lower95": 0.68505859375,
        "upper95": 0.73583984375,
        "n_seeds": 8
      },
      "permutation_drop": {
        "mean": 0.00390625,
        "lower95": -0.01611328125,
        "upper95": 0.0244140625,
        "n_seeds": 8
      },
      "reset_drop": {
        "mean": 0.20263671875,
        "lower95": 0.1923828125,
        "upper95": 0.212890625,
        "n_seeds": 8
      },
      "holm_signflip_p": 0.546875,
      "pass": false
    }
  },
  "representation_pass": false,
  "retention": {
    "updates": {
      "conditions": {
        "selected": {
          "acquisition": {
            "mean": 0.30097656250000004,
            "lower95": 0.29023437500000004,
            "upper95": 0.31210937499999997,
            "n_seeds": 8
          },
          "final_accuracy": {
            "mean": 0.29121093750000004,
            "lower95": 0.27578125,
            "upper95": 0.30859619140624994,
            "n_seeds": 8
          },
          "forgetting": {
            "mean": 0.1388671875,
            "lower95": 0.12109130859375,
            "upper95": 0.1533203125,
            "n_seeds": 8
          },
          "worst_forgetting": {
            "mean": 0.349609375,
            "lower95": 0.310546875,
            "upper95": 0.38671875,
            "n_seeds": 8
          },
          "train_seconds": {
            "mean": 2.3968236625187274,
            "lower95": 1.5559248125300655,
            "upper95": 3.6330264593901576,
            "n_seeds": 8
          },
          "updates": {
            "mean": 1920.0,
            "lower95": 1920.0,
            "upper95": 1920.0,
            "n_seeds": 8
          }
        },
        "reference": {
          "acquisition": {
            "mean": 0.29990234375,
            "lower95": 0.28935546875,
            "upper95": 0.31025390625,
            "n_seeds": 8
          },
          "final_accuracy": {
            "mean": 0.28837890624999996,
            "lower95": 0.27138671875,
            "upper95": 0.3072265625,
            "n_seeds": 8
          },
          "forgetting": {
            "mean": 0.14111328125,
            "lower95": 0.1224609375,
            "upper95": 0.15576416015624997,
            "n_seeds": 8
          },
          "worst_forgetting": {
            "mean": 0.345703125,
            "lower95": 0.33203125,
            "upper95": 0.357421875,
            "n_seeds": 8
          },
          "train_seconds": {
            "mean": 2.4194139875016845,
            "lower95": 1.548725899989222,
            "upper95": 3.6945842443903527,
            "n_seeds": 8
          },
          "updates": {
            "mean": 1920.0,
            "lower95": 1920.0,
            "upper95": 1920.0,
            "n_seeds": 8
          }
        },
        "no_hidden": {
          "acquisition": {
            "mean": 0.30283203125,
            "lower95": 0.28876953125,
            "upper95": 0.3169921875,
            "n_seeds": 8
          },
          "final_accuracy": {
            "mean": 0.29072265625,
            "lower95": 0.2736328125,
            "upper95": 0.30869140624999997,
            "n_seeds": 8
          },
          "forgetting": {
            "mean": 0.1419921875,
            "lower95": 0.123828125,
            "upper95": 0.15986328125,
            "n_seeds": 8
          },
          "worst_forgetting": {
            "mean": 0.359375,
            "lower95": 0.326171875,
            "upper95": 0.3828125,
            "n_seeds": 8
          },
          "train_seconds": {
            "mean": 2.231817024981865,
            "lower95": 1.408913824974661,
            "upper95": 3.40810068747669,
            "n_seeds": 8
          },
          "updates": {
            "mean": 1920.0,
            "lower95": 1920.0,
            "upper95": 1920.0,
            "n_seeds": 8
          }
        },
        "no_consolidation": {
          "acquisition": {
            "mean": 0.30097656250000004,
            "lower95": 0.29023437500000004,
            "upper95": 0.31210937499999997,
            "n_seeds": 8
          },
          "final_accuracy": {
            "mean": 0.29121093750000004,
            "lower95": 0.27578125,
            "upper95": 0.30859619140624994,
            "n_seeds": 8
          },
          "forgetting": {
            "mean": 0.1388671875,
            "lower95": 0.12109130859375,
            "upper95": 0.1533203125,
            "n_seeds": 8
          },
          "worst_forgetting": {
            "mean": 0.349609375,
            "lower95": 0.310546875,
            "upper95": 0.38671875,
            "n_seeds": 8
          },
          "train_seconds": {
            "mean": 2.653126274999522,
            "lower95": 1.7448659874953591,
            "upper95": 3.7548528806323245,
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
      "forgetting_improvement": {
        "mean": 0.0022460937500000076,
        "lower95": 0.0005859375000000083,
        "upper95": 0.004199218750000002,
        "n_seeds": 8
      },
      "acquisition_difference": {
        "mean": 0.0010742187500000014,
        "lower95": -0.0013671874999999986,
        "upper95": 0.003417968750000007,
        "n_seeds": 8
      },
      "relative_improvement": false,
      "absolute_pass": false
    },
    "time": {
      "conditions": {
        "selected": {
          "acquisition": {
            "mean": 0.28359375,
            "lower95": 0.270703125,
            "upper95": 0.2970703125,
            "n_seeds": 8
          },
          "final_accuracy": {
            "mean": 0.27841796875,
            "lower95": 0.2587841796875,
            "upper95": 0.29873046875000003,
            "n_seeds": 8
          },
          "forgetting": {
            "mean": 0.12939453125,
            "lower95": 0.1044873046875,
            "upper95": 0.1515625,
            "n_seeds": 8
          },
          "worst_forgetting": {
            "mean": 0.296875,
            "lower95": 0.265625,
            "upper95": 0.328125,
            "n_seeds": 8
          },
          "train_seconds": {
            "mean": 1.553436537491507,
            "lower95": 1.5163630178123184,
            "upper95": 1.5945190874845139,
            "n_seeds": 8
          },
          "updates": {
            "mean": 1391.0,
            "lower95": 914.625,
            "upper95": 1725.503125,
            "n_seeds": 8
          }
        },
        "reference": {
          "acquisition": {
            "mean": 0.27197265625,
            "lower95": 0.2607421875,
            "upper95": 0.2845703125,
            "n_seeds": 8
          },
          "final_accuracy": {
            "mean": 0.27119140625000004,
            "lower95": 0.255078125,
            "upper95": 0.289453125,
            "n_seeds": 8
          },
          "forgetting": {
            "mean": 0.11298828124999999,
            "lower95": 0.08828125,
            "upper95": 0.1337890625,
            "n_seeds": 8
          },
          "worst_forgetting": {
            "mean": 0.291015625,
            "lower95": 0.240234375,
            "upper95": 0.3359375,
            "n_seeds": 8
          },
          "train_seconds": {
            "mean": 1.581783337502202,
            "lower95": 1.5411444540649881,
            "upper95": 1.6252938875077234,
            "n_seeds": 8
          },
          "updates": {
            "mean": 1258.625,
            "lower95": 819.25,
            "upper95": 1698.625,
            "n_seeds": 8
          }
        },
        "no_hidden": {
          "acquisition": {
            "mean": 0.28662109375,
            "lower95": 0.26943359375,
            "upper95": 0.3029296875,
            "n_seeds": 8
          },
          "final_accuracy": {
            "mean": 0.2837890625,
            "lower95": 0.2623046875,
            "upper95": 0.306640625,
            "n_seeds": 8
          },
          "forgetting": {
            "mean": 0.12138671875000001,
            "lower95": 0.09843750000000001,
            "upper95": 0.147265625,
            "n_seeds": 8
          },
          "worst_forgetting": {
            "mean": 0.310546875,
            "lower95": 0.26953125,
            "upper95": 0.34765625,
            "n_seeds": 8
          },
          "train_seconds": {
            "mean": 1.513885800004573,
            "lower95": 1.4347623875037243,
            "upper95": 1.593920572819161,
            "n_seeds": 8
          },
          "updates": {
            "mean": 1291.375,
            "lower95": 828.375,
            "upper95": 1754.875,
            "n_seeds": 8
          }
        },
        "no_consolidation": {
          "acquisition": {
            "mean": 0.28271484375,
            "lower95": 0.2654296875,
            "upper95": 0.29921875,
            "n_seeds": 8
          },
          "final_accuracy": {
            "mean": 0.2783203125,
            "lower95": 0.26044921875000004,
            "upper95": 0.29599609375,
            "n_seeds": 8
          },
          "forgetting": {
            "mean": 0.12783203125,
            "lower95": 0.105078125,
            "upper95": 0.15088134765624997,
            "n_seeds": 8
          },
          "worst_forgetting": {
            "mean": 0.30859375,
            "lower95": 0.263671875,
            "upper95": 0.357421875,
            "n_seeds": 8
          },
          "train_seconds": {
            "mean": 1.5785540875185688,
            "lower95": 1.5241421250302665,
            "upper95": 1.6328008912569656,
            "n_seeds": 8
          },
          "updates": {
            "mean": 1243.25,
            "lower95": 768.5,
            "upper95": 1717.25,
            "n_seeds": 8
          }
        }
      },
      "forgetting_improvement": {
        "mean": -0.016406249999999997,
        "lower95": -0.035156249999999986,
        "upper95": 0.0023437500000000003,
        "n_seeds": 8
      },
      "acquisition_difference": {
        "mean": 0.011621093749999999,
        "lower95": 0.0033203124999999847,
        "upper95": 0.020898437500000006,
        "n_seeds": 8
      },
      "relative_improvement": false,
      "absolute_pass": false
    }
  },
  "retention_pass": false,
  "checkpoint_continuation": true,
  "resources": {
    "peak_rss_bytes": 47656960,
    "peak_commit_or_vms_bytes": 66334720,
    "os_process_peak_working_set_bytes": 48631808,
    "os_process_peak_pagefile_commit_bytes": 67096576,
    "samples": 9409,
    "interval_seconds": 0.01,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 288.3385411000054,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  },
  "verdict": "PIVOT_NO_SCALE"
}
```
