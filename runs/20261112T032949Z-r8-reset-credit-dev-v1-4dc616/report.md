# Registered reset-credit DEV

No architecture change or fresh confirmation.

{
  "summary": {
    "equal_updates": {
      "strict_binding": {
        "arm_means": {
          "multi_gated": {
            "id_direct": 0.826171875,
            "id_probe": 0.8528645833333334,
            "ood_direct": 0.8287760416666666,
            "ood_probe": 0.8502604166666666
          },
          "multi_gated_shuffle": {
            "id_direct": 0.529296875,
            "id_probe": 0.5475260416666666,
            "ood_direct": 0.5240885416666666,
            "ood_probe": 0.5690104166666666
          },
          "multi_gated_zero": {
            "id_direct": 0.6295572916666666,
            "id_probe": 0.623046875,
            "ood_direct": 0.6725260416666666,
            "ood_probe": 0.673828125
          },
          "multi_gated_readout_only": {
            "id_direct": 0.6295572916666666,
            "id_probe": 0.623046875,
            "ood_direct": 0.6725260416666666,
            "ood_probe": 0.673828125
          }
        },
        "contrasts": {
          "vs_own_reset": {
            "mean": -0.016927083333333332,
            "lower95": -0.05859375,
            "upper95": 0.046875,
            "exact_two_sided_p": 0.75,
            "seed_clusters": 3,
            "effects": [
              0.046875,
              -0.05859375,
              -0.0390625
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_shuffle": {
            "mean": 0.3053385416666667,
            "lower95": 0.15625,
            "upper95": 0.404296875,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.404296875,
              0.15625,
              0.35546875
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_readout": {
            "mean": 0.22981770833333334,
            "lower95": 0.140625,
            "upper95": 0.28125,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.28125,
              0.140625,
              0.267578125
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          }
        }
      },
      "xor": {
        "arm_means": {
          "multi_gated": {
            "id_direct": 0.9934895833333334,
            "id_probe": 0.9954427083333334,
            "ood_direct": 0.8665364583333334,
            "ood_probe": 0.900390625
          },
          "multi_gated_shuffle": {
            "id_direct": 0.9401041666666666,
            "id_probe": 0.955078125,
            "ood_direct": 0.78125,
            "ood_probe": 0.859375
          },
          "multi_gated_zero": {
            "id_direct": 0.8274739583333334,
            "id_probe": 0.8326822916666666,
            "ood_direct": 0.7154947916666666,
            "ood_probe": 0.6861979166666666
          },
          "multi_gated_readout_only": {
            "id_direct": 0.8274739583333334,
            "id_probe": 0.8326822916666666,
            "ood_direct": 0.7154947916666666,
            "ood_probe": 0.6861979166666666
          }
        },
        "contrasts": {
          "vs_own_reset": {
            "mean": 0.028645833333333332,
            "lower95": 0.01171875,
            "upper95": 0.060546875,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.013671875,
              0.01171875,
              0.060546875
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_shuffle": {
            "mean": 0.040364583333333336,
            "lower95": 0.009765625,
            "upper95": 0.0859375,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.0859375,
              0.025390625,
              0.009765625
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_readout": {
            "mean": 0.16276041666666666,
            "lower95": 0.1171875,
            "upper95": 0.197265625,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.1171875,
              0.173828125,
              0.197265625
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          }
        }
      },
      "distractor_binding": {
        "arm_means": {
          "multi_gated": {
            "id_direct": 0.556640625,
            "id_probe": 0.6341145833333334,
            "ood_direct": 0.5631510416666666,
            "ood_probe": 0.6276041666666666
          },
          "multi_gated_shuffle": {
            "id_direct": 0.5110677083333334,
            "id_probe": 0.4921875,
            "ood_direct": 0.501953125,
            "ood_probe": 0.5149739583333334
          },
          "multi_gated_zero": {
            "id_direct": 0.5266927083333334,
            "id_probe": 0.5384114583333334,
            "ood_direct": 0.5162760416666666,
            "ood_probe": 0.5559895833333334
          },
          "multi_gated_readout_only": {
            "id_direct": 0.5266927083333334,
            "id_probe": 0.5384114583333334,
            "ood_direct": 0.5162760416666666,
            "ood_probe": 0.5559895833333334
          }
        },
        "contrasts": {
          "vs_own_reset": {
            "mean": 0.064453125,
            "lower95": 0.001953125,
            "upper95": 0.109375,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.08203125,
              0.001953125,
              0.109375
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_shuffle": {
            "mean": 0.14192708333333334,
            "lower95": 0.109375,
            "upper95": 0.201171875,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.109375,
              0.115234375,
              0.201171875
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_readout": {
            "mean": 0.095703125,
            "lower95": 0.0234375,
            "upper95": 0.17578125,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.0234375,
              0.087890625,
              0.17578125
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          }
        }
      },
      "order": {
        "arm_means": {
          "multi_gated": {
            "id_direct": 0.99609375,
            "id_probe": 1.0,
            "ood_direct": 1.0,
            "ood_probe": 1.0
          },
          "multi_gated_shuffle": {
            "id_direct": 0.9173177083333334,
            "id_probe": 0.9869791666666666,
            "ood_direct": 0.9557291666666666,
            "ood_probe": 0.9876302083333334
          },
          "multi_gated_zero": {
            "id_direct": 0.9173177083333334,
            "id_probe": 0.9368489583333334,
            "ood_direct": 0.9680989583333334,
            "ood_probe": 0.9479166666666666
          },
          "multi_gated_readout_only": {
            "id_direct": 0.9173177083333334,
            "id_probe": 0.9368489583333334,
            "ood_direct": 0.9680989583333334,
            "ood_probe": 0.9479166666666666
          }
        },
        "contrasts": {
          "vs_own_reset": {
            "mean": 0.0,
            "lower95": 0.0,
            "upper95": 0.0,
            "exact_two_sided_p": 1.0,
            "seed_clusters": 3,
            "effects": [
              0.0,
              0.0,
              0.0
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_shuffle": {
            "mean": 0.013020833333333334,
            "lower95": 0.0,
            "upper95": 0.03125,
            "exact_two_sided_p": 0.5,
            "seed_clusters": 3,
            "effects": [
              0.03125,
              0.0078125,
              0.0
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_readout": {
            "mean": 0.06315104166666667,
            "lower95": 0.03125,
            "upper95": 0.111328125,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.03125,
              0.111328125,
              0.046875
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          }
        }
      }
    },
    "equal_cpu": {
      "strict_binding": {
        "arm_means": {
          "multi_gated": {
            "id_direct": 0.5377604166666666,
            "id_probe": 0.62890625,
            "ood_direct": 0.5130208333333334,
            "ood_probe": 0.6686197916666666
          },
          "multi_gated_shuffle": {
            "id_direct": 0.5123697916666666,
            "id_probe": 0.6022135416666666,
            "ood_direct": 0.501953125,
            "ood_probe": 0.5983072916666666
          },
          "multi_gated_zero": {
            "id_direct": 0.5520833333333334,
            "id_probe": 0.623046875,
            "ood_direct": 0.5904947916666666,
            "ood_probe": 0.673828125
          },
          "multi_gated_readout_only": {
            "id_direct": 0.6087239583333334,
            "id_probe": 0.623046875,
            "ood_direct": 0.6158854166666666,
            "ood_probe": 0.673828125
          }
        },
        "contrasts": {
          "vs_own_reset": {
            "mean": -0.044270833333333336,
            "lower95": -0.078125,
            "upper95": -0.025390625,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              -0.078125,
              -0.029296875,
              -0.025390625
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_shuffle": {
            "mean": 0.026692708333333332,
            "lower95": 0.013671875,
            "upper95": 0.048828125,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.013671875,
              0.048828125,
              0.017578125
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_readout": {
            "mean": 0.005859375,
            "lower95": -0.0390625,
            "upper95": 0.060546875,
            "exact_two_sided_p": 1.0,
            "seed_clusters": 3,
            "effects": [
              -0.0390625,
              -0.00390625,
              0.060546875
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          }
        }
      },
      "xor": {
        "arm_means": {
          "multi_gated": {
            "id_direct": 0.865234375,
            "id_probe": 0.93359375,
            "ood_direct": 0.7513020833333334,
            "ood_probe": 0.806640625
          },
          "multi_gated_shuffle": {
            "id_direct": 0.6764322916666666,
            "id_probe": 0.8372395833333334,
            "ood_direct": 0.6002604166666666,
            "ood_probe": 0.7083333333333334
          },
          "multi_gated_zero": {
            "id_direct": 0.6868489583333334,
            "id_probe": 0.8326822916666666,
            "ood_direct": 0.6067708333333334,
            "ood_probe": 0.6861979166666666
          },
          "multi_gated_readout_only": {
            "id_direct": 0.6907552083333334,
            "id_probe": 0.8326822916666666,
            "ood_direct": 0.61328125,
            "ood_probe": 0.6861979166666666
          }
        },
        "contrasts": {
          "vs_own_reset": {
            "mean": 0.08463541666666667,
            "lower95": 0.001953125,
            "upper95": 0.169921875,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.001953125,
              0.08203125,
              0.169921875
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_shuffle": {
            "mean": 0.09635416666666667,
            "lower95": 0.060546875,
            "upper95": 0.130859375,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.09765625,
              0.130859375,
              0.060546875
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_readout": {
            "mean": 0.10091145833333333,
            "lower95": 0.072265625,
            "upper95": 0.138671875,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.072265625,
              0.138671875,
              0.091796875
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          }
        }
      },
      "distractor_binding": {
        "arm_means": {
          "multi_gated": {
            "id_direct": 0.5065104166666666,
            "id_probe": 0.5397135416666666,
            "ood_direct": 0.5188802083333334,
            "ood_probe": 0.5579427083333334
          },
          "multi_gated_shuffle": {
            "id_direct": 0.5052083333333334,
            "id_probe": 0.5299479166666666,
            "ood_direct": 0.5143229166666666,
            "ood_probe": 0.552734375
          },
          "multi_gated_zero": {
            "id_direct": 0.5390625,
            "id_probe": 0.5384114583333334,
            "ood_direct": 0.521484375,
            "ood_probe": 0.5559895833333334
          },
          "multi_gated_readout_only": {
            "id_direct": 0.5442708333333334,
            "id_probe": 0.5384114583333334,
            "ood_direct": 0.5247395833333334,
            "ood_probe": 0.5559895833333334
          }
        },
        "contrasts": {
          "vs_own_reset": {
            "mean": 0.0006510416666666666,
            "lower95": -0.03515625,
            "upper95": 0.02734375,
            "exact_two_sided_p": 1.0,
            "seed_clusters": 3,
            "effects": [
              0.02734375,
              -0.03515625,
              0.009765625
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_shuffle": {
            "mean": 0.009765625,
            "lower95": -0.00390625,
            "upper95": 0.03515625,
            "exact_two_sided_p": 1.0,
            "seed_clusters": 3,
            "effects": [
              0.03515625,
              -0.001953125,
              -0.00390625
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_readout": {
            "mean": 0.0013020833333333333,
            "lower95": -0.01953125,
            "upper95": 0.01953125,
            "exact_two_sided_p": 1.0,
            "seed_clusters": 3,
            "effects": [
              -0.01953125,
              0.01953125,
              0.00390625
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          }
        }
      },
      "order": {
        "arm_means": {
          "multi_gated": {
            "id_direct": 0.9609375,
            "id_probe": 0.9921875,
            "ood_direct": 0.9993489583333334,
            "ood_probe": 0.9967447916666666
          },
          "multi_gated_shuffle": {
            "id_direct": 0.8736979166666666,
            "id_probe": 0.9661458333333334,
            "ood_direct": 0.9563802083333334,
            "ood_probe": 0.9615885416666666
          },
          "multi_gated_zero": {
            "id_direct": 0.8932291666666666,
            "id_probe": 0.9368489583333334,
            "ood_direct": 0.966796875,
            "ood_probe": 0.9479166666666666
          },
          "multi_gated_readout_only": {
            "id_direct": 0.8782552083333334,
            "id_probe": 0.9368489583333334,
            "ood_direct": 0.9694010416666666,
            "ood_probe": 0.9479166666666666
          }
        },
        "contrasts": {
          "vs_own_reset": {
            "mean": -0.005859375,
            "lower95": -0.017578125,
            "upper95": 0.001953125,
            "exact_two_sided_p": 0.75,
            "seed_clusters": 3,
            "effects": [
              0.001953125,
              -0.017578125,
              -0.001953125
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_shuffle": {
            "mean": 0.026041666666666668,
            "lower95": 0.005859375,
            "upper95": 0.041015625,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.005859375,
              0.041015625,
              0.03125
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          },
          "vs_readout": {
            "mean": 0.055338541666666664,
            "lower95": 0.03125,
            "upper95": 0.08984375,
            "exact_two_sided_p": 0.25,
            "seed_clusters": 3,
            "effects": [
              0.03125,
              0.08984375,
              0.044921875
            ],
            "scope": "Reused exploratory DEV seeds, not confirmation"
          }
        }
      }
    },
    "primary_holm_p": {
      "strict_binding/vs_shuffle": 1.0,
      "strict_binding/vs_readout": 1.0,
      "xor/vs_own_reset": 1.0,
      "xor/vs_shuffle": 1.0,
      "xor/vs_readout": 1.0,
      "distractor_binding/vs_own_reset": 1.0,
      "distractor_binding/vs_shuffle": 1.0,
      "distractor_binding/vs_readout": 1.0,
      "strict_binding/vs_own_reset": 1.0
    },
    "master_p1_certified": false,
    "reason": "Three reused DEV seeds and explicit reused own-reset baseline; no independent confirmation"
  },
  "row_count": 96,
  "completed_jobs": 48,
  "error": null,
  "resources": {
    "peak_rss_bytes": 222253056,
    "peak_commit_or_vms_bytes": 196063232,
    "os_process_peak_working_set_bytes": 61202432,
    "os_process_peak_pagefile_commit_bytes": 78725120,
    "samples": 7464,
    "interval_seconds": 0.05,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 440.13365969999904,
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
  "p0_test_command": [
    "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
    "-X",
    "utf8",
    "-m",
    "pytest",
    "-q",
    "-p",
    "no:cacheprovider",
    "tests/test_r8_stopped_reset.py",
    "tests/test_r8_matched_credit.py",
    "tests/test_r8_snapshot_adapter.py",
    "tests/test_research_gate_repairs.py",
    "--junitxml=C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-r8-matched-controls-20261112\\runs\\20261112T032949Z-r8-reset-credit-dev-v1-4dc616\\p0-tests.xml"
  ],
  "frozen_dataset_sha256": {
    "strict_binding-140011.npz": "0c9aaffa0a500f70b512fc83ed1ee733b0068b697c6a637eee7a6e10c7c7b04c",
    "strict_binding-140017.npz": "138fa3d570af2343bb5053c481b00a064c131c3db91d1d9afe51ccf7eb19c25e",
    "strict_binding-140023.npz": "4ac9bccafdb760de0b6c40f5002f652ca6cb7514e40999aa434cad1def058514",
    "xor-140011.npz": "87d67e91b7c41f40102934772c05ee6033787e4999b71b72f12094a4c4656c3b",
    "xor-140017.npz": "592d0b637d214ca4a949a2e826efc0bf491dcb82c54219af3d8b43e68f2171d3",
    "xor-140023.npz": "9b3dd9a538c5953c3aef72995e4073a32e1387bf682f2b6dd836de26c1382e22",
    "distractor_binding-140011.npz": "899accb761f5f1f2a6b6955ab9616b435d52b462b199a7e6da03d200dcee5910",
    "distractor_binding-140017.npz": "d24e6b33a85d04492971a47f130156d611c74ef8ec2c3028fb62168ca304172c",
    "distractor_binding-140023.npz": "d37f52d32968e19404e6a92daec323549b4ba25e03ec896e544a1b3fc06cac37",
    "order-140011.npz": "af17f404338d79803c1a37411c370dbd2cbbbbe373845c9c25a7bf8cec7b8f82",
    "order-140017.npz": "56170b7a8ad0cfa79e9b875468536d5d3d1568643ab2f5a2ed1737ea3cd28a68",
    "order-140023.npz": "137b9b47ce885fd03919f59408652f77ddeec52869bd471689377916f6f05205"
  },
  "base_run": "20261112T031057Z-r8-matched-credit-dev-v1-eec5ba",
  "main_sha_before": "e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541",
  "main_sha_after": "e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541",
  "forward_changed": false,
  "new_confirmation": false,
  "decision": "DEV_COMPLETE_NOT_MASTER_P1_GO"
}
