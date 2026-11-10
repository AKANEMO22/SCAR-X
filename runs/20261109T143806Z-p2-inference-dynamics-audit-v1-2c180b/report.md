# P2 inference dynamics diagnostic

Immutable core/oldhead, exploratory new inference views; no training, selection or deployment.

{
  "run": "runs/20261109T143806Z-p2-inference-dynamics-audit-v1-2c180b",
  "source_run": "runs/20261109T140620Z-p2-paired-score-diagnostic-v1-c5a357",
  "decision": "INFERENCE_DYNAMICS_DIAGNOSTIC_ONLY",
  "summary": {
    "offline_joint_full": {
      "0.0": {
        "1": {
          "6": {
            "direct": 0.2885416666666667,
            "probe": 0.6041666666666666
          },
          "12": {
            "direct": 0.27734375,
            "probe": 0.5015625
          }
        },
        "8": {
          "6": {
            "direct": 0.2885416666666667,
            "probe": 0.6041666666666666
          },
          "12": {
            "direct": 0.27734375,
            "probe": 0.5015625
          }
        }
      },
      "0.1": {
        "1": {
          "6": {
            "direct": 0.29583333333333334,
            "probe": 0.53671875
          },
          "12": {
            "direct": 0.29062499999999997,
            "probe": 0.45286458333333335
          }
        },
        "8": {
          "6": {
            "direct": 0.29440104166666664,
            "probe": 0.6652343749999999
          },
          "12": {
            "direct": 0.28489583333333335,
            "probe": 0.5635416666666667
          }
        }
      },
      "0.4": {
        "1": {
          "6": {
            "direct": 0.3111979166666667,
            "probe": 0.423046875
          },
          "12": {
            "direct": 0.300390625,
            "probe": 0.39869791666666665
          }
        },
        "8": {
          "6": {
            "direct": 0.3140625,
            "probe": 0.5446614583333332
          },
          "12": {
            "direct": 0.30494791666666665,
            "probe": 0.4811197916666667
          }
        }
      }
    },
    "offline_joint_frozen": {
      "0.0": {
        "1": {
          "6": {
            "direct": 0.26158854166666673,
            "probe": 0.6053385416666667
          },
          "12": {
            "direct": 0.24817708333333335,
            "probe": 0.5026041666666666
          }
        },
        "8": {
          "6": {
            "direct": 0.26158854166666673,
            "probe": 0.6053385416666667
          },
          "12": {
            "direct": 0.24817708333333335,
            "probe": 0.5026041666666666
          }
        }
      },
      "0.1": {
        "1": {
          "6": {
            "direct": 0.2651041666666667,
            "probe": 0.5424479166666667
          },
          "12": {
            "direct": 0.2546875,
            "probe": 0.44557291666666665
          }
        },
        "8": {
          "6": {
            "direct": 0.26015625000000003,
            "probe": 0.6630208333333333
          },
          "12": {
            "direct": 0.240625,
            "probe": 0.5520833333333334
          }
        }
      },
      "0.4": {
        "1": {
          "6": {
            "direct": 0.26705729166666664,
            "probe": 0.419140625
          },
          "12": {
            "direct": 0.2643229166666667,
            "probe": 0.3837239583333334
          }
        },
        "8": {
          "6": {
            "direct": 0.27122395833333335,
            "probe": 0.530859375
          },
          "12": {
            "direct": 0.26718749999999997,
            "probe": 0.45221354166666666
          }
        }
      }
    }
  },
  "measurements": [
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.256640625,
      "refit_probe": 0.5734375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.13959210000029998
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.263671875,
      "refit_probe": 0.498828125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.25259039999946253
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.256640625,
      "refit_probe": 0.5734375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.3195506000047317
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.263671875,
      "refit_probe": 0.498828125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.4316356999988784
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.256640625,
      "refit_probe": 0.515234375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.5114044000001741
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.2703125,
      "refit_probe": 0.443359375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.6298165000043809
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.25703125,
      "refit_probe": 0.623046875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.0391100000051665
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.26328125,
      "refit_probe": 0.53046875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.5493186000021524
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.27109375,
      "refit_probe": 0.415625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.6327673999985564
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.275390625,
      "refit_probe": 0.3953125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.7542414999988978
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.28515625,
      "refit_probe": 0.524609375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 2.0308005000042613
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.28828125,
      "refit_probe": 0.455078125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 2.7552399999985937
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.25390625,
      "refit_probe": 0.57421875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.08336569999664789
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.27578125,
      "refit_probe": 0.487890625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.1627976999952807
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.25390625,
      "refit_probe": 0.57421875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.2117727999939234
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.27578125,
      "refit_probe": 0.487890625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.29130979999899864
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.25546875,
      "refit_probe": 0.515234375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.3545642999961274
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.283203125,
      "refit_probe": 0.450390625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.4873654999973951
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.254296875,
      "refit_probe": 0.6203125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.7785140999985742
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.275,
      "refit_probe": 0.551171875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.5874642999988282
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.28046875,
      "refit_probe": 0.4140625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.667069499999343
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.2921875,
      "refit_probe": 0.394140625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.757520299994212
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.255078125,
      "refit_probe": 0.51796875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 2.031682599998021
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.288671875,
      "refit_probe": 0.480859375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 2.5204808999988018
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.300390625,
      "refit_probe": 0.58515625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.09156440000515431
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.253125,
      "refit_probe": 0.470703125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.1644598999992013
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.300390625,
      "refit_probe": 0.58515625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.21475150000333088
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.253125,
      "refit_probe": 0.470703125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.2840845000027912
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.29375,
      "refit_probe": 0.530078125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.3366124000021955
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.259765625,
      "refit_probe": 0.425390625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.4207620000015595
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.29375,
      "refit_probe": 0.643359375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.663855800004967
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.242578125,
      "refit_probe": 0.53125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.4780806999988272
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.27578125,
      "refit_probe": 0.403515625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.5288567000025068
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.266015625,
      "refit_probe": 0.370703125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.6516912000006414
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.286328125,
      "refit_probe": 0.508984375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.91263560000516
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.27421875,
      "refit_probe": 0.44296875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 2.3912303000033717
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.3140625,
      "refit_probe": 0.622265625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.08747060000314377
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.308984375,
      "refit_probe": 0.519140625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.1592205000051763
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.3140625,
      "refit_probe": 0.622265625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.2044746000028681
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.308984375,
      "refit_probe": 0.519140625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.28125079999881564
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.33671875,
      "refit_probe": 0.545703125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.3353330000027199
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.32265625,
      "refit_probe": 0.442578125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.4220714000039152
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.332421875,
      "refit_probe": 0.687890625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.6651864000014029
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.326171875,
      "refit_probe": 0.572265625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.1070649000030244
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.32421875,
      "refit_probe": 0.425390625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.1662085999996634
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.30859375,
      "refit_probe": 0.396484375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.239375099998142
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.36328125,
      "refit_probe": 0.551171875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.468643100000918
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.3265625,
      "refit_probe": 0.466015625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.9485028999988572
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.227734375,
      "refit_probe": 0.657421875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.0893004999961704
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.227734375,
      "refit_probe": 0.53828125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.15374480000173207
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.227734375,
      "refit_probe": 0.657421875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.20455390000279294
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.227734375,
      "refit_probe": 0.53828125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.26820479999878444
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.244921875,
      "refit_probe": 0.58203125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.33314789999712957
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.233984375,
      "refit_probe": 0.46796875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.42176869999821065
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.2296875,
      "refit_probe": 0.72265625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.6664604999969015
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.216015625,
      "refit_probe": 0.59453125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.1149786000023596
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.254296875,
      "refit_probe": 0.43828125,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.1681556999974418
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.2515625,
      "refit_probe": 0.38515625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.3096081999974558
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.2421875,
      "refit_probe": 0.558984375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.5433185000001686
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.2390625,
      "refit_probe": 0.45859375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.9898455000002286
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.29765625,
      "refit_probe": 0.616015625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.09136020000005374
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.247265625,
      "refit_probe": 0.49765625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.15779990000010002
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.29765625,
      "refit_probe": 0.616015625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.21105149999493733
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.0,
      "requested_rollouts": 8,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.247265625,
      "refit_probe": 0.49765625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.2807881999979145
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.2953125,
      "refit_probe": 0.54921875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.33408309999504127
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.266015625,
      "refit_probe": 0.465625,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.4210056999945664
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.296484375,
      "refit_probe": 0.6875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 0.6670920999968075
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.1,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.253515625,
      "refit_probe": 0.5671875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.1409650999994483
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 6,
      "direct_original_head": 0.32890625,
      "refit_probe": 0.4296875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.1972174999973504
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 1,
      "effective_rollouts": 1,
      "prefix_ticks": 12,
      "direct_original_head": 0.300390625,
      "refit_probe": 0.40546875,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.2817553999993834
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 6,
      "direct_original_head": 0.323828125,
      "refit_probe": 0.56484375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 1.5252949999994598
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "temperature": 0.4,
      "requested_rollouts": 8,
      "effective_rollouts": 8,
      "prefix_ticks": 12,
      "direct_original_head": 0.299609375,
      "refit_probe": 0.496484375,
      "core_state_unchanged": true,
      "elapsed_model_seconds": 2.1942798999953084
    }
  ],
  "all_state_unchanged": true,
  "reference_parity": true,
  "new_training": false,
  "deployed": false,
  "normal_chat_or_vision_proven": false,
  "retention_confirmed": false,
  "resources": {
    "peak_rss_bytes": 86339584,
    "peak_commit_or_vms_bytes": 105033728,
    "os_process_peak_working_set_bytes": 88498176,
    "os_process_peak_pagefile_commit_bytes": 106893312,
    "samples": 198,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 13.939739899993583,
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
    "Changed inference dynamics vs training, no fresh validated deployment/selection.",
    "Refit head is diagnostic, never a runtime or hidden skill claim.",
    "Offline joint checkpoints learned futurelabels; cannot be used as a lifelong starting point."
  ]
}
