# Existing ALIF neuronstate readout observability

No sourceweight/dynamicschanges; analogchannel assumption explicit, localdiagnosticheads only.

{
  "run": "runs/20261109T173923Z-p2-neuron-state-observability-v1-73a8f0",
  "source_run": "runs/20261109T140620Z-p2-paired-score-diagnostic-v1-c5a357",
  "decision": "STOP_STATE_OBSERVABILITY_SCREEN_V1",
  "summary": {
    "offline_joint_full": {
      "filtered_spikes": {
        "ridge": 0.486328125,
        "local": 0.3865885416666666,
        "gain": 0.0
      },
      "last_spikes": {
        "ridge": 0.2755208333333334,
        "local": 0.26276041666666666,
        "gain": -0.123828125
      },
      "membrane_after_reset": {
        "ridge": 0.40781249999999997,
        "local": 0.2907552083333333,
        "gain": -0.09583333333333334
      },
      "adaptation": {
        "ridge": 0.5005208333333333,
        "local": 0.397265625,
        "gain": 0.010677083333333337
      },
      "relative_membrane": {
        "ridge": 0.4811197916666667,
        "local": 0.29934895833333336,
        "gain": -0.08723958333333331
      }
    },
    "offline_joint_frozen": {
      "filtered_spikes": {
        "ridge": 0.4540364583333334,
        "local": 0.35507812499999997,
        "gain": 0.0
      },
      "last_spikes": {
        "ridge": 0.2615885416666667,
        "local": 0.253125,
        "gain": -0.10195312500000002
      },
      "membrane_after_reset": {
        "ridge": 0.36367187500000003,
        "local": 0.25611979166666665,
        "gain": -0.09895833333333333
      },
      "adaptation": {
        "ridge": 0.47148437499999996,
        "local": 0.36302083333333335,
        "gain": 0.007942708333333326
      },
      "relative_membrane": {
        "ridge": 0.4455729166666667,
        "local": 0.26979166666666665,
        "gain": -0.08528645833333333
      }
    }
  },
  "informative_screen": {
    "offline_joint_full": {
      "filtered_spikes": false,
      "last_spikes": false,
      "membrane_after_reset": false,
      "adaptation": false,
      "relative_membrane": false
    },
    "offline_joint_frozen": {
      "filtered_spikes": false,
      "last_spikes": false,
      "membrane_after_reset": false,
      "adaptation": false,
      "relative_membrane": false
    }
  },
  "paired_effects": {
    "offline_joint_full": {
      "filtered_spikes": {
        "gain": {
          "mean": 0.0,
          "lower95": 0.0,
          "upper95": 0.0,
          "n_seeds": 3
        },
        "exploratory_only": true
      },
      "last_spikes": {
        "gain": {
          "mean": -0.123828125,
          "lower95": -0.1296875,
          "upper95": -0.11367187499999998,
          "n_seeds": 3
        },
        "exploratory_only": true
      },
      "membrane_after_reset": {
        "gain": {
          "mean": -0.09583333333333334,
          "lower95": -0.10625000000000001,
          "upper95": -0.08671875000000001,
          "n_seeds": 3
        },
        "exploratory_only": true
      },
      "adaptation": {
        "gain": {
          "mean": 0.010677083333333337,
          "lower95": 0.005859375,
          "upper95": 0.01874999999999999,
          "n_seeds": 3
        },
        "exploratory_only": true
      },
      "relative_membrane": {
        "gain": {
          "mean": -0.08723958333333331,
          "lower95": -0.10117187499999997,
          "upper95": -0.07773437499999997,
          "n_seeds": 3
        },
        "exploratory_only": true
      }
    },
    "offline_joint_frozen": {
      "filtered_spikes": {
        "gain": {
          "mean": 0.0,
          "lower95": 0.0,
          "upper95": 0.0,
          "n_seeds": 3
        },
        "exploratory_only": true
      },
      "last_spikes": {
        "gain": {
          "mean": -0.10195312500000002,
          "lower95": -0.11210937500000001,
          "upper95": -0.08476562500000001,
          "n_seeds": 3
        },
        "exploratory_only": true
      },
      "membrane_after_reset": {
        "gain": {
          "mean": -0.09895833333333333,
          "lower95": -0.10859374999999999,
          "upper95": -0.080078125,
          "n_seeds": 3
        },
        "exploratory_only": true
      },
      "adaptation": {
        "gain": {
          "mean": 0.007942708333333326,
          "lower95": 0.001171874999999989,
          "upper95": 0.01640625000000001,
          "n_seeds": 3
        },
        "exploratory_only": true
      },
      "relative_membrane": {
        "gain": {
          "mean": -0.08528645833333333,
          "lower95": -0.11289062500000002,
          "upper95": -0.05585937499999999,
          "n_seeds": 3
        },
        "exploratory_only": true
      }
    }
  },
  "measurements": [
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "view": "filtered_spikes",
      "ridge": 0.450390625,
      "local_output_head": 0.371875,
      "head_updates": 19200,
      "head_loss": 2.15433205480833,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "view": "last_spikes",
      "ridge": 0.257421875,
      "local_output_head": 0.262890625,
      "head_updates": 19200,
      "head_loss": 1.9736153982845364,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "view": "membrane_after_reset",
      "ridge": 0.354296875,
      "local_output_head": 0.26328125,
      "head_updates": 19200,
      "head_loss": 2.432059267906265,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "view": "adaptation",
      "ridge": 0.469140625,
      "local_output_head": 0.378125,
      "head_updates": 19200,
      "head_loss": 2.1400440755835164,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15217,
      "condition": "offline_joint_frozen",
      "view": "relative_membrane",
      "ridge": 0.44296875,
      "local_output_head": 0.258984375,
      "head_updates": 19200,
      "head_loss": 2.80047970688504,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "view": "filtered_spikes",
      "ridge": 0.48828125,
      "local_output_head": 0.39765625,
      "head_updates": 19200,
      "head_loss": 2.8878240993253046,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "view": "last_spikes",
      "ridge": 0.271875,
      "local_output_head": 0.26953125,
      "head_updates": 19200,
      "head_loss": 5.70619983017426,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "view": "membrane_after_reset",
      "ridge": 0.40078125,
      "local_output_head": 0.29140625,
      "head_updates": 19200,
      "head_loss": 2.193767868985696,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "view": "adaptation",
      "ridge": 0.503515625,
      "local_output_head": 0.405078125,
      "head_updates": 19200,
      "head_loss": 2.78451579177446,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15217,
      "condition": "offline_joint_full",
      "view": "relative_membrane",
      "ridge": 0.465625,
      "local_output_head": 0.296484375,
      "head_updates": 19200,
      "head_loss": 2.461872513866533,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "view": "filtered_spikes",
      "ridge": 0.4453125,
      "local_output_head": 0.357421875,
      "head_updates": 19200,
      "head_loss": 2.2157473670062315,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "view": "last_spikes",
      "ridge": 0.259765625,
      "local_output_head": 0.2453125,
      "head_updates": 19200,
      "head_loss": 1.985156391520495,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "view": "membrane_after_reset",
      "ridge": 0.37265625,
      "local_output_head": 0.24921875,
      "head_updates": 19200,
      "head_loss": 2.2886791616152053,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "view": "adaptation",
      "ridge": 0.458203125,
      "local_output_head": 0.35859375,
      "head_updates": 19200,
      "head_loss": 2.2381990931895683,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15227,
      "condition": "offline_joint_frozen",
      "view": "relative_membrane",
      "ridge": 0.43828125,
      "local_output_head": 0.2703125,
      "head_updates": 19200,
      "head_loss": 2.328803150616639,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "view": "filtered_spikes",
      "ridge": 0.46875,
      "local_output_head": 0.372265625,
      "head_updates": 19200,
      "head_loss": 2.2424053573684017,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "view": "last_spikes",
      "ridge": 0.283984375,
      "local_output_head": 0.25859375,
      "head_updates": 19200,
      "head_loss": 1.9523107825672537,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "view": "membrane_after_reset",
      "ridge": 0.425390625,
      "local_output_head": 0.277734375,
      "head_updates": 19200,
      "head_loss": 2.139029608899496,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "view": "adaptation",
      "ridge": 0.48359375,
      "local_output_head": 0.378125,
      "head_updates": 19200,
      "head_loss": 2.3400393568132922,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15227,
      "condition": "offline_joint_full",
      "view": "relative_membrane",
      "ridge": 0.505078125,
      "local_output_head": 0.289453125,
      "head_updates": 19200,
      "head_loss": 2.642765437208917,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "view": "filtered_spikes",
      "ridge": 0.46640625,
      "local_output_head": 0.3359375,
      "head_updates": 19200,
      "head_loss": 2.1081337223224836,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "view": "last_spikes",
      "ridge": 0.267578125,
      "local_output_head": 0.251171875,
      "head_updates": 19200,
      "head_loss": 1.9492015782415781,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "view": "membrane_after_reset",
      "ridge": 0.3640625,
      "local_output_head": 0.255859375,
      "head_updates": 19200,
      "head_loss": 1.958759825872343,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "view": "adaptation",
      "ridge": 0.487109375,
      "local_output_head": 0.35234375,
      "head_updates": 19200,
      "head_loss": 2.1003471536113265,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15241,
      "condition": "offline_joint_frozen",
      "view": "relative_membrane",
      "ridge": 0.45546875,
      "local_output_head": 0.280078125,
      "head_updates": 19200,
      "head_loss": 1.9314188680610709,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "view": "filtered_spikes",
      "ridge": 0.501953125,
      "local_output_head": 0.38984375,
      "head_updates": 19200,
      "head_loss": 2.1729025345529815,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "view": "last_spikes",
      "ridge": 0.270703125,
      "local_output_head": 0.26015625,
      "head_updates": 19200,
      "head_loss": 1.9460832380541668,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": false
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "view": "membrane_after_reset",
      "ridge": 0.397265625,
      "local_output_head": 0.303125,
      "head_updates": 19200,
      "head_loss": 1.9859769557799465,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "view": "adaptation",
      "ridge": 0.514453125,
      "local_output_head": 0.40859375,
      "head_updates": 19200,
      "head_loss": 2.1553557913099595,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    },
    {
      "seed": 15241,
      "condition": "offline_joint_full",
      "view": "relative_membrane",
      "ridge": 0.47265625,
      "local_output_head": 0.312109375,
      "head_updates": 19200,
      "head_loss": 1.9948508356670072,
      "source_state_unchanged": true,
      "feature_dimensions": 48,
      "analog_state_channel": true
    }
  ],
  "all_core_state_unchanged": true,
  "reference_forward_parity": true,
  "new_core_training": false,
  "deployed": false,
  "normal_chat_or_vision_proven": false,
  "retention_confirmed": false,
  "resources": {
    "peak_rss_bytes": 75153408,
    "peak_commit_or_vms_bytes": 93507584,
    "os_process_peak_working_set_bytes": 82665472,
    "os_process_peak_pagefile_commit_bytes": 100810752,
    "samples": 146,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 13.947471199993743,
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
    "Analogpresynapticstatechannel is an additionalassumption, not pureaxonspikecommunication or biologicalproof.",
    "Offlinejointsource hadfuturelabels, probesare diagnostic not deployable lifelongmodels.",
    "Three developmentseeds, no fullP2gate/confirmation or worddecoderfamilyreopening."
  ]
}
