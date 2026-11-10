# Read-only visual information diagnostic

No neurallearning/mainchange; auxiliaryclassifier labels are NOT modelanswers.

{
  "run": "runs/20261110T000339Z-visual-information-audit-v1-882afb",
  "source_run": "runs/20261109T234349Z-visual-word-learnability-control-v1-e51fe9",
  "frozen_source_run": "runs/20261109T222354Z-visual-word-readout-v1-a11424",
  "decision": "VISUAL_INFORMATION_DIAGNOSTIC_ONLY",
  "visual_information_rows": [
    {
      "family": "frozen32",
      "seed": 33113,
      "actual_BOS_firstword": 0.27083333333333337,
      "auxiliary_firstword_probe": 0.111328125,
      "auxiliary_semantic_probe": 0.10785590277777778,
      "auxiliary_shuffled_semantic_probe": 0.019097222222222224,
      "head_clip_fraction": 0.0028935185185185184,
      "actual_head_feature_rank": 47
    },
    {
      "family": "frozen32",
      "seed": 33119,
      "actual_BOS_firstword": 0.31510416666666663,
      "auxiliary_firstword_probe": 0.173828125,
      "auxiliary_semantic_probe": 0.050130208333333336,
      "auxiliary_shuffled_semantic_probe": 0.0013020833333333333,
      "head_clip_fraction": 0.000626929012345679,
      "actual_head_feature_rank": 44
    },
    {
      "family": "frozen32",
      "seed": 33133,
      "actual_BOS_firstword": 0.27604166666666663,
      "auxiliary_firstword_probe": 0.0954861111111111,
      "auxiliary_semantic_probe": 0.09114583333333333,
      "auxiliary_shuffled_semantic_probe": 0.0625,
      "head_clip_fraction": 0.0011815200617283949,
      "actual_head_feature_rank": 45
    },
    {
      "family": "bptt_control33",
      "seed": 33113,
      "actual_BOS_firstword": 0.376953125,
      "auxiliary_firstword_probe": 0.14539930555555555,
      "auxiliary_semantic_probe": 0.06597222222222221,
      "auxiliary_shuffled_semantic_probe": 0.06835937500000001,
      "head_clip_fraction": 0.01986882716049383,
      "actual_head_feature_rank": 45
    },
    {
      "family": "bptt_control33",
      "seed": 33119,
      "actual_BOS_firstword": 0.4344618055555555,
      "auxiliary_firstword_probe": 0.20551215277777776,
      "auxiliary_semantic_probe": 0.13151041666666669,
      "auxiliary_shuffled_semantic_probe": 0.12239583333333333,
      "head_clip_fraction": 0.011284722222222222,
      "actual_head_feature_rank": 46
    },
    {
      "family": "bptt_control33",
      "seed": 33133,
      "actual_BOS_firstword": 0.2530381944444444,
      "auxiliary_firstword_probe": 0.130859375,
      "auxiliary_semantic_probe": 0.1506076388888889,
      "auxiliary_shuffled_semantic_probe": 0.03797743055555555,
      "head_clip_fraction": 0.0012297453703703704,
      "actual_head_feature_rank": 43
    }
  ],
  "information_collapse_rows": [
    {
      "family": "frozen32",
      "stage": "input_tape",
      "dev_semantic_ceiling_mean": 0.999999999999997,
      "dev_firstword_ceiling_mean": 0.999999999999997,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "frozen32",
      "stage": "question",
      "dev_semantic_ceiling_mean": 0.46875,
      "dev_firstword_ceiling_mean": 0.53125,
      "image_pair_collapse_mean": 1.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": true
    },
    {
      "family": "frozen32",
      "stage": "rgb_question",
      "dev_semantic_ceiling_mean": 0.999999999999997,
      "dev_firstword_ceiling_mean": 0.999999999999997,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "frozen32",
      "stage": "prepared_state",
      "dev_semantic_ceiling_mean": 0.999999999999997,
      "dev_firstword_ceiling_mean": 0.999999999999997,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "frozen32",
      "stage": "visual_trace",
      "dev_semantic_ceiling_mean": 0.9999999999999972,
      "dev_firstword_ceiling_mean": 0.9999999999999972,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "frozen32",
      "stage": "bos_feature",
      "dev_semantic_ceiling_mean": 0.9999999999999972,
      "dev_firstword_ceiling_mean": 0.9999999999999972,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "frozen32",
      "stage": "head_feature",
      "dev_semantic_ceiling_mean": 0.9999999999999972,
      "dev_firstword_ceiling_mean": 0.9999999999999972,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "frozen32",
      "stage": "logits",
      "dev_semantic_ceiling_mean": 0.9999999999999972,
      "dev_firstword_ceiling_mean": 0.9999999999999972,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "bptt_control33",
      "stage": "input_tape",
      "dev_semantic_ceiling_mean": 0.999999999999997,
      "dev_firstword_ceiling_mean": 0.999999999999997,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "bptt_control33",
      "stage": "question",
      "dev_semantic_ceiling_mean": 0.46875,
      "dev_firstword_ceiling_mean": 0.53125,
      "image_pair_collapse_mean": 1.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": true
    },
    {
      "family": "bptt_control33",
      "stage": "rgb_question",
      "dev_semantic_ceiling_mean": 0.999999999999997,
      "dev_firstword_ceiling_mean": 0.999999999999997,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "bptt_control33",
      "stage": "prepared_state",
      "dev_semantic_ceiling_mean": 0.999999999999997,
      "dev_firstword_ceiling_mean": 0.999999999999997,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "bptt_control33",
      "stage": "visual_trace",
      "dev_semantic_ceiling_mean": 0.9999999999999972,
      "dev_firstword_ceiling_mean": 0.9999999999999972,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "bptt_control33",
      "stage": "bos_feature",
      "dev_semantic_ceiling_mean": 0.999999999999997,
      "dev_firstword_ceiling_mean": 0.999999999999997,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "bptt_control33",
      "stage": "head_feature",
      "dev_semantic_ceiling_mean": 0.999999999999997,
      "dev_firstword_ceiling_mean": 0.999999999999997,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    },
    {
      "family": "bptt_control33",
      "stage": "logits",
      "dev_semantic_ceiling_mean": 0.999999999999997,
      "dev_firstword_ceiling_mean": 0.999999999999997,
      "image_pair_collapse_mean": 0.0,
      "question_pair_collapse_mean": 0.0,
      "expected_missingmodality_branch": false
    }
  ],
  "diagnostic_flags": {
    "observed_component_information_collapse": true,
    "BOS_readout_training_gap": {
      "frozen32": false,
      "bptt_control33": false
    },
    "CONTROL_representation_gain": false
  },
  "firstword_probe_minus_realhead": {
    "frozen32": {
      "mean": -0.16044560185185183,
      "lower95": -0.1805555555555555,
      "upper95": -0.14127604166666663,
      "n_seeds": 3
    },
    "bptt_control33": {
      "mean": -0.19422743055555555,
      "lower95": -0.23155381944444445,
      "upper95": -0.12217881944444442,
      "n_seeds": 3
    }
  },
  "control_minus_frozen_semantic_probe": {
    "mean": 0.03298611111111111,
    "lower95": -0.04188368055555556,
    "upper95": 0.08138020833333334,
    "n_seeds": 3
  },
  "raw_input_references": {
    "33113": {
      "question": {
        "train_observed": {
          "macro": 1.0,
          "micro": 1.0,
          "by_kind": {
            "single_color": 1.0,
            "single_shape": 1.0,
            "object_count": 1.0,
            "left_color": 1.0
          },
          "by_language": {
            "vi": 1.0,
            "en": 1.0
          },
          "novel_macro": null
        },
        "development": {
          "macro": 0.75,
          "micro": 0.8888888888888888,
          "by_kind": {
            "single_color": 0.25,
            "single_shape": 0.75,
            "object_count": 1.0,
            "left_color": 1.0
          },
          "by_language": {
            "vi": 0.75,
            "en": 0.75
          },
          "novel_macro": 0.75
        },
        "fit_array_bytes": 194560
      },
      "rgb_question": {
        "train_observed": {
          "macro": 0.9942407579504353,
          "micro": 0.994140625,
          "by_kind": {
            "single_color": 0.9949931498318596,
            "single_shape": 0.9846153846153847,
            "object_count": 1.0,
            "left_color": 0.9973544973544973
          },
          "by_language": {
            "vi": 0.9896335896335897,
            "en": 0.9988479262672811
          },
          "novel_macro": null
        },
        "development": {
          "macro": 0.49631076388888884,
          "micro": 0.5682870370370371,
          "by_kind": {
            "single_color": 0.29166666666666663,
            "single_shape": 0.375,
            "object_count": 0.4921875,
            "left_color": 0.8263888888888888
          },
          "by_language": {
            "vi": 0.5611979166666667,
            "en": 0.43142361111111116
          },
          "novel_macro": 0.408234126984127
        },
        "fit_array_bytes": 6704128,
        "shuffled_development": {
          "macro": 0.0954861111111111,
          "micro": 0.08217592592592593,
          "by_kind": {
            "single_color": 0.1875,
            "single_shape": 0.041666666666666664,
            "object_count": 0.052083333333333336,
            "left_color": 0.10069444444444445
          },
          "by_language": {
            "vi": 0.10460069444444445,
            "en": 0.08637152777777776
          },
          "novel_macro": 0.10255456349206349
        }
      }
    },
    "33119": {
      "question": {
        "train_observed": {
          "macro": 1.0,
          "micro": 1.0,
          "by_kind": {
            "single_color": 1.0,
            "single_shape": 1.0,
            "object_count": 1.0,
            "left_color": 1.0
          },
          "by_language": {
            "vi": 1.0,
            "en": 1.0
          },
          "novel_macro": null
        },
        "development": {
          "macro": 0.75,
          "micro": 0.8888888888888888,
          "by_kind": {
            "single_color": 0.25,
            "single_shape": 0.75,
            "object_count": 1.0,
            "left_color": 1.0
          },
          "by_language": {
            "vi": 0.75,
            "en": 0.75
          },
          "novel_macro": 0.75
        },
        "fit_array_bytes": 194560
      },
      "rgb_question": {
        "train_observed": {
          "macro": 0.9974279975026467,
          "micro": 0.9973958333333334,
          "by_kind": {
            "single_color": 0.9973262032085561,
            "single_shape": 1.0,
            "object_count": 0.9923857868020305,
            "left_color": 1.0
          },
          "by_language": {
            "vi": 1.0,
            "en": 0.9948559950052933
          },
          "novel_macro": null
        },
        "development": {
          "macro": 0.4876302083333333,
          "micro": 0.5532407407407407,
          "by_kind": {
            "single_color": 0.3229166666666667,
            "single_shape": 0.34375,
            "object_count": 0.4609375,
            "left_color": 0.8229166666666666
          },
          "by_language": {
            "vi": 0.5638020833333333,
            "en": 0.41145833333333337
          },
          "novel_macro": 0.40736607142857145
        },
        "fit_array_bytes": 6704128,
        "shuffled_development": {
          "macro": 0.07335069444444445,
          "micro": 0.07291666666666667,
          "by_kind": {
            "single_color": 0.041666666666666664,
            "single_shape": 0.08333333333333333,
            "object_count": 0.026041666666666664,
            "left_color": 0.1423611111111111
          },
          "by_language": {
            "vi": 0.07118055555555555,
            "en": 0.07552083333333334
          },
          "novel_macro": 0.06758432539682539
        }
      }
    },
    "33133": {
      "question": {
        "train_observed": {
          "macro": 1.0,
          "micro": 1.0,
          "by_kind": {
            "single_color": 1.0,
            "single_shape": 1.0,
            "object_count": 1.0,
            "left_color": 1.0
          },
          "by_language": {
            "vi": 1.0,
            "en": 1.0
          },
          "novel_macro": null
        },
        "development": {
          "macro": 0.75,
          "micro": 0.8888888888888888,
          "by_kind": {
            "single_color": 0.25,
            "single_shape": 0.75,
            "object_count": 1.0,
            "left_color": 1.0
          },
          "by_language": {
            "vi": 0.75,
            "en": 0.75
          },
          "novel_macro": 0.75
        },
        "fit_array_bytes": 194560
      },
      "rgb_question": {
        "train_observed": {
          "macro": 0.9961316215008618,
          "micro": 0.99609375,
          "by_kind": {
            "single_color": 0.9871580649508153,
            "single_shape": 0.9973684210526316,
            "object_count": 1.0,
            "left_color": 1.0
          },
          "by_language": {
            "vi": 0.99502567394095,
            "en": 0.9972375690607735
          },
          "novel_macro": null
        },
        "development": {
          "macro": 0.5225694444444444,
          "micro": 0.5729166666666666,
          "by_kind": {
            "single_color": 0.38541666666666663,
            "single_shape": 0.40625,
            "object_count": 0.46875,
            "left_color": 0.8298611111111112
          },
          "by_language": {
            "vi": 0.5989583333333333,
            "en": 0.4461805555555556
          },
          "novel_macro": 0.46713789682539686
        },
        "fit_array_bytes": 6704128,
        "shuffled_development": {
          "macro": 0.04513888888888889,
          "micro": 0.05092592592592592,
          "by_kind": {
            "single_color": 0.041666666666666664,
            "single_shape": 0.020833333333333332,
            "object_count": 0.041666666666666664,
            "left_color": 0.0763888888888889
          },
          "by_language": {
            "vi": 0.047309027777777776,
            "en": 0.04296875
          },
          "novel_macro": 0.057787698412698416
        }
      }
    }
  },
  "forward_observations": 10368,
  "forward_ticks": 373248,
  "dataset": "data/candidates/20261109T221711Z-visual-word-dataset-v1-f93af5/dataset.json",
  "dataset_sha256": "0fb04abefe67451952a2d73634fded9c86243d5874356d68299ff047deecf6a2",
  "same_training_exposure_multisets": true,
  "all_source_weights_learningstate_rng_unchanged": true,
  "neural_training": false,
  "neural_update_steps": 0,
  "probe_fitting": "Disposable closed-form classifier on observedTRAINlabels; not neuralupdate, not inferenceanswers.",
  "torch_imported": false,
  "deployed": false,
  "normal_chat_or_vision_proven": false,
  "main_local_learning_success": false,
  "proposal30_approved": false,
  "resources": {
    "peak_rss_bytes": 133386240,
    "peak_commit_or_vms_bytes": 150056960,
    "os_process_peak_working_set_bytes": 134443008,
    "os_process_peak_pagefile_commit_bytes": 150884352,
    "samples": 428,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 47.60268759999599,
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
  "seconds": 48.00510770000983,
  "limits": [
    "Historical6sources/3pairedseeds, inspecteddev and ownedshapes; no finaltest/generalvision/lifelong proof.",
    "Firstwordprobe comparison onlyFIRSTWORD; semanticprobe is classification, never verbal generation.",
    "Question-only branch lacksRGB bydesign; branchceilings do not bound fullcombinedstate/futurewords. Noexactcollisions doesnotprovelearnability.",
    "Offline ridge differentcompute/order than onlinehead, same1536labels; not a deployedlearner or matched-cost qualitygain.",
    "GlobalCONTROLrepresentation gain, ifany, cannot prove permittedlocalrepresentation. STOP32/33/earlierfamilies retained."
  ]
}
