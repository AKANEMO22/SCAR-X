# ISOLATED visual ALIF-BPTT learnability CONTROL

CONTROL_LEARNABILITY_NOT_DEMONSTRATED_IN_BUDGET

No controlweights/algorithm imported into SCAR-X main.

{
  "run": "runs/20261109T234349Z-visual-word-learnability-control-v1-e51fe9",
  "source_run": "runs/20261109T222354Z-visual-word-readout-v1-a11424",
  "preflight_run": "runs/20261109T233806Z-visual-word-control-preflight-v1-805213",
  "decision": "ISOLATED_VISUAL_WORD_CONTROL_ONLY",
  "interpretation": "CONTROL_LEARNABILITY_NOT_DEMONSTRATED_IN_BUDGET",
  "control_only": true,
  "summary": {
    "bptt_control": {
      "macro_exact": 0.2567274305555556,
      "novel_composition_macro": 0.2744295634920635,
      "by_kind": {
        "single_color": 0.3576388888888889,
        "single_shape": 0.052083333333333336,
        "object_count": 0.15538194444444442,
        "left_color": 0.4618055555555555
      }
    },
    "frozen_full": {
      "macro_exact": 0.18880208333333334,
      "novel_composition_macro": 0.16939484126984128,
      "by_kind": {
        "single_color": 0.2986111111111111,
        "single_shape": 0.06597222222222222,
        "object_count": 0.057291666666666664,
        "left_color": 0.3333333333333333
      }
    },
    "frozen_no_image": {
      "macro_exact": 0.1796875,
      "novel_composition_macro": 0.1830357142857143,
      "by_kind": {
        "single_color": 0.3333333333333333,
        "single_shape": 0.0,
        "object_count": 0.09375,
        "left_color": 0.2916666666666667
      }
    },
    "frozen_no_question": {
      "macro_exact": 0.21498842592592593,
      "novel_composition_macro": 0.2203207671957672,
      "by_kind": {
        "single_color": 0.34722222222222215,
        "single_shape": 0.041666666666666664,
        "object_count": 0.34375,
        "left_color": 0.1273148148148148
      }
    }
  },
  "paired_trained_control_gains": {
    "frozen_full": {
      "mean": 0.06792534722222225,
      "lower95": 0.0030381944444444475,
      "upper95": 0.10763888888888891,
      "n_seeds": 3
    }
  },
  "paired_causal_drops": {
    "blank_image": {
      "mean": 0.13953993055555558,
      "lower95": 0.08637152777777779,
      "upper95": 0.21809895833333337,
      "n_seeds": 3
    },
    "blank_question": {
      "mean": 0.20724826388888892,
      "lower95": 0.08723958333333333,
      "upper95": 0.2719184027777778,
      "n_seeds": 3
    },
    "swapped_image": {
      "mean": 0.06401909722222225,
      "lower95": 0.05403645833333337,
      "upper95": 0.07074652777777779,
      "n_seeds": 3
    }
  },
  "gates": {
    "dev65_eachseed": false,
    "kind50_eachseed": false,
    "novel60_eachseed": false,
    "gain20_allpositive_lowerCIpositive": false,
    "each_causal_mean_drop20": false
  },
  "dataset": "data/candidates/20261109T221711Z-visual-word-dataset-v1-f93af5/dataset.json",
  "dataset_sha256": "0fb04abefe67451952a2d73634fded9c86243d5874356d68299ff047deecf6a2",
  "reused_frozen_trials": 9,
  "all_cold_exact": true,
  "resources": {
    "peak_rss_bytes": 639852544,
    "peak_commit_or_vms_bytes": 1025499136,
    "os_process_peak_working_set_bytes": 59092992,
    "os_process_peak_pagefile_commit_bytes": 46931968,
    "samples": 1258,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 97.54744540000684,
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
  "seconds": 97.82031539999298,
  "deployed": false,
  "weights_imported_into_scarx": false,
  "main_local_learning_success": false,
  "normal_photo_qa_proven": false,
  "retention_confirmed": false,
  "proposal30_approved": false,
  "limits": [
    "Threepairedseeds on inspectedsyntheticdev; not freshfinaltest or normalvision/chat/lifelong.",
    "BiasedGLOBAL/TEMPORAL surrogateBPTT onlycontrol; headlocal unchanged. Extra graphcompute/allrowsAdam not matchedcost.",
    "Frozenmissingmodality baselines are reused context, not matched BPTTmissingmodality ablations. Mainstilllocal/ALIF, STOP32retained."
  ]
}
