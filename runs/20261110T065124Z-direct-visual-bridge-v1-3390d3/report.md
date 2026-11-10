# Directvisual37 research

STOP_DIRECT_VISUAL_BRIDGE_V1

Actualneuralwords on same32forward; no main/P1/P2/normalvisionclaim.

{
  "run": "runs/20261110T065124Z-direct-visual-bridge-v1-3390d3",
  "decision": "STOP_DIRECT_VISUAL_BRIDGE_V1",
  "summary": {
    "correct": {
      "macro_exact": 0.25101273148148145,
      "macro_train": 0.5153356481481481,
      "novel_composition_macro": 0.25764715608465605,
      "by_kind": {
        "single_color": 0.38888888888888884,
        "single_shape": 0.017361111111111112,
        "object_count": 0.13368055555555555,
        "left_color": 0.4641203703703704
      },
      "probe_gain": -0.0925925925925926
    },
    "shuffled": {
      "macro_exact": 0.21317997685185186,
      "macro_train": 0.4016927083333333,
      "novel_composition_macro": 0.2158151455026455,
      "by_kind": {
        "single_color": 0.25,
        "single_shape": 0.0,
        "object_count": 0.24045138888888892,
        "left_color": 0.36226851851851855
      },
      "probe_gain": -0.08449074074074074
    },
    "zero": {
      "macro_exact": 0.150462962962963,
      "macro_train": 0.5418113425925926,
      "novel_composition_macro": 0.14699074074074073,
      "by_kind": {
        "single_color": 0.25,
        "single_shape": 0.041666666666666664,
        "object_count": 0.0,
        "left_color": 0.3101851851851852
      },
      "probe_gain": 0.0
    },
    "frozen": {
      "macro_exact": 0.150462962962963,
      "macro_train": 0.5418113425925926,
      "novel_composition_macro": 0.14699074074074073,
      "by_kind": {
        "single_color": 0.25,
        "single_shape": 0.041666666666666664,
        "object_count": 0.0,
        "left_color": 0.3101851851851852
      },
      "probe_gain": 0.0
    },
    "no_image": {
      "macro_exact": 0.13541666666666666,
      "macro_train": 0.4270833333333333,
      "novel_composition_macro": 0.14583333333333334,
      "by_kind": {
        "single_color": 0.20833333333333334,
        "single_shape": 0.041666666666666664,
        "object_count": 0.0,
        "left_color": 0.2916666666666667
      },
      "probe_gain": -0.01157407407407407
    },
    "no_question": {
      "macro_exact": 0.17650462962962962,
      "macro_train": 0.18836805555555555,
      "novel_composition_macro": 0.17253637566137567,
      "by_kind": {
        "single_color": 0.32638888888888884,
        "single_shape": 0.0,
        "object_count": 0.13194444444444445,
        "left_color": 0.2476851851851852
      },
      "probe_gain": -0.026234567901234556
    },
    "bptt_control": {
      "macro_exact": 0.2630208333333333,
      "macro_train": 0.5379050925925927,
      "novel_composition_macro": 0.27046130952380953,
      "by_kind": {
        "single_color": 0.43402777777777785,
        "single_shape": 0.05555555555555556,
        "object_count": 0.16319444444444445,
        "left_color": 0.3993055555555555
      },
      "probe_gain": 0.0030864197530864244
    }
  },
  "gates": {
    "macro65_eachseed": false,
    "kind50_eachseed": false,
    "novel60_eachseed": false,
    "missingmodality_gains20_allpositive_lower95": false,
    "frozen_gain10_allpositive_lower95": true,
    "causal_drop20_each": false,
    "resethead_probe_gain5_lower95": false
  },
  "paired_gains": {
    "frozen": {
      "mean": 0.1005497685185185,
      "lower95": 0.078125,
      "upper95": 0.13324652777777776,
      "n_seeds": 3
    },
    "shuffled": {
      "mean": 0.03783275462962962,
      "lower95": -0.06163194444444442,
      "upper95": 0.11002604166666667,
      "n_seeds": 3
    },
    "no_image": {
      "mean": 0.11559606481481481,
      "lower95": 0.07161458333333331,
      "upper95": 0.19704861111111108,
      "n_seeds": 3
    },
    "no_question": {
      "mean": 0.07450810185185185,
      "lower95": 0.052083333333333315,
      "upper95": 0.09114583333333331,
      "n_seeds": 3
    }
  },
  "paired_causal_drops": {
    "blank_image": {
      "mean": 0.19632523148148148,
      "lower95": 0.09375,
      "upper95": 0.2907986111111111,
      "n_seeds": 3
    },
    "blank_question": {
      "mean": 0.12080439814814814,
      "lower95": 0.06510416666666666,
      "upper95": 0.1519097222222222,
      "n_seeds": 3
    },
    "swapped_image": {
      "mean": 0.021846064814814808,
      "lower95": -0.046875,
      "upper95": 0.1189236111111111,
      "n_seeds": 3
    }
  },
  "paired_resethead_probe_gain": {
    "mean": -0.0925925925925926,
    "lower95": -0.11458333333333333,
    "upper95": -0.07291666666666666,
    "n_seeds": 3
  },
  "all_cold_exact": true,
  "zero_frozen_exact": true,
  "proposal30_approved": true,
  "dataset": "data/candidates/20261109T221711Z-visual-word-dataset-v1-f93af5/dataset.json",
  "dataset_sha256": "0fb04abefe67451952a2d73634fded9c86243d5874356d68299ff047deecf6a2",
  "deployed": false,
  "main_weights_changed": false,
  "normal_photo_qa_proven": false,
  "normal_chat_or_vision_proven": false,
  "retention_proven": false,
  "resources": {
    "peak_rss_bytes": 643469312,
    "peak_commit_or_vms_bytes": 1042432000,
    "os_process_peak_working_set_bytes": 57692160,
    "os_process_peak_pagefile_commit_bytes": 44490752,
    "samples": 4198,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 329.84961420000764,
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
  "seconds": 330.1422261999978,
  "preflight_run": "runs/20261110T064337Z-direct-visual-preflight-v1-927d70",
  "limits": [
    "3freshseeds on inspectedhistorical864devgeometriccases, not freshfinal/naturalphotos/OCR/lifelong.",
    "No teacher/Qwenanswers used; actualAGY32wordings reused, Qwen4Btext-only not visionbaseline.",
    "STOP32/33/35/36gates retained. No automaticencoder/cache/LR/epoch/modelscale/mainimport from results."
  ]
}
