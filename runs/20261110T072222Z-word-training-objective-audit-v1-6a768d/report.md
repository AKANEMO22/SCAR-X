# Wordtraining objective audit38

No neuraltraining; FREE/GOLDPREFIX distinct. Corpusstructure/potentialdirections not modelanswers or qualitysuccess.

{
  "run": "runs/20261110T072222Z-word-training-objective-audit-v1-6a768d",
  "source_run": "runs/20261110T062226Z-direct-word-bridge-v1-a6a908",
  "decision": "WORD_TRAINING_OBJECTIVE_DIAGNOSTIC_ONLY",
  "corpus_structure": {
    "targets": 8,
    "unique_first_words": 8,
    "total_positions_uniform_templates": 35,
    "later_positions": 27,
    "later_fraction": 0.7714285714285715,
    "punctuation_or_EOS_positions": 16,
    "nonempty_prefix_determines_next_fraction": 1.0,
    "PREFIX_STRUCTURE_SHORTCUT": true,
    "scope": "Finiteownedtemplateproperty only; not neuralanswers, skill or deployedlookup."
  },
  "word_objective_rows": [
    {
      "seed": 36313,
      "method": "bptt_control",
      "split": "train",
      "free_exact": 1.0,
      "tf_first": 1.0,
      "tf_later": 1.0,
      "wrong_first": 0.0,
      "wrong_later": 0.5833333333333333,
      "wrong_first_drop": 1.0,
      "wrong_later_drop": 0.41666666666666674,
      "zero_first": 0.125,
      "zero_later": 0.8958333333333333,
      "matched_later_exposure_gap": 0.0,
      "free_later_positions_reached": 216,
      "free_gold_later_positions_censored": 0,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36313,
      "method": "bptt_control",
      "split": "development",
      "free_exact": 0.3125,
      "tf_first": 0.5,
      "tf_later": 0.8333333333333333,
      "wrong_first": 0.0,
      "wrong_later": 0.5833333333333333,
      "wrong_first_drop": 0.5,
      "wrong_later_drop": 0.25,
      "zero_first": 0.125,
      "zero_later": 0.8958333333333333,
      "matched_later_exposure_gap": 0.29557291666666663,
      "free_later_positions_reached": 180,
      "free_gold_later_positions_censored": 36,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36313,
      "method": "correct",
      "split": "train",
      "free_exact": 0.984375,
      "tf_first": 0.984375,
      "tf_later": 1.0,
      "wrong_first": 0.0,
      "wrong_later": 0.6458333333333333,
      "wrong_first_drop": 0.984375,
      "wrong_later_drop": 0.35416666666666674,
      "zero_first": 0.125,
      "zero_later": 0.8645833333333333,
      "matched_later_exposure_gap": 0.015625,
      "free_later_positions_reached": 214,
      "free_gold_later_positions_censored": 2,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": true
      },
      "potential_first_later_cosine": -0.09728516649852582
    },
    {
      "seed": 36313,
      "method": "correct",
      "split": "development",
      "free_exact": 0.328125,
      "tf_first": 0.53125,
      "tf_later": 0.828125,
      "wrong_first": 0.0,
      "wrong_later": 0.6458333333333333,
      "wrong_first_drop": 0.53125,
      "wrong_later_drop": 0.18229166666666674,
      "zero_first": 0.125,
      "zero_later": 0.8645833333333333,
      "matched_later_exposure_gap": 0.29427083333333337,
      "free_later_positions_reached": 176,
      "free_gold_later_positions_censored": 40,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": false
      },
      "potential_first_later_cosine": 0.10270016916495543
    },
    {
      "seed": 36313,
      "method": "frozen",
      "split": "train",
      "free_exact": 0.96875,
      "tf_first": 1.0,
      "tf_later": 0.9921875,
      "wrong_first": 0.0,
      "wrong_later": 0.4583333333333333,
      "wrong_first_drop": 1.0,
      "wrong_later_drop": 0.5338541666666667,
      "zero_first": 0.125,
      "zero_later": 0.8958333333333333,
      "matched_later_exposure_gap": 0.005208333333333259,
      "free_later_positions_reached": 214,
      "free_gold_later_positions_censored": 2,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36313,
      "method": "frozen",
      "split": "development",
      "free_exact": 0.15625,
      "tf_first": 0.4375,
      "tf_later": 0.6848958333333333,
      "wrong_first": 0.0,
      "wrong_later": 0.4583333333333333,
      "wrong_first_drop": 0.4375,
      "wrong_later_drop": 0.22656249999999994,
      "zero_first": 0.125,
      "zero_later": 0.8958333333333333,
      "matched_later_exposure_gap": 0.32812499999999994,
      "free_later_positions_reached": 167,
      "free_gold_later_positions_censored": 49,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36313,
      "method": "shuffled",
      "split": "train",
      "free_exact": 0.640625,
      "tf_first": 0.953125,
      "tf_later": 0.8932291666666666,
      "wrong_first": 0.0,
      "wrong_later": 0.5520833333333333,
      "wrong_first_drop": 0.953125,
      "wrong_later_drop": 0.34114583333333337,
      "zero_first": 0.125,
      "zero_later": 0.8020833333333333,
      "matched_later_exposure_gap": 0.14453125,
      "free_later_positions_reached": 205,
      "free_gold_later_positions_censored": 11,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36313,
      "method": "shuffled",
      "split": "development",
      "free_exact": 0.15625,
      "tf_first": 0.421875,
      "tf_later": 0.71875,
      "wrong_first": 0.0,
      "wrong_later": 0.5520833333333333,
      "wrong_first_drop": 0.421875,
      "wrong_later_drop": 0.16666666666666674,
      "zero_first": 0.125,
      "zero_later": 0.8020833333333333,
      "matched_later_exposure_gap": 0.2734375,
      "free_later_positions_reached": 180,
      "free_gold_later_positions_censored": 36,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36313,
      "method": "zero",
      "split": "train",
      "free_exact": 0.96875,
      "tf_first": 1.0,
      "tf_later": 0.9921875,
      "wrong_first": 0.0,
      "wrong_later": 0.4583333333333333,
      "wrong_first_drop": 1.0,
      "wrong_later_drop": 0.5338541666666667,
      "zero_first": 0.125,
      "zero_later": 0.8958333333333333,
      "matched_later_exposure_gap": 0.005208333333333259,
      "free_later_positions_reached": 214,
      "free_gold_later_positions_censored": 2,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36313,
      "method": "zero",
      "split": "development",
      "free_exact": 0.15625,
      "tf_first": 0.4375,
      "tf_later": 0.6848958333333333,
      "wrong_first": 0.0,
      "wrong_later": 0.4583333333333333,
      "wrong_first_drop": 0.4375,
      "wrong_later_drop": 0.22656249999999994,
      "zero_first": 0.125,
      "zero_later": 0.8958333333333333,
      "matched_later_exposure_gap": 0.32812499999999994,
      "free_later_positions_reached": 167,
      "free_gold_later_positions_censored": 49,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36319,
      "method": "bptt_control",
      "split": "train",
      "free_exact": 1.0,
      "tf_first": 1.0,
      "tf_later": 1.0,
      "wrong_first": 0.0,
      "wrong_later": 0.7395833333333333,
      "wrong_first_drop": 1.0,
      "wrong_later_drop": 0.26041666666666674,
      "zero_first": 0.125,
      "zero_later": 0.6145833333333333,
      "matched_later_exposure_gap": 0.0,
      "free_later_positions_reached": 216,
      "free_gold_later_positions_censored": 0,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36319,
      "method": "bptt_control",
      "split": "development",
      "free_exact": 0.34375,
      "tf_first": 0.5,
      "tf_later": 0.8177083333333333,
      "wrong_first": 0.0,
      "wrong_later": 0.7395833333333333,
      "wrong_first_drop": 0.5,
      "wrong_later_drop": 0.078125,
      "zero_first": 0.125,
      "zero_later": 0.6145833333333333,
      "matched_later_exposure_gap": 0.3020833333333333,
      "free_later_positions_reached": 184,
      "free_gold_later_positions_censored": 32,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36319,
      "method": "correct",
      "split": "train",
      "free_exact": 1.0,
      "tf_first": 1.0,
      "tf_later": 1.0,
      "wrong_first": 0.0,
      "wrong_later": 0.7708333333333333,
      "wrong_first_drop": 1.0,
      "wrong_later_drop": 0.22916666666666674,
      "zero_first": 0.125,
      "zero_later": 0.6770833333333333,
      "matched_later_exposure_gap": 0.0,
      "free_later_positions_reached": 216,
      "free_gold_later_positions_censored": 0,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": true
      },
      "potential_first_later_cosine": -0.08417098871943669
    },
    {
      "seed": 36319,
      "method": "correct",
      "split": "development",
      "free_exact": 0.296875,
      "tf_first": 0.546875,
      "tf_later": 0.7747395833333333,
      "wrong_first": 0.0,
      "wrong_later": 0.7708333333333333,
      "wrong_first_drop": 0.546875,
      "wrong_later_drop": 0.00390625,
      "zero_first": 0.125,
      "zero_later": 0.6770833333333333,
      "matched_later_exposure_gap": 0.26692708333333337,
      "free_later_positions_reached": 192,
      "free_gold_later_positions_censored": 24,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": false
      },
      "potential_first_later_cosine": 0.11356065691327406
    },
    {
      "seed": 36319,
      "method": "frozen",
      "split": "train",
      "free_exact": 0.78125,
      "tf_first": 0.96875,
      "tf_later": 0.9270833333333333,
      "wrong_first": 0.0,
      "wrong_later": 0.5208333333333333,
      "wrong_first_drop": 0.96875,
      "wrong_later_drop": 0.40625,
      "zero_first": 0.125,
      "zero_later": 0.8020833333333333,
      "matched_later_exposure_gap": 0.04036458333333337,
      "free_later_positions_reached": 207,
      "free_gold_later_positions_censored": 9,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36319,
      "method": "frozen",
      "split": "development",
      "free_exact": 0.125,
      "tf_first": 0.390625,
      "tf_later": 0.6458333333333333,
      "wrong_first": 0.0,
      "wrong_later": 0.5208333333333333,
      "wrong_first_drop": 0.390625,
      "wrong_later_drop": 0.125,
      "zero_first": 0.125,
      "zero_later": 0.8020833333333333,
      "matched_later_exposure_gap": 0.20182291666666663,
      "free_later_positions_reached": 192,
      "free_gold_later_positions_censored": 24,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36319,
      "method": "shuffled",
      "split": "train",
      "free_exact": 0.8125,
      "tf_first": 0.953125,
      "tf_later": 0.9440104166666666,
      "wrong_first": 0.0,
      "wrong_later": 0.6770833333333333,
      "wrong_first_drop": 0.953125,
      "wrong_later_drop": 0.26692708333333337,
      "zero_first": 0.125,
      "zero_later": 0.5208333333333333,
      "matched_later_exposure_gap": 0.06510416666666663,
      "free_later_positions_reached": 202,
      "free_gold_later_positions_censored": 14,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36319,
      "method": "shuffled",
      "split": "development",
      "free_exact": 0.15625,
      "tf_first": 0.390625,
      "tf_later": 0.6614583333333333,
      "wrong_first": 0.0,
      "wrong_later": 0.6770833333333333,
      "wrong_first_drop": 0.390625,
      "wrong_later_drop": -0.015625,
      "zero_first": 0.125,
      "zero_later": 0.5208333333333333,
      "matched_later_exposure_gap": 0.2109375,
      "free_later_positions_reached": 169,
      "free_gold_later_positions_censored": 47,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36319,
      "method": "zero",
      "split": "train",
      "free_exact": 0.78125,
      "tf_first": 0.96875,
      "tf_later": 0.9270833333333333,
      "wrong_first": 0.0,
      "wrong_later": 0.5208333333333333,
      "wrong_first_drop": 0.96875,
      "wrong_later_drop": 0.40625,
      "zero_first": 0.125,
      "zero_later": 0.8020833333333333,
      "matched_later_exposure_gap": 0.04036458333333337,
      "free_later_positions_reached": 207,
      "free_gold_later_positions_censored": 9,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36319,
      "method": "zero",
      "split": "development",
      "free_exact": 0.125,
      "tf_first": 0.390625,
      "tf_later": 0.6458333333333333,
      "wrong_first": 0.0,
      "wrong_later": 0.5208333333333333,
      "wrong_first_drop": 0.390625,
      "wrong_later_drop": 0.125,
      "zero_first": 0.125,
      "zero_later": 0.8020833333333333,
      "matched_later_exposure_gap": 0.20182291666666663,
      "free_later_positions_reached": 192,
      "free_gold_later_positions_censored": 24,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36341,
      "method": "bptt_control",
      "split": "train",
      "free_exact": 1.0,
      "tf_first": 1.0,
      "tf_later": 1.0,
      "wrong_first": 0.0,
      "wrong_later": 0.7708333333333333,
      "wrong_first_drop": 1.0,
      "wrong_later_drop": 0.22916666666666674,
      "zero_first": 0.125,
      "zero_later": 0.9375,
      "matched_later_exposure_gap": 0.0,
      "free_later_positions_reached": 216,
      "free_gold_later_positions_censored": 0,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36341,
      "method": "bptt_control",
      "split": "development",
      "free_exact": 0.375,
      "tf_first": 0.578125,
      "tf_later": 0.8619791666666666,
      "wrong_first": 0.0,
      "wrong_later": 0.7708333333333333,
      "wrong_first_drop": 0.578125,
      "wrong_later_drop": 0.09114583333333337,
      "zero_first": 0.125,
      "zero_later": 0.9375,
      "matched_later_exposure_gap": 0.30338541666666674,
      "free_later_positions_reached": 191,
      "free_gold_later_positions_censored": 25,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36341,
      "method": "correct",
      "split": "train",
      "free_exact": 1.0,
      "tf_first": 1.0,
      "tf_later": 1.0,
      "wrong_first": 0.0,
      "wrong_later": 0.7395833333333333,
      "wrong_first_drop": 1.0,
      "wrong_later_drop": 0.26041666666666674,
      "zero_first": 0.125,
      "zero_later": 0.9375,
      "matched_later_exposure_gap": 0.0,
      "free_later_positions_reached": 216,
      "free_gold_later_positions_censored": 0,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": true
      },
      "potential_first_later_cosine": -0.04017610785019207
    },
    {
      "seed": 36341,
      "method": "correct",
      "split": "development",
      "free_exact": 0.359375,
      "tf_first": 0.5625,
      "tf_later": 0.86328125,
      "wrong_first": 0.0,
      "wrong_later": 0.7395833333333333,
      "wrong_first_drop": 0.5625,
      "wrong_later_drop": 0.12369791666666674,
      "zero_first": 0.125,
      "zero_later": 0.9375,
      "matched_later_exposure_gap": 0.31901041666666663,
      "free_later_positions_reached": 191,
      "free_gold_later_positions_censored": 25,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": false
      },
      "potential_first_later_cosine": 0.1578743505753594
    },
    {
      "seed": 36341,
      "method": "frozen",
      "split": "train",
      "free_exact": 0.921875,
      "tf_first": 0.984375,
      "tf_later": 0.9765625,
      "wrong_first": 0.0,
      "wrong_later": 0.7395833333333333,
      "wrong_first_drop": 0.984375,
      "wrong_later_drop": 0.23697916666666674,
      "zero_first": 0.125,
      "zero_later": 0.78125,
      "matched_later_exposure_gap": 0.02994791666666663,
      "free_later_positions_reached": 214,
      "free_gold_later_positions_censored": 2,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36341,
      "method": "frozen",
      "split": "development",
      "free_exact": 0.171875,
      "tf_first": 0.46875,
      "tf_later": 0.7057291666666666,
      "wrong_first": 0.0,
      "wrong_later": 0.7395833333333333,
      "wrong_first_drop": 0.46875,
      "wrong_later_drop": -0.03385416666666663,
      "zero_first": 0.125,
      "zero_later": 0.78125,
      "matched_later_exposure_gap": 0.30338541666666663,
      "free_later_positions_reached": 178,
      "free_gold_later_positions_censored": 38,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36341,
      "method": "shuffled",
      "split": "train",
      "free_exact": 0.84375,
      "tf_first": 0.984375,
      "tf_later": 0.9609375,
      "wrong_first": 0.0,
      "wrong_later": 0.7708333333333333,
      "wrong_first_drop": 0.984375,
      "wrong_later_drop": 0.19010416666666674,
      "zero_first": 0.125,
      "zero_later": 0.75,
      "matched_later_exposure_gap": 0.0703125,
      "free_later_positions_reached": 205,
      "free_gold_later_positions_censored": 11,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36341,
      "method": "shuffled",
      "split": "development",
      "free_exact": 0.1875,
      "tf_first": 0.4375,
      "tf_later": 0.76171875,
      "wrong_first": 0.0,
      "wrong_later": 0.7708333333333333,
      "wrong_first_drop": 0.4375,
      "wrong_later_drop": -0.00911458333333326,
      "zero_first": 0.125,
      "zero_later": 0.75,
      "matched_later_exposure_gap": 0.34244791666666663,
      "free_later_positions_reached": 178,
      "free_gold_later_positions_censored": 38,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36341,
      "method": "zero",
      "split": "train",
      "free_exact": 0.921875,
      "tf_first": 0.984375,
      "tf_later": 0.9765625,
      "wrong_first": 0.0,
      "wrong_later": 0.7395833333333333,
      "wrong_first_drop": 0.984375,
      "wrong_later_drop": 0.23697916666666674,
      "zero_first": 0.125,
      "zero_later": 0.78125,
      "matched_later_exposure_gap": 0.02994791666666663,
      "free_later_positions_reached": 214,
      "free_gold_later_positions_censored": 2,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": false,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    },
    {
      "seed": 36341,
      "method": "zero",
      "split": "development",
      "free_exact": 0.171875,
      "tf_first": 0.46875,
      "tf_later": 0.7057291666666666,
      "wrong_first": 0.0,
      "wrong_later": 0.7395833333333333,
      "wrong_first_drop": 0.46875,
      "wrong_later_drop": -0.03385416666666663,
      "zero_first": 0.125,
      "zero_later": 0.78125,
      "matched_later_exposure_gap": 0.30338541666666663,
      "free_later_positions_reached": 178,
      "free_gold_later_positions_censored": 38,
      "flags": {
        "POTENTIAL_PREFIX_RELIANCE": false,
        "EXPOSURE_SENSITIVITY": true,
        "DIRECTION_CONFLICT": null
      },
      "potential_first_later_cosine": null
    }
  ],
  "paired_effects": {
    "correct": {
      "train": {
        "wrong_first_drop": {
          "mean": 0.9947916666666666,
          "lower95": 0.984375,
          "upper95": 1.0,
          "n_seeds": 3
        },
        "wrong_later_drop": {
          "mean": 0.28125000000000006,
          "lower95": 0.22916666666666674,
          "upper95": 0.35416666666666674,
          "n_seeds": 3
        },
        "matched_later_exposure_gap": {
          "mean": 0.005208333333333333,
          "lower95": 0.0,
          "upper95": 0.015625,
          "n_seeds": 3
        }
      },
      "development": {
        "wrong_first_drop": {
          "mean": 0.546875,
          "lower95": 0.53125,
          "upper95": 0.5625,
          "n_seeds": 3
        },
        "wrong_later_drop": {
          "mean": 0.10329861111111116,
          "lower95": 0.00390625,
          "upper95": 0.18229166666666674,
          "n_seeds": 3
        },
        "matched_later_exposure_gap": {
          "mean": 0.2934027777777778,
          "lower95": 0.26692708333333337,
          "upper95": 0.31901041666666663,
          "n_seeds": 3
        }
      }
    },
    "shuffled": {
      "train": {
        "wrong_first_drop": {
          "mean": 0.9635416666666666,
          "lower95": 0.953125,
          "upper95": 0.984375,
          "n_seeds": 3
        },
        "wrong_later_drop": {
          "mean": 0.26605902777777785,
          "lower95": 0.19010416666666674,
          "upper95": 0.3411458333333333,
          "n_seeds": 3
        },
        "matched_later_exposure_gap": {
          "mean": 0.09331597222222221,
          "lower95": 0.06510416666666663,
          "upper95": 0.14453125,
          "n_seeds": 3
        }
      },
      "development": {
        "wrong_first_drop": {
          "mean": 0.4166666666666667,
          "lower95": 0.390625,
          "upper95": 0.4375,
          "n_seeds": 3
        },
        "wrong_later_drop": {
          "mean": 0.047309027777777825,
          "lower95": -0.015625,
          "upper95": 0.16666666666666674,
          "n_seeds": 3
        },
        "matched_later_exposure_gap": {
          "mean": 0.2756076388888889,
          "lower95": 0.2109375,
          "upper95": 0.3424479166666667,
          "n_seeds": 3
        }
      }
    },
    "zero": {
      "train": {
        "wrong_first_drop": {
          "mean": 0.984375,
          "lower95": 0.96875,
          "upper95": 1.0,
          "n_seeds": 3
        },
        "wrong_later_drop": {
          "mean": 0.39236111111111116,
          "lower95": 0.23697916666666674,
          "upper95": 0.5338541666666667,
          "n_seeds": 3
        },
        "matched_later_exposure_gap": {
          "mean": 0.025173611111111088,
          "lower95": 0.005208333333333259,
          "upper95": 0.04036458333333337,
          "n_seeds": 3
        }
      },
      "development": {
        "wrong_first_drop": {
          "mean": 0.4322916666666667,
          "lower95": 0.390625,
          "upper95": 0.46875,
          "n_seeds": 3
        },
        "wrong_later_drop": {
          "mean": 0.10590277777777778,
          "lower95": -0.03385416666666663,
          "upper95": 0.22656249999999992,
          "n_seeds": 3
        },
        "matched_later_exposure_gap": {
          "mean": 0.27777777777777773,
          "lower95": 0.20182291666666663,
          "upper95": 0.32812499999999994,
          "n_seeds": 3
        }
      }
    },
    "frozen": {
      "train": {
        "wrong_first_drop": {
          "mean": 0.984375,
          "lower95": 0.96875,
          "upper95": 1.0,
          "n_seeds": 3
        },
        "wrong_later_drop": {
          "mean": 0.39236111111111116,
          "lower95": 0.23697916666666674,
          "upper95": 0.5338541666666667,
          "n_seeds": 3
        },
        "matched_later_exposure_gap": {
          "mean": 0.025173611111111088,
          "lower95": 0.005208333333333259,
          "upper95": 0.04036458333333337,
          "n_seeds": 3
        }
      },
      "development": {
        "wrong_first_drop": {
          "mean": 0.4322916666666667,
          "lower95": 0.390625,
          "upper95": 0.46875,
          "n_seeds": 3
        },
        "wrong_later_drop": {
          "mean": 0.10590277777777778,
          "lower95": -0.03385416666666663,
          "upper95": 0.22656249999999992,
          "n_seeds": 3
        },
        "matched_later_exposure_gap": {
          "mean": 0.27777777777777773,
          "lower95": 0.20182291666666663,
          "upper95": 0.32812499999999994,
          "n_seeds": 3
        }
      }
    },
    "bptt_control": {
      "train": {
        "wrong_first_drop": {
          "mean": 1.0,
          "lower95": 1.0,
          "upper95": 1.0,
          "n_seeds": 3
        },
        "wrong_later_drop": {
          "mean": 0.3020833333333334,
          "lower95": 0.22916666666666674,
          "upper95": 0.41666666666666674,
          "n_seeds": 3
        },
        "matched_later_exposure_gap": {
          "mean": 0.0,
          "lower95": 0.0,
          "upper95": 0.0,
          "n_seeds": 3
        }
      },
      "development": {
        "wrong_first_drop": {
          "mean": 0.5260416666666666,
          "lower95": 0.5,
          "upper95": 0.578125,
          "n_seeds": 3
        },
        "wrong_later_drop": {
          "mean": 0.13975694444444445,
          "lower95": 0.078125,
          "upper95": 0.25,
          "n_seeds": 3
        },
        "matched_later_exposure_gap": {
          "mean": 0.30034722222222227,
          "lower95": 0.29557291666666663,
          "upper95": 0.30338541666666674,
          "n_seeds": 3
        }
      }
    }
  },
  "model_sources": 15,
  "public_cases_per_source": 128,
  "actual_forward_segments": 7680,
  "all_source_checkpoints_and_protected_fields_unchanged": true,
  "all_source_freeoutputs_and_teachermetrics_reproduced": true,
  "all_teacher_all_argmax_equals_free_exact": true,
  "neural_training": false,
  "neural_update_steps": 0,
  "torch_imported": false,
  "deployed": false,
  "main_weights_changed": false,
  "normal_chat_or_vision_proven": false,
  "resources": {
    "peak_rss_bytes": 52740096,
    "peak_commit_or_vms_bytes": 37486592,
    "os_process_peak_working_set_bytes": 53800960,
    "os_process_peak_pagefile_commit_bytes": 38334464,
    "samples": 212,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 18.54891490000591,
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
  "seconds": 18.83673280000221,
  "limits": [
    "Closed8styles/historicalcorpus,3pairedmodelseeds percondition, no generalchat/vision/lifelong or architecture-optimality claim.",
    "GOLDPREFIXexplicitdiagnostic, not freeanswer; wrong/zeroquestion are conflicts/OOD, flagsdescriptive not qualitygates.",
    "Potentialcredit FINALfixedcheckpoint only, NOT actualhistoricaltrainingupdates or prooflarge-normsynapses useful.",
    "Exposuregap onlysamepositions FREEreached; censoredlaterpositions explicitlyreported, not inventedpredictions.",
    "No newlearner/objective/epoch/LR/grid from thisaudit; any selectivecreditproposal needs separateprotocol. ExistingSTOPs/P1/P2 unchanged."
  ]
}
