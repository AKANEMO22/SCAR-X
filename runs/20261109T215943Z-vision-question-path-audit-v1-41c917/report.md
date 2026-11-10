# Vision question path audit

240actual Generation.vision calls on an isolatedpublicrelease clone. Questionsemanticchannels zero, identicalimage/state/budget gives identicalneuralinputs/output across differentquestionkinds. NOT training/generalvision success. Detailed outputs.json includes allwrong/rightdigits, targets onlyrendererannotations.

{
  "run": "runs/20261109T215943Z-vision-question-path-audit-v1-41c917",
  "decision": "VISION_QUESTION_DROPPED_BEFORE_NEURAL_CORE",
  "data": "data/candidates/20261109T215943Z-vision-question-path-audit-v1-41c917/qa-fixtures.json",
  "data_sha256": "a2a66ca97cfbd12a4f163ea215fe6e1a3c0aaea6daa1fc0c7593d8317ff5852c",
  "scene_count": 12,
  "question_count": 32,
  "inferences": 240,
  "by_scene": {
    "s01": {
      "cases": 24,
      "question_kinds": [
        "object_count",
        "single_color",
        "single_shape"
      ],
      "distinct_renderer_targets": 5,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 24
    },
    "s02": {
      "cases": 24,
      "question_kinds": [
        "object_count",
        "single_color",
        "single_shape"
      ],
      "distinct_renderer_targets": 5,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 24
    },
    "s03": {
      "cases": 24,
      "question_kinds": [
        "object_count",
        "single_color",
        "single_shape"
      ],
      "distinct_renderer_targets": 5,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 24
    },
    "s04": {
      "cases": 24,
      "question_kinds": [
        "object_count",
        "single_color",
        "single_shape"
      ],
      "distinct_renderer_targets": 5,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 24
    },
    "s05": {
      "cases": 24,
      "question_kinds": [
        "object_count",
        "single_color",
        "single_shape"
      ],
      "distinct_renderer_targets": 5,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 24
    },
    "s06": {
      "cases": 24,
      "question_kinds": [
        "object_count",
        "single_color",
        "single_shape"
      ],
      "distinct_renderer_targets": 5,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 24
    },
    "s07": {
      "cases": 16,
      "question_kinds": [
        "left_color",
        "object_count"
      ],
      "distinct_renderer_targets": 3,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 16
    },
    "s08": {
      "cases": 16,
      "question_kinds": [
        "left_color",
        "object_count"
      ],
      "distinct_renderer_targets": 3,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 16
    },
    "s09": {
      "cases": 16,
      "question_kinds": [
        "left_color",
        "object_count"
      ],
      "distinct_renderer_targets": 3,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 16
    },
    "s10": {
      "cases": 16,
      "question_kinds": [
        "left_color",
        "object_count"
      ],
      "distinct_renderer_targets": 3,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 16
    },
    "s11": {
      "cases": 16,
      "question_kinds": [
        "left_color",
        "object_count"
      ],
      "distinct_renderer_targets": 3,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 16
    },
    "s12": {
      "cases": 16,
      "question_kinds": [
        "left_color",
        "object_count"
      ],
      "distinct_renderer_targets": 3,
      "all_questions_same_neural_input": true,
      "all_questions_same_neural_probabilities": true,
      "prototype_distinct_multimodal_sequences": 16
    }
  },
  "color_pairs": [
    {
      "first": "s01",
      "second": "s02",
      "same_grayscale_pixels": true,
      "RGB_prototype_inputs_differ": true
    },
    {
      "first": "s03",
      "second": "s04",
      "same_grayscale_pixels": true,
      "RGB_prototype_inputs_differ": true
    },
    {
      "first": "s05",
      "second": "s06",
      "same_grayscale_pixels": true,
      "RGB_prototype_inputs_differ": true
    },
    {
      "first": "s07",
      "second": "s08",
      "same_grayscale_pixels": true,
      "RGB_prototype_inputs_differ": true
    },
    {
      "first": "s09",
      "second": "s10",
      "same_grayscale_pixels": true,
      "RGB_prototype_inputs_differ": true
    },
    {
      "first": "s11",
      "second": "s12",
      "same_grayscale_pixels": true,
      "RGB_prototype_inputs_differ": true
    }
  ],
  "all_scene_question_inputs_identical": true,
  "all_scene_question_outputs_identical": true,
  "question_context_channels_nonzero_total": 0,
  "source_release_sha256": "0c1e8bab2c4ca2ec10578bd4ee0e745366ce1f0b483b4b72b418e09fc5fe10a1",
  "all_source_weights_updates_dialogue_unchanged": true,
  "discarded_session_restored_exact": true,
  "source_encoder_pixel_sha256": {
    "s01": "72b4a624cb29f567a16d45601d524b2b040d10c1f095e04a606d595630634d15",
    "s02": "72b4a624cb29f567a16d45601d524b2b040d10c1f095e04a606d595630634d15",
    "s03": "bfbddf9b6b2d70c4ae3edd22cbe7a7fdfc78e318b74a21f534b800f621689979",
    "s04": "bfbddf9b6b2d70c4ae3edd22cbe7a7fdfc78e318b74a21f534b800f621689979",
    "s05": "0ce485da9c1f2f1dac7dd2ca3aeb38bd2d5c3442e5217b3789bced2649b66654",
    "s06": "0ce485da9c1f2f1dac7dd2ca3aeb38bd2d5c3442e5217b3789bced2649b66654",
    "s07": "bcd57cbfd6b098ec3fa6a06cb6e5fd0494105dbbc75947fdd1b5c96f71f70e48",
    "s08": "bcd57cbfd6b098ec3fa6a06cb6e5fd0494105dbbc75947fdd1b5c96f71f70e48",
    "s09": "0a7870bf66296b049c902a802f63ef9aedf214bd2c4eb2442dca1bd9b32525e7",
    "s10": "0a7870bf66296b049c902a802f63ef9aedf214bd2c4eb2442dca1bd9b32525e7",
    "s11": "cde23b5594d5440e5a96b167650261950e379ed353c853cbc2bec9049bd1849a",
    "s12": "cde23b5594d5440e5a96b167650261950e379ed353c853cbc2bec9049bd1849a"
  },
  "new_training": false,
  "deployed": false,
  "normal_photo_qa_proven": false,
  "learned_caption_or_language_proven": false,
  "Qwen_vision_reference_measured": false,
  "reason_missing_Qwen_vision": "Installed Qwen3-4B-Instruct text-only; no QwenVLstarted/downloaded or image-as-text substitute.",
  "proposal30_approved": false,
  "resources": {
    "peak_rss_bytes": 59060224,
    "peak_commit_or_vms_bytes": 42590208,
    "os_process_peak_working_set_bytes": 60039168,
    "os_process_peak_pagefile_commit_bytes": 43970560,
    "samples": 75,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 6.9435549000045285,
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
    "One fixedpubliccheckpoint, syntheticfixturecounterexamples, not240models/qualitybenchmark.",
    "Colorloss repeats knownreport13; newend-to-end questioninsensitivity audit, no novelvisionrepresentationclaim.",
    "Different prototypeinputsequences do not prove learnedvision or semanticquestionunderstanding.",
    "Question mayaffect defaultcapacity heuristics; here48capacityfixed to isolate actualsemanticinput."
  ]
}
