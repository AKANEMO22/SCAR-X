# Grounded developmentreferences

Offline data/reference/encoder audit only. No modellearning or normalvision claim.

{
  "run": "runs/20261110T142721Z-adaptive-curriculum-grounding-v1-a7c646",
  "decision": "DATA_GROUNDED_NOT_TRAINING_READY",
  "data": "data/candidates/20261110T142721Z-adaptive-curriculum-grounding-v1-a7c646/dataset.json",
  "sha256": "1b914509ab96c8b1c5375c72950fa22deb1038d7068e7d4eec4463a06f2e8c87",
  "chat_references": 95,
  "owned_images": 36,
  "vision_qa": 576,
  "independent_pixel_checks": 36,
  "pixel_manifest_disagreements": 0,
  "question_only_ceiling_by_kind": {
    "single_color": 0.5,
    "single_shape": 0.3333333333333333,
    "object_count": 0.3333333333333333,
    "left_color": 0.5
  },
  "legacy64_conditional_ceiling_by_kind": {
    "single_color": 0.5,
    "single_shape": 1.0,
    "object_count": 1.0,
    "left_color": 0.5
  },
  "left_bag_of_objects_ceiling": 0.5,
  "counterfactual_groups": 18,
  "legacy64_equal_color_groups": 18,
  "compatibility": {
    "chat_public_release": {
      "references": 95,
      "distinct_reference_tokens": 360,
      "max_reference_tokens": 24,
      "longer_than_24": 0,
      "vocabulary_cap_in_current_code": 256,
      "target_token_limit_in_current_code": 24,
      "scope": "Lexical/length compatibility only, not proof of learnability or API modality support",
      "metadata_available": true,
      "existing_vocabulary_size": 24,
      "needed_vocabulary_size": 362,
      "exceeds_256": true,
      "fully_known_reference_count": 1
    },
    "vision_public_release": {
      "references": 576,
      "distinct_reference_tokens": 35,
      "max_reference_tokens": 8,
      "longer_than_24": 0,
      "vocabulary_cap_in_current_code": 256,
      "target_token_limit_in_current_code": 24,
      "scope": "Lexical/length compatibility only, not proof of learnability or API modality support",
      "metadata_available": true,
      "existing_vocabulary_size": 24,
      "needed_vocabulary_size": 55,
      "exceeds_256": false,
      "fully_known_reference_count": 0
    },
    "combined_public_release": {
      "references": 671,
      "distinct_reference_tokens": 386,
      "max_reference_tokens": 24,
      "longer_than_24": 0,
      "vocabulary_cap_in_current_code": 256,
      "target_token_limit_in_current_code": 24,
      "scope": "Lexical/length compatibility only, not proof of learnability or API modality support",
      "metadata_available": true,
      "existing_vocabulary_size": 24,
      "needed_vocabulary_size": 388,
      "exceeds_256": true,
      "fully_known_reference_count": 1
    },
    "combined_live_aggregate": {
      "references": 671,
      "distinct_reference_tokens": 386,
      "max_reference_tokens": 24,
      "longer_than_24": 0,
      "vocabulary_cap_in_current_code": 256,
      "target_token_limit_in_current_code": 24,
      "scope": "Lexical/length compatibility only, not proof of learnability or API modality support",
      "metadata_available": true,
      "existing_vocabulary_size": 25,
      "needed_vocabulary_size": 388,
      "exceeds_256": true,
      "fully_known_reference_count": 1
    },
    "private_vocabulary_or_fingerprint_published": false,
    "current_vision_api_is_digits_only": true
  },
  "preview": "artifacts/adaptive-grounding-preview.png",
  "color_loss_preview": "artifacts/adaptive-grounding-color-loss.png",
  "live_metadata_available": true,
  "live_weights_unchanged_during_build": true,
  "automatic_learning_allowed": false,
  "pilot_run": false,
  "main_training": false,
  "main_restored": false,
  "neural_instances_created": 0,
  "normal_chat_or_vision_proven": false,
  "new_teacher_or_qwen_call": false,
  "resources": {
    "peak_rss_bytes": 57856000,
    "peak_commit_or_vms_bytes": 41062400,
    "os_process_peak_working_set_bytes": 58982400,
    "os_process_peak_pagefile_commit_bytes": 43286528,
    "samples": 61,
    "interval_seconds": 0.05,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 1.647148299991386,
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
  "seconds": 1.8461686999944504,
  "error": null
}
