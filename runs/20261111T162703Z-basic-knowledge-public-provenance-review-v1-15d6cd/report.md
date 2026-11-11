# Frozen basic knowledge origin/source review

No new generation or learning. Private evidence was checked locally; public outputs contain hashes/counts only.

{
  "decision": "PUBLIC_SOURCE_LINEAGE_VERIFIED_ORIGIN_NOT_REATTESTED_CODEC_BLOCKED",
  "public_only_mode": true,
  "source_pool": "data/candidates/20261110T191533Z-basic-knowledge-repair-v1-231806/verified-pool.json",
  "source_pool_sha256": "13bb051467b27caa10bf8d16039d52fa830baf77a5e87bd63e3520d8b96ed062",
  "records": 320,
  "known_facts": 32,
  "outside_domain_facts": 8,
  "actual_historical_generation_calls": 5,
  "new_generation_calls": 0,
  "historical_cli_version": "1.5.5",
  "historical_executable_sha256": "fe9eb2f3884ca6cd630c57eed117580819341bce4b87d2c40b441ddd01cb8f2a",
  "historical_model": "gemini-3.8-flash-high",
  "historical_effort": "high",
  "source_checks": [
    {
      "path": "src/scarx/neural/core.py",
      "sha256": "c6aea5f9189931e53cc9f32d83cf5a09ac814f997fc5932890501bd91fa8365c",
      "git_blob_content_exact": true
    },
    {
      "path": "src/scarx/neural/unified.py",
      "sha256": "0db6606e3b9f6972c758becf7c0d96c4fc5aa01214b61f64b47f981edada9616",
      "git_blob_content_exact": true
    },
    {
      "path": "src/scarx/app/generation_server.py",
      "sha256": "19f7ec96b867325ef27a053a96b4e4a2821ae166062164e76a28800f4fafcb57",
      "git_blob_content_exact": true
    },
    {
      "path": "src/scarx/app/generation.py",
      "sha256": "0328c401db2aaa42bf7cf342b87e0b1b4f5c06b1fb40a4030291e9c22257dfd9",
      "git_blob_content_exact": true
    }
  ],
  "facts_reproduced_from_ast": true,
  "batch_checks": [
    {
      "batch": 1,
      "candidate_records": 80,
      "candidate_sha256": "b01c63b0154a85969063d72ffbfbcd4872c0bf50742314ec6e0b275f76fda8fb",
      "model_metadata_matches_recorded_receipt": true,
      "origin_attestation_pass": null,
      "submitted_prompt_hash_reproduced_from_public_sources": true,
      "origin_evidence_scope": "Public receipt and candidate hash only; original private CLI stream not independently reattested in this mode",
      "raw_private_content_published": false
    },
    {
      "batch": 2,
      "candidate_records": 80,
      "candidate_sha256": "4825c968a9070ff3b0e602639d6dc5ebff3945c62ab069249096849917fb6b40",
      "model_metadata_matches_recorded_receipt": true,
      "origin_attestation_pass": null,
      "submitted_prompt_hash_reproduced_from_public_sources": true,
      "origin_evidence_scope": "Public receipt and candidate hash only; original private CLI stream not independently reattested in this mode",
      "raw_private_content_published": false
    },
    {
      "batch": 3,
      "candidate_records": 80,
      "candidate_sha256": "c9c43832c3cf7d69b1e1c0066e1cf9fc5acd4aace21b07dac9bd6fa58f11da31",
      "model_metadata_matches_recorded_receipt": true,
      "origin_attestation_pass": null,
      "submitted_prompt_hash_reproduced_from_public_sources": true,
      "origin_evidence_scope": "Public receipt and candidate hash only; original private CLI stream not independently reattested in this mode",
      "raw_private_content_published": false
    },
    {
      "batch": 4,
      "candidate_records": 80,
      "candidate_sha256": "7004238d6422c8439d46d34f9198119113e7c4c25bc16e75701b2bfaff8016a3",
      "model_metadata_matches_recorded_receipt": true,
      "origin_attestation_pass": null,
      "submitted_prompt_hash_reproduced_from_public_sources": true,
      "origin_evidence_scope": "Public receipt and candidate hash only; original private CLI stream not independently reattested in this mode",
      "raw_private_content_published": false
    },
    {
      "batch": 5,
      "candidate_records": 72,
      "candidate_sha256": "86665ee21d0420059bb2c01b597fc984902f89ccde70b6f63b08f178923737d8",
      "model_metadata_matches_recorded_receipt": true,
      "origin_attestation_pass": null,
      "submitted_prompt_hash_reproduced_from_public_sources": true,
      "origin_evidence_scope": "Public receipt and candidate hash only; original private CLI stream not independently reattested in this mode",
      "raw_private_content_published": false
    }
  ],
  "accepted_raw_candidate_origin_matches": 320,
  "split_file_matches": {
    "development": true,
    "heldout_confirmation": true,
    "train": true,
    "unknown_confirmation": true,
    "unknown_development": true
  },
  "split_counts": {
    "unknown_confirmation": 32,
    "unknown_development": 32,
    "train": 128,
    "development": 64,
    "heldout_confirmation": 64
  },
  "issues": [],
  "resources": {
    "peak_rss_bytes": 60964864,
    "peak_commit_or_vms_bytes": 67878912,
    "os_process_peak_working_set_bytes": 48656384,
    "os_process_peak_pagefile_commit_bytes": 36446208,
    "samples": 16,
    "interval_seconds": 0.02,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 0.3373594000004232,
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
  "private_streams_or_invocations_published": false,
  "historical_cost_usd_reported": null,
  "compatibility_decision_preserved": "DATASET_NOT_LEARNABLE_UNDER_CURRENT_CODEC",
  "automatic_learning_allowed": false,
  "student_training": false,
  "final_quality_inference_performed": false,
  "production_ready": false
}
