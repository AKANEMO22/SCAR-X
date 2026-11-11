# Approved isolated output-codec P0

Reference roundtrip strings are not neural answers. Fixture updates are state tests, not dataset training.

{
  "scope": "P0 engineering only; not a learned knowledge or retention result",
  "gates": {
    "v2_all_target_roundtrips_and_coverage": true,
    "fixed_55_inventory_within_256": true,
    "old_context_word_digit_neuron_parity_exact": true,
    "cold_state_rng_replay_cursor_exact": true,
    "full_snapshot_rollback_exact": true,
    "source_release_and_dataset_bytes_unchanged": true,
    "unit_bounds_corruption_foreign_loader_checks": true,
    "observed_process_tree_below_2gb": true
  },
  "details": {
    "naive_width_expansion": {
      "voltage": 4.76837158203125e-07,
      "adaptation": 0.0,
      "spikes": 0.0,
      "features": 0.0,
      "status": "Diagnostic control, not used to relax exact compatibility gate"
    },
    "codec_roundtrips": {
      "legacy-word-v1": {
        "rows": 320,
        "roundtrip_matches": 192,
        "covered_rows": 320
      },
      "numeric-atom-v2": {
        "rows": 320,
        "roundtrip_matches": 320,
        "covered_rows": 320
      }
    },
    "vocabulary_sizes": {
      "legacy-word-v1": 55,
      "numeric-atom-v2": 55
    },
    "legacy_parity_errors": {
      "context": 0.0,
      "word_probabilities": 0.0,
      "voltage": 0.0,
      "adaptation": 0.0,
      "spikes": 0.0,
      "features": 0.0,
      "elig_rec": 0.0,
      "digit_probabilities": 0.0
    },
    "cold_process_checks": [
      {
        "codec": "legacy-word-v1",
        "command": [
          "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
          "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-output-codec-p0-20261111\\scripts\\output_codec_p0_cold.py",
          "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-output-codec-p0-20261111\\runs\\20261111T171706Z-output-codec-p0-v1-ecc876\\legacy-word-v1-partial.zip",
          "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-output-codec-p0-20261111\\runs\\20261111T171706Z-output-codec-p0-v1-ecc876\\legacy-word-v1-cold-output.json"
        ],
        "exit_code": 0,
        "seconds": 0.16446699999505654,
        "full_continuation_json_exact": true,
        "saved_checkpoint_preserved": true,
        "updates_before_snapshot": 1,
        "parent_continuation_updates": 4,
        "cold_continuation_updates": 4,
        "expected": {
          "losses": [
            0.5923210174641974,
            18.762581693623716,
            0.26750858137373473,
            0.07702639857278237
          ],
          "replay_fixture": {
            "prompt": "P0 replay decimal",
            "answer": "0.25.",
            "codec_version": "legacy-word-v1"
          },
          "rng_draw": 242909,
          "capacity_rng_draw": 925054,
          "generated_tokens_for_continuation_identity_only": [
            "đang",
            "học",
            "trả",
            "lời"
          ],
          "state_sha256": "f7fd95905da045d6456d8459cf4af06e4ee80fc732ef530a01f421d32f1d4ddb",
          "fixture_updates": 5,
          "scope": "Engineering state identity, not answer correctness or acquisition"
        },
        "actual": {
          "losses": [
            0.5923210174641974,
            18.762581693623716,
            0.26750858137373473,
            0.07702639857278237
          ],
          "replay_fixture": {
            "prompt": "P0 replay decimal",
            "answer": "0.25.",
            "codec_version": "legacy-word-v1"
          },
          "rng_draw": 242909,
          "capacity_rng_draw": 925054,
          "generated_tokens_for_continuation_identity_only": [
            "đang",
            "học",
            "trả",
            "lời"
          ],
          "state_sha256": "f7fd95905da045d6456d8459cf4af06e4ee80fc732ef530a01f421d32f1d4ddb",
          "fixture_updates": 5,
          "scope": "Engineering state identity, not answer correctness or acquisition"
        }
      },
      {
        "codec": "numeric-atom-v2",
        "command": [
          "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
          "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-output-codec-p0-20261111\\scripts\\output_codec_p0_cold.py",
          "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-output-codec-p0-20261111\\runs\\20261111T171706Z-output-codec-p0-v1-ecc876\\numeric-atom-v2-partial.zip",
          "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-output-codec-p0-20261111\\runs\\20261111T171706Z-output-codec-p0-v1-ecc876\\numeric-atom-v2-cold-output.json"
        ],
        "exit_code": 0,
        "seconds": 0.1640250999989803,
        "full_continuation_json_exact": true,
        "saved_checkpoint_preserved": true,
        "updates_before_snapshot": 1,
        "parent_continuation_updates": 2,
        "cold_continuation_updates": 2,
        "expected": {
          "losses": [
            0.5923210174641974,
            0.22805493030289017
          ],
          "replay_fixture": {
            "prompt": "P0 replay decimal",
            "answer": "0.25.",
            "codec_version": "numeric-atom-v2"
          },
          "rng_draw": 242909,
          "capacity_rng_draw": 925054,
          "generated_tokens_for_continuation_identity_only": [
            "đang",
            "học",
            "trả",
            "lời"
          ],
          "state_sha256": "d054356f34831f764ef7e3caf3e04973bb745ba4b0096adb7ac1581b6f60c7e1",
          "fixture_updates": 3,
          "scope": "Engineering state identity, not answer correctness or acquisition"
        },
        "actual": {
          "losses": [
            0.5923210174641974,
            0.22805493030289017
          ],
          "replay_fixture": {
            "prompt": "P0 replay decimal",
            "answer": "0.25.",
            "codec_version": "numeric-atom-v2"
          },
          "rng_draw": 242909,
          "capacity_rng_draw": 925054,
          "generated_tokens_for_continuation_identity_only": [
            "đang",
            "học",
            "trả",
            "lời"
          ],
          "state_sha256": "d054356f34831f764ef7e3caf3e04973bb745ba4b0096adb7ac1581b6f60c7e1",
          "fixture_updates": 3,
          "scope": "Engineering state identity, not answer correctness or acquisition"
        }
      }
    ],
    "logical_fixture_synaptic_updates_including_rejected": 11,
    "actual_runner_fixture_update_calls_including_cold_verification": 17,
    "unit_test_fixture_update_calls_included": false,
    "unit_test_command": {
      "command": [
        "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_output_codec_p0.py",
        "--junitxml=C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-output-codec-p0-20261111\\runs\\20261111T171706Z-output-codec-p0-v1-ecc876\\p0-unit-tests.xml"
      ],
      "exit_code": 0
    }
  },
  "error": null,
  "resources": {
    "peak_rss_bytes": 172408832,
    "peak_commit_or_vms_bytes": 142229504,
    "os_process_peak_working_set_bytes": 48340992,
    "os_process_peak_pagefile_commit_bytes": 32976896,
    "samples": 72,
    "interval_seconds": 0.02,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 2.1269460000039544,
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
  "main_training": false,
  "dataset_training": false,
  "new_teacher_calls": 0,
  "codec_used_for_final_quality_inference": false,
  "main_or_live_migrated": false,
  "production_ready": false,
  "limits": [
    "Public H48 only; capacity growth and foreign PlasticCore/importance/optimizer migration rejected, not certified.",
    "Fixture controller/replay only; not production Generation-session/SQLite migration.",
    "Numeric compatibility block arithmetic; not a speed/quality improvement or a new learning rule."
  ],
  "decision": "P0_TECHNICAL_PASS_NOT_TRAINING_OR_RELEASE_APPROVAL"
}
