# P0 outputusefulness engineering

No pilot/qualityclaim.

{
  "run": "runs/20261110T135250Z-readout-usefulness-p0-v1-ca2357",
  "decision": "P0_ENGINEERING_PASS_WITH_MATCHING_LIMITATION",
  "checks": {
    "math": {
      "deletion_synapses": 13092,
      "max_absolute_deletion_error": 2.4424906541753444e-15,
      "finite_difference_elements": 56,
      "max_absolute_output_derivative_error": 5.695937055349987e-10,
      "extreme_logits_checked": true,
      "rounded_probability_one_transition_checked": true,
      "masked_and_zero_feature_checked": true,
      "hidden_derivatives_used": false
    },
    "trajectories": [
      {
        "seed": 42201,
        "beta0_base_payload_exact_after_each_segment": true,
        "beta0_segments": 12,
        "beta0_endpoints": 36,
        "beta4_segments": 12,
        "beta4_endpoints": 36,
        "beta4_seconds": 0.08961789999739267,
        "beta4_losses": [
          1.4680914982818771,
          1.3899610290807507,
          1.327497099815225,
          1.1577683962135108,
          1.1198692366624632,
          1.0993085118424872,
          0.9896146205016526,
          0.9754786332515696,
          0.9647798426105996,
          0.8901240806182242,
          0.8821526249269289,
          0.874921973678754
        ],
        "beta4_omega_nonzero": 236,
        "beta4_omega_bytes": 3072,
        "weight_delta": {
          "w_in": 0.006748811341822147,
          "w_rec": 0.0027446651365607977,
          "w_out": 0.7361831068992615
        },
        "changed_synapses": {
          "w_in": 1190,
          "w_rec": 217,
          "w_out": 190
        }
      },
      {
        "seed": 42203,
        "beta0_base_payload_exact_after_each_segment": true,
        "beta0_segments": 12,
        "beta0_endpoints": 36,
        "beta4_segments": 12,
        "beta4_endpoints": 36,
        "beta4_seconds": 0.08830760000273585,
        "beta4_losses": [
          1.5565018548639038,
          1.44964612669573,
          1.4939235267160083,
          1.2548440578158138,
          1.1933096973050705,
          1.219803135445453,
          1.0707114404838767,
          1.0270660913917806,
          1.046406557603395,
          0.9455117170703722,
          0.9181462590052001,
          0.923583547722164
        ],
        "beta4_omega_nonzero": 240,
        "beta4_omega_bytes": 3072,
        "weight_delta": {
          "w_in": 0.004580958746373653,
          "w_rec": 0.001792989089153707,
          "w_out": 0.7470640540122986
        },
        "changed_synapses": {
          "w_in": 1058,
          "w_rec": 168,
          "w_out": 165
        }
      },
      {
        "seed": 42207,
        "beta0_base_payload_exact_after_each_segment": true,
        "beta0_segments": 12,
        "beta0_endpoints": 36,
        "beta4_segments": 12,
        "beta4_endpoints": 36,
        "beta4_seconds": 0.08152939999126829,
        "beta4_losses": [
          1.5591364924128224,
          1.4064443855320434,
          1.4018177906915874,
          1.2782297362437882,
          1.1937793410469608,
          1.194683065797819,
          1.1067077661157283,
          1.0514060613678038,
          1.0615717459954157,
          0.9959227811562416,
          0.9553425727563963,
          0.9663232962127362
        ],
        "beta4_omega_nonzero": 239,
        "beta4_omega_bytes": 3072,
        "weight_delta": {
          "w_in": 0.004227417521178722,
          "w_rec": 0.001362942042760551,
          "w_out": 0.6847448348999023
        },
        "changed_synapses": {
          "w_in": 912,
          "w_rec": 115,
          "w_out": 145
        }
      }
    ],
    "checkpoint": {
      "pending_token_boundary_new_instance_exact": true,
      "post_resume_trajectories_exact": true,
      "rng_fixture_and_shuffle_exact": true,
      "replay_and_controller_exact": true,
      "rejected_transaction_full_rollback": true,
      "core_identity_preserved_on_rollback": true,
      "unverified_sources_rejected_without_mutation": true,
      "product_codec_rejects_wrapper": true,
      "snapshot_bytes": 116598,
      "separate_process_resume_exact": true,
      "child_resources": {
        "peak_rss_bytes": 49586176,
        "peak_commit_or_vms_bytes": 27414528,
        "os_process_peak_working_set_bytes": 41086976,
        "os_process_peak_pagefile_commit_bytes": 26312704,
        "samples": 3,
        "interval_seconds": 0.01,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 0.060885999992024153,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "child_cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    "applied_norm": {
      "valid_interior_groups": 32,
      "maximum_relative_applied_norm_error": 2.1473569667076386e-08,
      "quantized_edge": {
        "radius": 5.960464477539063e-08,
        "actual_norms": [
          0.0,
          0.0
        ],
        "rejected": true,
        "reason": "Positive half-ULP radius not representable in float32 weight state"
      },
      "zero_radius_reported": true,
      "universally_feasible_matching": false,
      "raw_norm_used_as_applied": false
    },
    "signal_audit": {
      "imports_checked": [
        "ast",
        "copy",
        "hashlib",
        "io",
        "json",
        "neural.language",
        "neural.unified",
        "numpy",
        "os",
        "pathlib",
        "readout_usefulness",
        "runtime.atomic_file",
        "tempfile",
        "time",
        "zipfile"
      ],
      "autograd_or_transformer_imports": false,
      "backward_calls": 0,
      "hidden_signal": "Delegates exactly UnifiedCore.learn scalar rule; only Wout candidate overwritten",
      "scope": "Research modules only; full suite also checks production/native parity"
    },
    "max_output_timing": {
      "outputs": 266,
      "hidden": 48,
      "calls": 100,
      "seconds": 0.2978352000063751,
      "importance_float32_bytes": 51072,
      "quality_corpus_read": false
    }
  },
  "completed_checks": 6,
  "expected_checks": 6,
  "error": null,
  "pilot_approved": false,
  "pilot_run": false,
  "quality_accuracy_retention_measured": false,
  "normal_chat_or_vision_proven": false,
  "main_training": false,
  "main_restored": false,
  "neural_source_unchanged": true,
  "cap": {
    "enforced": true,
    "limit_bytes": 2000000000,
    "kind": "Windows per-process committed memory; shared pages monitored separately"
  },
  "resources": {
    "peak_rss_bytes": 95494144,
    "peak_commit_or_vms_bytes": 92352512,
    "os_process_peak_working_set_bytes": 49487872,
    "os_process_peak_pagefile_commit_bytes": 67993600,
    "samples": 29,
    "interval_seconds": 0.05,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 1.4937118000234477,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  },
  "seconds": 1.6581821999861859,
  "limits": [
    "P0 does not establish usefulimportance, representationgain or lifelonglearning.",
    "Applied-norm matching must reject unrepresentable float32 states; not universally feasible.",
    "Oracle registry validates P0 labels only, not arbitrary userfacts or verified flags.",
    "Pilot and deployment require separate userapproval."
  ]
}
