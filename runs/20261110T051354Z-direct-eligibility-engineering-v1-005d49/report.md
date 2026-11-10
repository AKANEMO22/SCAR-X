# Proposal30 engineering

No devquality evaluation, no mainchange. Finite-difference oracle is anchored local smooth sensitivity, NOT hardspike/fullnetworkgradient.

{
  "run": "runs/20261110T051354Z-direct-eligibility-engineering-v1-005d49",
  "decision": "DIRECT_ELIGIBILITY_ENGINEERING_PASS",
  "proposal30_approved": true,
  "mathematics": {
    "finite_difference_synapses": 64,
    "max_absolute_error": 2.5921401469730654e-08,
    "neuron_rows": [
      {
        "neuron": 0,
        "synapses": 16,
        "max_absolute_error": 1.7412357118662403e-08
      },
      {
        "neuron": 7,
        "synapses": 16,
        "max_absolute_error": 2.5921401469730654e-08
      },
      {
        "neuron": 19,
        "synapses": 16,
        "max_absolute_error": 1.579114661254266e-08
      },
      {
        "neuron": 37,
        "synapses": 16,
        "max_absolute_error": 2.071997374369916e-08
      }
    ],
    "forward_state_exact": true,
    "oracle_scope": "Anchored smooth own-neuron surrogate, fixed presynaptic tape; NOT finite derivative of hardspikes/fullnetwork."
  },
  "partialsegment_optimizer_rng_exact": true,
  "zero_frozen_neural_head_exact": true,
  "timing": {
    "correct": {
      "training_examples": 64,
      "training_seconds": 0.06629619999148417,
      "setup_seconds": 0.04159510000317823,
      "forecast_trial_with25pct_headroom": 20.127917874833656,
      "trace_and_core_array_bytes": 29666,
      "checkpoint_bytes": 46704,
      "optimizer_hidden_updates_timing": 64,
      "codec_nextupdate_exact": true
    },
    "bptt_control": {
      "training_examples": 64,
      "training_seconds": 0.2685123000119347,
      "setup_seconds": 1.3479716000001645,
      "forecast_trial_with25pct_headroom": 25.8052105002389,
      "trace_and_core_array_bytes": 29666,
      "checkpoint_bytes": 47324,
      "optimizer_hidden_updates_timing": 64,
      "codec_nextupdate_exact": true,
      "max_graph_state_error": 2.384185791015625e-07,
      "max_graph_feature_error": 0.0
    }
  },
  "forecast_screen_seconds": 956.8519379961617,
  "formal_screen_permitted": true,
  "no_development_quality_evaluated": true,
  "deployed": false,
  "main_weights_changed": false,
  "normal_chat_or_vision_proven": false,
  "resources": {
    "peak_rss_bytes": 556941312,
    "peak_commit_or_vms_bytes": 978034688,
    "os_process_peak_working_set_bytes": 557969408,
    "os_process_peak_pagefile_commit_bytes": 978837504,
    "samples": 21,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 1.9783979000058025,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  },
  "cap": {
    "enforced": true,
    "limit_bytes": 2000000000,
    "kind": "Windows per-process committed memory; shared pages monitored separately"
  }
}
