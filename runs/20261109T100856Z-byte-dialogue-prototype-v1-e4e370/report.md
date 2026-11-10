# Fixed UTF8 byte dialogue prototype

Codec removes word-dictionary OOV, but valid UTF8 is enforced by a deterministic host mask. Semantic task success and lifelong retention remain separate.

| Condition | Dev exact | Train exact | Forced byte accuracy | Nonempty |
|---|---:|---:|---:|---:|
| byte_frozen | 0.0000 | 0.0000 | 0.0048 | 1.0000 |
| byte_raw | 0.0000 | 0.0000 | 0.2292 | 1.0000 |
| byte_normalized | 0.0000 | 0.0000 | 0.1233 | 1.0000 |
| byte_normalized_full | 0.0000 | 0.0000 | 0.1316 | 1.0000 |

{
  "run": "runs/20261109T100856Z-byte-dialogue-prototype-v1-e4e370",
  "protocol": "experiments/byte-dialogue-prototype-v1.md",
  "data": "data/candidates/antigravity-events-curated-20261109T094142Z",
  "representability": {
    "training": 96,
    "development": 24,
    "all_roundtrip_exact": true,
    "oov_bytes": 0,
    "alphabet_fit_from_data": false,
    "reference_old_word_exact_representable": 9,
    "reference_old_word_dev_total": 24,
    "constraint": "Representable does not mean a neural model can generate a coherent answer"
  },
  "effects": {
    "normalization": {
      "mean": 0.0,
      "lower95": 0.0,
      "upper95": 0.0,
      "independent_seeds": 3,
      "method": "paired seed-cluster bootstrap; exploratory with only three clusters"
    },
    "hidden": {
      "mean": 0.0,
      "lower95": 0.0,
      "upper95": 0.0,
      "independent_seeds": 3,
      "method": "paired seed-cluster bootstrap; exploratory with only three clusters"
    }
  },
  "resources": {
    "peak_rss_bytes": 98566144,
    "peak_commit_or_vms_bytes": 129986560,
    "os_process_peak_working_set_bytes": 55660544,
    "os_process_peak_pagefile_commit_bytes": 76189696,
    "samples": 831,
    "interval_seconds": 0.25,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 210.57769299999927,
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
  "historical_word_trial_not_compute_matched": true,
  "syntax_validity_is_host_enforced": true,
  "normal_chat_or_vision_proven": false,
  "retention_proven": false,
  "deployed": false,
  "decision": "STOP_BYTE_DIALOGUE_V1"
}

Historical word-vs-byte compute and output dimensions differ. Same48ALIFneuron topology; no private session/weights replaced.
