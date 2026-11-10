# Knowledge load / retention diagnostic

One fixed48-neuron public G1 initialization; three curriculum orders. No private weights changed.

| Condition | Order | Lessons | Worded recall | Acquired | Retained | Old forgetting at last stage | Stop |
|---|---:|---:|---:|---:|---:|---:|---|
| frozen | 17 | 128 | 0.000 | 0 | 0 | 0.0pp | maximum128 reached |
| frozen | 29 | 128 | 0.000 | 0 | 0 | 0.0pp | maximum128 reached |
| frozen | 43 | 128 | 0.000 | 0 | 0 | 0.0pp | maximum128 reached |
| full | 17 | 16 | 0.000 | 0 | 0 | 25.0pp | two successive acquisition failures |
| full | 29 | 16 | 0.000 | 0 | 0 | 25.0pp | two successive acquisition failures |
| full | 43 | 16 | 0.000 | 0 | 0 | 25.0pp | two successive acquisition failures |
| readout | 17 | 16 | 0.000 | 0 | 0 | 25.0pp | two successive acquisition failures |
| readout | 29 | 16 | 0.000 | 0 | 0 | 25.0pp | two successive acquisition failures |
| readout | 43 | 16 | 0.000 | 0 | 0 | 25.0pp | two successive acquisition failures |
| full_replay | 17 | 16 | 0.000 | 0 | 0 | 25.0pp | two successive acquisition failures |
| full_replay | 29 | 16 | 0.000 | 0 | 0 | 25.0pp | two successive acquisition failures |
| full_replay | 43 | 16 | 0.000 | 0 | 0 | 25.0pp | two successive acquisition failures |
| guarded | 17 | 16 | 0.000 | 0 | 0 | 6.25pp | two successive acquisition failures |
| guarded | 29 | 16 | 0.000 | 0 | 0 | 0.0pp | two successive acquisition failures |
| guarded | 43 | 16 | 0.000 | 0 | 0 | 0.0pp | two successive acquisition failures |

Retention is undefined when no examples were acquired. Quiet10k ticks and exact restart do not establish days/months retention. Same primitive update slots only for frozen/full/readout/full_replay; guarded has additional guard/SQLite overhead.

{
  "effects_at16": {
    "full": {
      "mean": 0.0,
      "lower95": 0.0,
      "upper95": 0.0,
      "independent_seeds": 3,
      "method": "paired seed-cluster bootstrap; exploratory with only three clusters"
    },
    "readout": {
      "mean": 0.0,
      "lower95": 0.0,
      "upper95": 0.0,
      "independent_seeds": 3,
      "method": "paired seed-cluster bootstrap; exploratory with only three clusters"
    },
    "full_replay": {
      "mean": 0.0,
      "lower95": 0.0,
      "upper95": 0.0,
      "independent_seeds": 3,
      "method": "paired seed-cluster bootstrap; exploratory with only three clusters"
    },
    "guarded": {
      "mean": 0.0,
      "lower95": 0.0,
      "upper95": 0.0,
      "independent_seeds": 3,
      "method": "paired seed-cluster bootstrap; exploratory with only three clusters"
    }
  },
  "resources": {
    "peak_rss_bytes": 102662144,
    "peak_commit_or_vms_bytes": 66527232,
    "os_process_peak_working_set_bytes": 53911552,
    "os_process_peak_pagefile_commit_bytes": 38567936,
    "samples": 2060,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 179.51875479999944,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  },
  "decision": "PIVOT_NO_SCALE"
}
