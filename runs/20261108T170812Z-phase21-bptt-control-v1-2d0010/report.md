# Phase2.1 isolated BPTT control

```json
{
  "trials": [
    {
      "seed": 101,
      "task": "xor",
      "accuracy": 0.5,
      "train_seconds": 1.6752121000026818,
      "updates": 512
    },
    {
      "seed": 101,
      "task": "order",
      "accuracy": 0.5,
      "train_seconds": 1.612955000004149,
      "updates": 512
    },
    {
      "seed": 101,
      "task": "binding",
      "accuracy": 0.734375,
      "train_seconds": 1.6293259000085527,
      "updates": 512
    },
    {
      "seed": 127,
      "task": "xor",
      "accuracy": 0.5,
      "train_seconds": 1.7223965000011958,
      "updates": 512
    },
    {
      "seed": 127,
      "task": "order",
      "accuracy": 0.796875,
      "train_seconds": 1.65803789999336,
      "updates": 512
    },
    {
      "seed": 127,
      "task": "binding",
      "accuracy": 0.67578125,
      "train_seconds": 1.6916784000059124,
      "updates": 512
    },
    {
      "seed": 149,
      "task": "xor",
      "accuracy": 0.5,
      "train_seconds": 1.7174372000008589,
      "updates": 512
    },
    {
      "seed": 149,
      "task": "order",
      "accuracy": 0.82421875,
      "train_seconds": 1.6353701999905752,
      "updates": 512
    },
    {
      "seed": 149,
      "task": "binding",
      "accuracy": 0.6953125,
      "train_seconds": 1.6470492000080412,
      "updates": 512
    },
    {
      "seed": 173,
      "task": "xor",
      "accuracy": 0.5,
      "train_seconds": 1.7038352999952622,
      "updates": 512
    },
    {
      "seed": 173,
      "task": "order",
      "accuracy": 0.703125,
      "train_seconds": 1.7913628999958746,
      "updates": 512
    },
    {
      "seed": 173,
      "task": "binding",
      "accuracy": 0.7890625,
      "train_seconds": 1.8641122999979416,
      "updates": 512
    },
    {
      "seed": 197,
      "task": "xor",
      "accuracy": 0.5,
      "train_seconds": 1.8427445999986958,
      "updates": 512
    },
    {
      "seed": 197,
      "task": "order",
      "accuracy": 0.56640625,
      "train_seconds": 1.7701479000097606,
      "updates": 512
    },
    {
      "seed": 197,
      "task": "binding",
      "accuracy": 0.734375,
      "train_seconds": 1.6423748999950476,
      "updates": 512
    },
    {
      "seed": 223,
      "task": "xor",
      "accuracy": 0.5,
      "train_seconds": 1.6610946000000695,
      "updates": 512
    },
    {
      "seed": 223,
      "task": "order",
      "accuracy": 0.71875,
      "train_seconds": 1.626297099996009,
      "updates": 512
    },
    {
      "seed": 223,
      "task": "binding",
      "accuracy": 0.76171875,
      "train_seconds": 1.7977616999996826,
      "updates": 512
    },
    {
      "seed": 251,
      "task": "xor",
      "accuracy": 0.5,
      "train_seconds": 1.7531644999980927,
      "updates": 512
    },
    {
      "seed": 251,
      "task": "order",
      "accuracy": 0.73046875,
      "train_seconds": 1.6796116000041366,
      "updates": 512
    },
    {
      "seed": 251,
      "task": "binding",
      "accuracy": 0.71484375,
      "train_seconds": 1.6527174999937415,
      "updates": 512
    },
    {
      "seed": 277,
      "task": "xor",
      "accuracy": 0.5,
      "train_seconds": 1.7372459999896819,
      "updates": 512
    },
    {
      "seed": 277,
      "task": "order",
      "accuracy": 0.80078125,
      "train_seconds": 1.9256178000068758,
      "updates": 512
    },
    {
      "seed": 277,
      "task": "binding",
      "accuracy": 0.7421875,
      "train_seconds": 6.987189799998305,
      "updates": 512
    }
  ],
  "resources": {
    "peak_rss_bytes": 630181888,
    "peak_commit_or_vms_bytes": 1050951680,
    "os_process_peak_working_set_bytes": 631255040,
    "os_process_peak_pagefile_commit_bytes": 1051738112,
    "samples": 1817,
    "interval_seconds": 0.01,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 55.90846999999485,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  },
  "control_only": true,
  "caveat": "Same sparse shape and training examples; deterministic surrogate-ALIF vs stochastic local network. Updates matched, actual CPU time reported; not evidence of fully matched hardware efficiency."
}
```
