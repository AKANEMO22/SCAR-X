# Phase 2 measured result

Verdict: **NO_GO_SCALE_PIVOT_LOCAL_RULE**

| Seed | Condition | Recall | Delay | Mean forgetting | Worst forgetting |
|---|---|---:|---:|---:|---:|
| 17 | frozen | 0.281 | 0.219 | 0.000 | 0.000 |
| 17 | readout | 1.000 | 0.758 | 0.181 | 0.469 |
| 17 | stdp | 1.000 | 0.883 | 0.237 | 0.406 |
| 17 | full | 1.000 | 0.766 | 0.208 | 0.688 |
| 17 | rate | 1.000 | 1.000 | 0.509 | 0.781 |
| 29 | frozen | 0.258 | 0.250 | 0.000 | 0.000 |
| 29 | readout | 1.000 | 0.992 | 0.242 | 0.500 |
| 29 | stdp | 1.000 | 1.000 | 0.248 | 0.562 |
| 29 | full | 1.000 | 1.000 | 0.239 | 0.531 |
| 29 | rate | 1.000 | 1.000 | 0.472 | 1.000 |
| 43 | frozen | 0.250 | 0.250 | 0.000 | 0.000 |
| 43 | readout | 1.000 | 0.750 | 0.244 | 0.750 |
| 43 | stdp | 1.000 | 1.000 | 0.222 | 0.500 |
| 43 | full | 1.000 | 0.750 | 0.222 | 0.750 |
| 43 | rate | 1.000 | 1.000 | 0.420 | 0.750 |

## Gates

- learns_vs_frozen: True
- hidden_plasticity_contribution: False
- delayed_above_chance: True
- retention: False
- restart_parity: True
- observed_memory: True

Full − frozen: {'mean': 0.7369791666666666, 'lower95': 0.71875, 'upper95': 0.75, 'independent_seeds': 3, 'method': 'paired seed-cluster bootstrap; exploratory with only three clusters'}
Full − readout-only: {'mean': 0.0, 'lower95': 0.0, 'upper95': 0.0, 'independent_seeds': 3, 'method': 'paired seed-cluster bootstrap; exploratory with only three clusters'}

Resources: {'peak_rss_bytes': 79896576, 'peak_commit_or_vms_bytes': 1331998720, 'os_process_peak_working_set_bytes': 45305856, 'os_process_peak_pagefile_commit_bytes': 669847552, 'samples': 4526, 'interval_seconds': 0.01, 'limit_bytes': 2000000000, 'observed_over_limit': False, 'elapsed_seconds': 134.06871229999524, 'hard_cap_enforced': False, 'igpu_measured': False, 'energy_joules': None, 'note': 'Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated.'}

Synthetic recall only. No claim of semantic understanding. Backprop control is a separate run.

Unmeasured: pretrained low-RAM LLM, commercial parity, 8h endurance, iGPU, energy, 10M-token language training
