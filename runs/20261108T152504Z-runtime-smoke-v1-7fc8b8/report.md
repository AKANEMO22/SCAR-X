# Runtime smoke measurements

```json
{
  "numpy": {
    "warm_prefill_plus_first_byte_p95_seconds": 0.019101399997452972,
    "end_to_end_64byte_generation_p50_seconds": 0.012238999996043276,
    "bytes_per_second_including_prefill": 5229.185392653848,
    "display_characters_per_second": 5229.185392653848,
    "valid_utf8": false,
    "quality": "random weights; meaningless throughput only"
  },
  "native": {
    "warm_prefill_plus_first_byte_p95_seconds": 0.029498229996534064,
    "end_to_end_64byte_generation_p50_seconds": 0.030291399998532142,
    "bytes_per_second_including_prefill": 2112.8108969245827,
    "display_characters_per_second": 2112.8108969245827,
    "valid_utf8": false,
    "quality": "random weights; meaningless throughput only"
  },
  "retrieval": {
    "utf8_bytes_or_byte_tokens": 1000000,
    "ingest_seconds": 0.5615560000005644,
    "queries": 28,
    "exact_identifier_recall": 1.0,
    "p95_seconds": 0.0001747449947288258,
    "scope": "ASCII literal identifier lookup; not semantic, multihop, native-context or language-token evidence"
  },
  "endurance": {
    "seconds": 15.00236749999749,
    "cycles": 6546,
    "eight_hour_pass": false
  },
  "resources": {
    "peak_rss_bytes": 48152576,
    "peak_commit_or_vms_bytes": 671952896,
    "os_process_peak_working_set_bytes": 49111040,
    "os_process_peak_pagefile_commit_bytes": 672727040,
    "samples": 728,
    "interval_seconds": 0.01,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 16.501498900004663,
    "hard_cap_enforced": false,
    "igpu_measured": false,
    "energy_joules": null,
    "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
  }
}
```
