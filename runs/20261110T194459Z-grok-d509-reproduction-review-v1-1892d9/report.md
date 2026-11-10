# Grok exactreplay review

No newhypothesis, no sweep or changedgate.

{
  "run": "runs/20261110T194459Z-grok-d509-reproduction-review-v1-1892d9",
  "decision": "GROK_D509_NUMERICAL_REPRODUCTION_EXACT",
  "reviewed_commit": "d50987862ff80d2bd8556eecfb6af1b259cc591e",
  "all_registered_metrics_and_predictions_exact": true,
  "recorded_results_unchanged": true,
  "lower95_slow_vs_frozen": -0.0028908192607432423,
  "slow_zero_equals_frozen_all8": true,
  "source_commit_of_replay": "d50987862ff80d2bd8556eecfb6af1b259cc591e",
  "recorded_source_commit": "14bf4b2574d77052e3c8d4d682238b2dbd5320c5",
  "output_path_override_only": true,
  "new_hypothesis_or_sweep": false,
  "p2_pilot_run": false,
  "main_training": false,
  "error": null,
  "seconds": 33.20440590000362,
  "resources": {
    "peak_rss_bytes": 90664960,
    "peak_commit_or_vms_bytes": 91254784,
    "os_process_peak_working_set_bytes": 42831872,
    "os_process_peak_pagefile_commit_bytes": 31363072,
    "samples": 504,
    "interval_seconds": 0.05,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 33.204392899991944,
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
  "command": [
    "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
    "-c",
    "import sys,importlib.util\nfrom pathlib import Path\np=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('grok_locked_runner',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=Path(sys.argv[2]);m.main()\n",
    "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\review-grok-d509878\\scripts\\grok_slow_trace_screen.py",
    "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-neural-learning-v2\\runs\\20261110T194459Z-grok-d509-reproduction-review-v1-1892d9\\reproduction"
  ]
}
