# Forensic r6b

{
  "decision": "AUDIT_EXECUTION_COMPLETE_REQUIRES_ANALYSIS",
  "phase": "r6b",
  "run_class": "HISTORICAL_REPRODUCTION",
  "commands": [
    {
      "command": [
        "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
        "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-forensic-r6-20261111\\scripts\\forensic_replay_worker.py",
        "--source",
        "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\forensic-source-r6b-c39e9db",
        "--output",
        "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-forensic-r6-20261111\\runs\\20261111T100542Z-forensic-r6b-v1-3374ec\\replay",
        "--phase",
        "r6b"
      ],
      "cwd": "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\forensic-source-r6b-c39e9db",
      "exit_code": 0,
      "error": null,
      "seconds": 9.30516009999701,
      "resources": {
        "peak_rss_bytes": 102051840,
        "peak_commit_or_vms_bytes": 101437440,
        "os_process_peak_working_set_bytes": 41201664,
        "os_process_peak_pagefile_commit_bytes": 28798976,
        "samples": 96,
        "interval_seconds": 0.1,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 9.305022200001986,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "log": "r6b.log"
    }
  ],
  "cap": {
    "enforced": true,
    "limit_bytes": 2000000000,
    "kind": "Windows per-process committed memory; shared pages monitored separately"
  },
  "source_commits": {
    "r6d": "4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893",
    "r6b": "c39e9db7771582d9933af402a209917328b71119"
  },
  "fresh_confirmation": false,
  "candidate_equations_changed": false,
  "production_or_live_state_modified": false,
  "error": null
}
