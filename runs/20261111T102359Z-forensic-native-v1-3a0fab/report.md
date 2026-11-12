# Forensic native

{
  "decision": "AUDIT_EXECUTION_COMPLETE_REQUIRES_ANALYSIS",
  "phase": "native",
  "run_class": "ENGINEERING_CHECK",
  "commands": [
    {
      "command": [
        "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
        "scripts/build_native.py"
      ],
      "cwd": "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-forensic-r6-20261111",
      "exit_code": 0,
      "error": null,
      "seconds": 2.059502500000235,
      "resources": {
        "peak_rss_bytes": 211390464,
        "peak_commit_or_vms_bytes": 152961024,
        "os_process_peak_working_set_bytes": 41652224,
        "os_process_peak_pagefile_commit_bytes": 29237248,
        "samples": 24,
        "interval_seconds": 0.1,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 2.0592016000009608,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "log": "build-native.log"
    },
    {
      "command": [
        "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_native.py",
        "--junitxml=C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-forensic-r6-20261111\\runs\\20261111T102359Z-forensic-native-v1-3a0fab\\native-parity.xml"
      ],
      "cwd": "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-forensic-r6-20261111",
      "exit_code": 0,
      "error": null,
      "seconds": 2.100424900003418,
      "resources": {
        "peak_rss_bytes": 171884544,
        "peak_commit_or_vms_bytes": 136269824,
        "os_process_peak_working_set_bytes": 41680896,
        "os_process_peak_pagefile_commit_bytes": 29237248,
        "samples": 24,
        "interval_seconds": 0.1,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 2.100278799996886,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "log": "native-parity.log"
    }
  ],
  "cap": {
    "enforced": true,
    "limit_bytes": 2000000000,
    "kind": "Windows per-process committed memory; shared pages monitored separately"
  },
  "source_commits": {
    "r6d": "4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893",
    "r6b": "c39e9db7771582d9933af402a209917328b71119",
    "gemini": "8f24a5abde0598201e558eee559564ccde889145"
  },
  "fresh_confirmation": false,
  "candidate_equations_changed": false,
  "production_or_live_state_modified": false,
  "error": null
}
