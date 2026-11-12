# Isolated R8 snapshot state repair

No new quality experiment.

{
  "source_sha": "fe472b3b1257cdb264805bf79fd53e5647e3568d",
  "cap": {
    "enforced": true,
    "limit_bytes": 2000000000,
    "kind": "Windows per-process committed memory; shared pages monitored separately"
  },
  "cold_continuation": [
    {
      "kind": "mtsg",
      "next_forward_and_update_exact": true,
      "expected": {
        "next_forward_state_sha256": "d14fdb4ef94cce4eb136232a6a0e30e5b7e18e84b9d1e90a36ad3866b47b66b9",
        "next_fixture_loss": 0.818353819956022,
        "controller_rng_draw": 3884,
        "state_sha256": "21c805f957a903ae396c845a8337530e79d0e10a79ff6e9c57e5edb0572adcda",
        "hidden_updates": 1,
        "scope": "One technical fixture update, not acquisition/retention or a live session"
      },
      "actual": {
        "next_forward_state_sha256": "d14fdb4ef94cce4eb136232a6a0e30e5b7e18e84b9d1e90a36ad3866b47b66b9",
        "next_fixture_loss": 0.818353819956022,
        "controller_rng_draw": 3884,
        "state_sha256": "21c805f957a903ae396c845a8337530e79d0e10a79ff6e9c57e5edb0572adcda",
        "hidden_updates": 1,
        "scope": "One technical fixture update, not acquisition/retention or a live session"
      },
      "saved_checkpoint_unchanged": true,
      "snapshot_sha256": "c3116866c7aa9b77fb1b16b31abf2260dfbc53b97c3146d076831610224b0602"
    },
    {
      "kind": "bf",
      "next_forward_and_update_exact": true,
      "expected": {
        "next_forward_state_sha256": "78d04f6965fb4f7e164605107e10f93cac4e3c897f9c7940fce63d4f9fd26358",
        "next_fixture_loss": 0.818353819956022,
        "controller_rng_draw": 3884,
        "state_sha256": "505db03892c743a45da7b9b6d20cc1b0bd1f641ba2a529db644fb888c7d489a5",
        "hidden_updates": 1,
        "scope": "One technical fixture update, not acquisition/retention or a live session"
      },
      "actual": {
        "next_forward_state_sha256": "78d04f6965fb4f7e164605107e10f93cac4e3c897f9c7940fce63d4f9fd26358",
        "next_fixture_loss": 0.818353819956022,
        "controller_rng_draw": 3884,
        "state_sha256": "505db03892c743a45da7b9b6d20cc1b0bd1f641ba2a529db644fb888c7d489a5",
        "hidden_updates": 1,
        "scope": "One technical fixture update, not acquisition/retention or a live session"
      },
      "saved_checkpoint_unchanged": true,
      "snapshot_sha256": "d3137440072fc3b1d6389b9d7d7bc9bef58637ac53aa34bb2f16891b15844cad"
    }
  ],
  "commands": [
    {
      "command": [
        "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
        "-X",
        "utf8",
        "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-r8-matched-controls-20261112\\scripts\\r8_snapshot_cold.py",
        "mtsg",
        "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-r8-matched-controls-20261112\\runs\\20261112T070737Z-r8-snapshot-repair-p0-v1-149782\\mtsg-state.zip",
        "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-r8-matched-controls-20261112\\runs\\20261112T070737Z-r8-snapshot-repair-p0-v1-149782\\mtsg-cold.json"
      ],
      "cwd": "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-r8-matched-controls-20261112",
      "exit_code": 0,
      "error": null,
      "seconds": 0.5296553000007407,
      "resources": {
        "peak_rss_bytes": 85811200,
        "peak_commit_or_vms_bytes": 51585024,
        "os_process_peak_working_set_bytes": 44830720,
        "os_process_peak_pagefile_commit_bytes": 29659136,
        "samples": 7,
        "interval_seconds": 0.1,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 0.5292899000014586,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "log": "mtsg-cold.log"
    },
    {
      "command": [
        "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
        "-X",
        "utf8",
        "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-r8-matched-controls-20261112\\scripts\\r8_snapshot_cold.py",
        "bf",
        "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-r8-matched-controls-20261112\\runs\\20261112T070737Z-r8-snapshot-repair-p0-v1-149782\\bf-state.zip",
        "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-r8-matched-controls-20261112\\runs\\20261112T070737Z-r8-snapshot-repair-p0-v1-149782\\bf-cold.json"
      ],
      "cwd": "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-r8-matched-controls-20261112",
      "exit_code": 0,
      "error": null,
      "seconds": 0.5312881999998353,
      "resources": {
        "peak_rss_bytes": 78368768,
        "peak_commit_or_vms_bytes": 46608384,
        "os_process_peak_working_set_bytes": 44957696,
        "os_process_peak_pagefile_commit_bytes": 29659136,
        "samples": 7,
        "interval_seconds": 0.1,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 0.531023899999127,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "log": "bf-cold.log"
    },
    {
      "command": [
        "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
        "-X",
        "utf8",
        "-m",
        "pytest",
        "-q",
        "-rs",
        "-p",
        "no:cacheprovider",
        "tests/test_r8_snapshot_adapter.py",
        "tests/test_research_gate_repairs.py",
        "tests/test_cross_snn_review.py",
        "tests/test_forensic_evaluator.py",
        "tests/test_forensic_mutations.py",
        "tests/test_neural.py",
        "tests/test_native.py",
        "tests/test_generation1.py",
        "--junitxml=C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-r8-matched-controls-20261112\\runs\\20261112T070737Z-r8-snapshot-repair-p0-v1-149782\\repair-regression.xml"
      ],
      "cwd": "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-r8-matched-controls-20261112",
      "exit_code": 0,
      "error": null,
      "seconds": 8.787727499999164,
      "resources": {
        "peak_rss_bytes": 236695552,
        "peak_commit_or_vms_bytes": 193085440,
        "os_process_peak_working_set_bytes": 44978176,
        "os_process_peak_pagefile_commit_bytes": 29659136,
        "samples": 91,
        "interval_seconds": 0.1,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 8.787588200000755,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "log": "repair-regression.log"
    }
  ],
  "actual_cold_verification_fixture_update_calls": 4,
  "pytest_fixture_updates_counted_separately": true,
  "new_quality_training": false,
  "candidate_equations_changed": false,
  "main_sha_before": "e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541",
  "main_sha_after": "e94ad1e7a4029d8d41c8e0e1aafc0fa3bf430541",
  "peer_tracked_unchanged": true,
  "limits": [
    "H48 D8 O2 K8 R12 only; explicit matching peer template required.",
    "Model plus explicit caller JSON controller only, no live SQLite/Generation/Hermes migration.",
    "Next-forward and next-episode update continuation, not a new mid-episode training controller."
  ],
  "decision": "P0_STATE_REPAIR_PASS_RESEARCH_ONLY"
}
