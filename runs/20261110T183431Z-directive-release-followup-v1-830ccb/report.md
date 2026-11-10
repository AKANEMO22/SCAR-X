# Isolated release baseline

Native/build and prior smoke fixtures, no newqualityclaim.

{
  "run": "runs/20261110T183431Z-directive-release-followup-v1-830ccb",
  "decision": "ISOLATED_RELEASE_BASELINE_PASS",
  "commands": [
    {
      "name": "build_native",
      "command": [
        "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
        "scripts/build_native.py"
      ],
      "exit_code": 0,
      "seconds": 1.3810281000041869,
      "log": "build_native.log"
    },
    {
      "name": "native_parity",
      "command": [
        "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
        "-m",
        "pytest",
        "-q",
        "-rs",
        "tests/test_native.py"
      ],
      "exit_code": 0,
      "seconds": 1.8664888000057545,
      "log": "native_parity.log"
    },
    {
      "name": "release_smoke",
      "command": "Existing smoke_release.main with real-model/temp path guards only",
      "exit_code": 0,
      "seconds": 2.047038100019563,
      "log": "release-smoke.log"
    }
  ],
  "smoke": {
    "port": 61050,
    "data_dir": "C:\\Users\\hachimi\\AppData\\Local\\Temp\\scarx-smoke-kkx19xtb",
    "live_generation1_untouched": true,
    "first_ticks": 2171755,
    "first_hidden": 48,
    "ram_bytes": 46137344,
    "chat_text": "[Hỏi để học] SCAR-X cần bạn hướng dẫn cho: «xin chao». Bạn muốn dạy câu trả lời nào, hoặc có bằng chứng nào để kiểm tra? Nếu chưa biết, trả lời “bỏ qua”.",
    "neural_draft": "Tạm biệt",
    "chat_source": "learning controller clarification; draft is unverified",
    "openai_exposes_neural_draft": "Tạm",
    "accepted_http": 200,
    "updates_before_teach": 241123,
    "accepted_checkpoint": "db0543ce8a9e4ac28dee6c8b9508f322",
    "teach": "accepted",
    "newly_acquired": false,
    "updates_after_teach": 241129,
    "ticks_after_chat": 2171845,
    "ticks_after_restart": 2171845,
    "updates_after_restart": 241129,
    "reject_data_dir": "C:\\Users\\hachimi\\AppData\\Local\\Temp\\scarx-smoke-reject-bmzq4ver",
    "rejected_http": 400,
    "rejected_updates": 241123,
    "rollback": {
      "model_state_sha_before": "e4248753f65e7ba5e064e6d15d35688bf6f96f2a93a676b6eaa3fd9df8d7e422",
      "model_state_sha_after": "e4248753f65e7ba5e064e6d15d35688bf6f96f2a93a676b6eaa3fd9df8d7e422",
      "model_state_sha_unchanged": true,
      "controller_rng_dialogue_match": true,
      "audit_sequence_before": 0,
      "audit_sequence_after": 466,
      "identity": "model.zip plus session controller, app RNG, dialogue, memories and idle policy",
      "excluded_audit": "sequence cursor and the events/trace log may change",
      "regression_test": "tests/test_rollback_full_state.py"
    },
    "rejected_checkpoints_unchanged": true,
    "passed": true
  },
  "error": null,
  "no_live_sqlite_writes": true,
  "no_live_port_used": true,
  "recursive_cleanup_owned_temp_checked": true,
  "research_pilot_run": false,
  "main_training": false,
  "newly_acquired_skill_claim": false,
  "resources": {
    "peak_rss_bytes": 183025664,
    "peak_commit_or_vms_bytes": 135757824,
    "os_process_peak_working_set_bytes": 43917312,
    "os_process_peak_pagefile_commit_bytes": 30756864,
    "samples": 81,
    "interval_seconds": 0.05,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 5.372611099999631,
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
  "seconds": 5.585949600004824
}
