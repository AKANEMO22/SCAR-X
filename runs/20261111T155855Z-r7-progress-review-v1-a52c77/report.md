# R7 review evidence

Raw metrics and bounded engineering/reproduction, not production approval.

{
  "p0_command": {
    "command": [
      "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "-p",
      "no:cacheprovider",
      "tests/test_r7_hidden_plasticity.py",
      "--junitxml=C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-forensic-r6-20261111\\runs\\20261111T155855Z-r7-progress-review-v1-a52c77\\gemini-r7-p0.xml"
    ],
    "cwd": "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\review-gemini-r7-64441fb",
    "exit_code": 0,
    "error": null,
    "seconds": 2.575796500001161,
    "resources": {
      "peak_rss_bytes": 178401280,
      "peak_commit_or_vms_bytes": 173912064,
      "os_process_peak_working_set_bytes": 45867008,
      "os_process_peak_pagefile_commit_bytes": 31559680,
      "samples": 29,
      "interval_seconds": 0.1,
      "limit_bytes": 2000000000,
      "observed_over_limit": false,
      "elapsed_seconds": 2.575409500001115,
      "hard_cap_enforced": false,
      "igpu_measured": false,
      "energy_joules": null,
      "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
    },
    "log": "gemini-r7-p0.log"
  },
  "historical_check_command": {
    "command": [
      "C:\\Users\\hachimi\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-c",
      "from pathlib import Path\nimport hashlib,json,sys\nfrom scarx.runtime.limits import apply_memory_limit\nfrom scarx.evaluation import r7_hidden_plasticity as R\ncap=apply_memory_limit()\nvalue=R.run_p1_task_seed('strict_binding',140011)\nold=json.loads(Path('artifacts/gemini-r7/dev-screen.json').read_text())\nexpected=[r for r in old['dev_p1_trials'] if r['task']=='strict_binding' and r['seed']==140011][0]['conditions']\nmetrics=('id_direct_acc','ood_direct_acc','id_probe_acc','ood_probe_acc')\ncomparison={c:{k:value[c].get(k)==expected[c].get(k) for k in metrics} for c in expected}\nresult={'run_class':'HISTORICAL_SELECTED_POSITIVE_TASK_REPRODUCTION','seed':140011,'task':'strict_binding','all_primary_scores_exact':all(all(v.values()) for v in comparison.values()),'comparison':comparison,'conditions':value,'cap':cap,'fresh_confirmation':False,'whole_R7_campaign_reproduced':False}\nPath(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\\n')\nprint(json.dumps({'all_primary_scores_exact':result['all_primary_scores_exact'],'learned_id_probe':value['learned_ew_he']['id_probe_acc'],'frozen_id_probe':value['frozen_hidden']['id_probe_acc']}))\n",
      "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\scarx-forensic-r6-20261111\\runs\\20261111T155855Z-r7-progress-review-v1-a52c77\\positive-history.json"
    ],
    "cwd": "C:\\Users\\hachimi\\Downloads\\model agent\\scar-x\\local\\worktrees\\review-gemini-r7-64441fb",
    "exit_code": 0,
    "error": null,
    "seconds": 9.229295999997703,
    "resources": {
      "peak_rss_bytes": 97673216,
      "peak_commit_or_vms_bytes": 95182848,
      "os_process_peak_working_set_bytes": 45903872,
      "os_process_peak_pagefile_commit_bytes": 31559680,
      "samples": 101,
      "interval_seconds": 0.1,
      "limit_bytes": 2000000000,
      "observed_over_limit": false,
      "elapsed_seconds": 9.2291464000009,
      "hard_cap_enforced": false,
      "igpu_measured": false,
      "energy_joules": null,
      "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
    },
    "log": "gemini-r7-positive-history.log"
  }
}
