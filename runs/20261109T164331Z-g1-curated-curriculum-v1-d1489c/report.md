# Bounded G1 curated curriculum

Actual localreadouttraining, fixeddevelopmentselection, strictguardedproductacceptance; no automaticdeployment or researchgatepass.

{
  "run": "runs/20261109T164331Z-g1-curated-curriculum-v1-d1489c",
  "decision": "CURATED_SCREEN_FAILED_KEEP_LIVE",
  "selection": {
    "condition": "surface_expanded",
    "order_seed_for_serving_if_eligible": 1801,
    "development_scores": {
      "raw_old": 0.026881720430107527,
      "raw_expanded": 0.09677419354838711,
      "surface_old": 0.03494623655913978,
      "surface_expanded": 0.11559139784946237
    },
    "acceptance_opened": false
  },
  "development_trials": [
    {
      "condition": "raw_old",
      "curriculum_order_seed": 1811,
      "development_by_state": {
        "savedstate": 0.03225806451612903,
        "silent_idle": 0.0,
        "public_idle_probes": 0.06451612903225806,
        "canonicalreset": 0.0
      },
      "development_mean": 0.024193548387096774,
      "training_examples": 3072,
      "label_group_counts": [
        321,
        316,
        322,
        333,
        294,
        277,
        299,
        295,
        615
      ],
      "neural_updates": 14693,
      "neural_ticks": 151026,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.23883176188632382,
      "first256_loss": 0.9646381891970486,
      "last256_loss": 0.10888899578179224,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/raw_old-1811.zip",
      "candidate_sha256": "c4c3488abf016d933d36ad757895c7abd023ffc4569db8a3c64ba5b31337232b",
      "seconds": 16.03980749999755,
      "resources": {
        "peak_rss_bytes": 53260288,
        "peak_commit_or_vms_bytes": 37724160,
        "os_process_peak_working_set_bytes": 54329344,
        "os_process_peak_pagefile_commit_bytes": 38563840,
        "samples": 86,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 16.024713699996937,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    {
      "condition": "raw_old",
      "curriculum_order_seed": 1801,
      "development_by_state": {
        "savedstate": 0.03225806451612903,
        "silent_idle": 0.03225806451612903,
        "public_idle_probes": 0.03225806451612903,
        "canonicalreset": 0.03225806451612903
      },
      "development_mean": 0.03225806451612903,
      "training_examples": 3072,
      "label_group_counts": [
        304,
        330,
        292,
        302,
        340,
        286,
        288,
        315,
        615
      ],
      "neural_updates": 14647,
      "neural_ticks": 151032,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.2374328009000176,
      "first256_loss": 1.0161849454725658,
      "last256_loss": 0.10424767459602778,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/raw_old-1801.zip",
      "candidate_sha256": "8f6dc74edaaf982cc36c3c6e2db1e92418e293f0afb63c9423fcde7c892da2c5",
      "seconds": 16.077621999997064,
      "resources": {
        "peak_rss_bytes": 53424128,
        "peak_commit_or_vms_bytes": 38043648,
        "os_process_peak_working_set_bytes": 54493184,
        "os_process_peak_pagefile_commit_bytes": 38883328,
        "samples": 86,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 16.062101000003167,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    {
      "condition": "raw_old",
      "curriculum_order_seed": 1823,
      "development_by_state": {
        "savedstate": 0.06451612903225806,
        "silent_idle": 0.0,
        "public_idle_probes": 0.03225806451612903,
        "canonicalreset": 0.0
      },
      "development_mean": 0.024193548387096774,
      "training_examples": 3072,
      "label_group_counts": [
        301,
        307,
        309,
        284,
        297,
        277,
        340,
        342,
        615
      ],
      "neural_updates": 14809,
      "neural_ticks": 151419,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.23697140854108398,
      "first256_loss": 0.9410776860465229,
      "last256_loss": 0.11362902191276852,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/raw_old-1823.zip",
      "candidate_sha256": "ac724704d5072ff6e84fa1e0364df4a5ee5bb859fd94df7ee10646ed8e363285",
      "seconds": 21.269957199998316,
      "resources": {
        "peak_rss_bytes": 52850688,
        "peak_commit_or_vms_bytes": 37326848,
        "os_process_peak_working_set_bytes": 54329344,
        "os_process_peak_pagefile_commit_bytes": 38563840,
        "samples": 109,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.260255099994538,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    {
      "condition": "raw_expanded",
      "curriculum_order_seed": 1801,
      "development_by_state": {
        "savedstate": 0.0967741935483871,
        "silent_idle": 0.12903225806451613,
        "public_idle_probes": 0.0967741935483871,
        "canonicalreset": 0.12903225806451613
      },
      "development_mean": 0.11290322580645161,
      "training_examples": 3072,
      "label_group_counts": [
        304,
        330,
        292,
        302,
        340,
        286,
        288,
        315,
        615
      ],
      "neural_updates": 14647,
      "neural_ticks": 151428,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.885759808292272,
      "first256_loss": 2.0663563432809804,
      "last256_loss": 0.573808371830785,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/raw_expanded-1801.zip",
      "candidate_sha256": "34fd6143c242a85783ea793f8fc0b57280c7cfe2989ba218e77886436ec633c7",
      "seconds": 21.40264429999661,
      "resources": {
        "peak_rss_bytes": 53198848,
        "peak_commit_or_vms_bytes": 38150144,
        "os_process_peak_working_set_bytes": 54493184,
        "os_process_peak_pagefile_commit_bytes": 38985728,
        "samples": 110,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.392874299999676,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    {
      "condition": "raw_expanded",
      "curriculum_order_seed": 1811,
      "development_by_state": {
        "savedstate": 0.0967741935483871,
        "silent_idle": 0.03225806451612903,
        "public_idle_probes": 0.0967741935483871,
        "canonicalreset": 0.03225806451612903
      },
      "development_mean": 0.06451612903225806,
      "training_examples": 3072,
      "label_group_counts": [
        321,
        316,
        322,
        333,
        294,
        277,
        299,
        295,
        615
      ],
      "neural_updates": 14693,
      "neural_ticks": 151569,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.8714767921452751,
      "first256_loss": 1.9702142951592816,
      "last256_loss": 0.49595074420977336,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/raw_expanded-1811.zip",
      "candidate_sha256": "349a52003928d3877726af073f9de058d40b8ed2e282a8694213b18874716451",
      "seconds": 21.338463899999624,
      "resources": {
        "peak_rss_bytes": 56311808,
        "peak_commit_or_vms_bytes": 40914944,
        "os_process_peak_working_set_bytes": 57753600,
        "os_process_peak_pagefile_commit_bytes": 42332160,
        "samples": 110,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.329049200001464,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    {
      "condition": "raw_expanded",
      "curriculum_order_seed": 1823,
      "development_by_state": {
        "savedstate": 0.0967741935483871,
        "silent_idle": 0.12903225806451613,
        "public_idle_probes": 0.0967741935483871,
        "canonicalreset": 0.12903225806451613
      },
      "development_mean": 0.11290322580645161,
      "training_examples": 3072,
      "label_group_counts": [
        301,
        307,
        309,
        284,
        297,
        277,
        340,
        342,
        615
      ],
      "neural_updates": 14809,
      "neural_ticks": 151893,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.8717112639141563,
      "first256_loss": 1.8384251708327728,
      "last256_loss": 0.5501400879827846,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/raw_expanded-1823.zip",
      "candidate_sha256": "b00f237dd197a2a644d0eb1610ec898a4187d862c7dea6fd1ac1a6a300442d2f",
      "seconds": 21.62595059999876,
      "resources": {
        "peak_rss_bytes": 56758272,
        "peak_commit_or_vms_bytes": 41476096,
        "os_process_peak_working_set_bytes": 58421248,
        "os_process_peak_pagefile_commit_bytes": 43171840,
        "samples": 111,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.617723000003025,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    {
      "condition": "surface_old",
      "curriculum_order_seed": 1801,
      "development_by_state": {
        "savedstate": 0.03225806451612903,
        "silent_idle": 0.03225806451612903,
        "public_idle_probes": 0.03225806451612903,
        "canonicalreset": 0.03225806451612903
      },
      "development_mean": 0.03225806451612903,
      "training_examples": 3072,
      "label_group_counts": [
        304,
        330,
        292,
        302,
        340,
        286,
        288,
        315,
        615
      ],
      "neural_updates": 14647,
      "neural_ticks": 150846,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.28064904359622217,
      "first256_loss": 1.2205699744009748,
      "last256_loss": 0.11721486199846978,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/surface_old-1801.zip",
      "candidate_sha256": "dd3cbc25774c0adb784870a65a47fe1e0ced4ba1a99e8645358e626762c85fbf",
      "seconds": 21.834485100000165,
      "resources": {
        "peak_rss_bytes": 53170176,
        "peak_commit_or_vms_bytes": 38309888,
        "os_process_peak_working_set_bytes": 57753600,
        "os_process_peak_pagefile_commit_bytes": 42332160,
        "samples": 112,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.825679800000216,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    {
      "condition": "surface_old",
      "curriculum_order_seed": 1811,
      "development_by_state": {
        "savedstate": 0.03225806451612903,
        "silent_idle": 0.0,
        "public_idle_probes": 0.06451612903225806,
        "canonicalreset": 0.0
      },
      "development_mean": 0.024193548387096774,
      "training_examples": 3072,
      "label_group_counts": [
        321,
        316,
        322,
        333,
        294,
        277,
        299,
        295,
        615
      ],
      "neural_updates": 14693,
      "neural_ticks": 150942,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.28715618540375154,
      "first256_loss": 1.186628320864603,
      "last256_loss": 0.12067316711247002,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/surface_old-1811.zip",
      "candidate_sha256": "9fc1ba28e6d48c28663060662387a3e2d656b6c6bdb6c3b2257c38ce754421ea",
      "seconds": 21.968475800000306,
      "resources": {
        "peak_rss_bytes": 54251520,
        "peak_commit_or_vms_bytes": 39231488,
        "os_process_peak_working_set_bytes": 58421248,
        "os_process_peak_pagefile_commit_bytes": 43171840,
        "samples": 113,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.96018520000507,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    {
      "condition": "surface_old",
      "curriculum_order_seed": 1823,
      "development_by_state": {
        "savedstate": 0.06451612903225806,
        "silent_idle": 0.03225806451612903,
        "public_idle_probes": 0.06451612903225806,
        "canonicalreset": 0.03225806451612903
      },
      "development_mean": 0.04838709677419355,
      "training_examples": 3072,
      "label_group_counts": [
        301,
        307,
        309,
        284,
        297,
        277,
        340,
        342,
        615
      ],
      "neural_updates": 14809,
      "neural_ticks": 151335,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.27996647657915485,
      "first256_loss": 1.1442407863485187,
      "last256_loss": 0.12091495386472007,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/surface_old-1823.zip",
      "candidate_sha256": "0b669b9a870465432cecfd1c1661741aaea9cc8ed3412568f2fc3f26c9514668",
      "seconds": 21.546818100003293,
      "resources": {
        "peak_rss_bytes": 53604352,
        "peak_commit_or_vms_bytes": 38166528,
        "os_process_peak_working_set_bytes": 57753600,
        "os_process_peak_pagefile_commit_bytes": 42332160,
        "samples": 111,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.537345900003857,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    {
      "condition": "surface_expanded",
      "curriculum_order_seed": 1801,
      "development_by_state": {
        "savedstate": 0.12903225806451613,
        "silent_idle": 0.12903225806451613,
        "public_idle_probes": 0.0967741935483871,
        "canonicalreset": 0.12903225806451613
      },
      "development_mean": 0.12096774193548387,
      "training_examples": 3072,
      "label_group_counts": [
        304,
        330,
        292,
        302,
        340,
        286,
        288,
        315,
        615
      ],
      "neural_updates": 14647,
      "neural_ticks": 151395,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.9357922605297163,
      "first256_loss": 2.3049465118680907,
      "last256_loss": 0.5963089889878994,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/surface_expanded-1801.zip",
      "candidate_sha256": "596d3b1ac43858f3116c2ab1d6eb6fd941619769d77efe9a242133fd0f3172e5",
      "seconds": 21.830107900001167,
      "resources": {
        "peak_rss_bytes": 54018048,
        "peak_commit_or_vms_bytes": 39120896,
        "os_process_peak_working_set_bytes": 58421248,
        "os_process_peak_pagefile_commit_bytes": 43171840,
        "samples": 112,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.82134049999877,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    {
      "condition": "surface_expanded",
      "curriculum_order_seed": 1811,
      "development_by_state": {
        "savedstate": 0.0967741935483871,
        "silent_idle": 0.12903225806451613,
        "public_idle_probes": 0.12903225806451613,
        "canonicalreset": 0.12903225806451613
      },
      "development_mean": 0.12096774193548387,
      "training_examples": 3072,
      "label_group_counts": [
        321,
        316,
        322,
        333,
        294,
        277,
        299,
        295,
        615
      ],
      "neural_updates": 14693,
      "neural_ticks": 151491,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.9192227189524161,
      "first256_loss": 2.1865625784527136,
      "last256_loss": 0.49895825587602843,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/surface_expanded-1811.zip",
      "candidate_sha256": "8e2adf6cf5bfff89caa91c349eecd5a874beb0fffeb6e9e4140c3cfbbd42d1e3",
      "seconds": 21.739546700002393,
      "resources": {
        "peak_rss_bytes": 53440512,
        "peak_commit_or_vms_bytes": 37859328,
        "os_process_peak_working_set_bytes": 57753600,
        "os_process_peak_pagefile_commit_bytes": 42332160,
        "samples": 112,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.731122299999697,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    },
    {
      "condition": "surface_expanded",
      "curriculum_order_seed": 1823,
      "development_by_state": {
        "savedstate": 0.0967741935483871,
        "silent_idle": 0.12903225806451613,
        "public_idle_probes": 0.0967741935483871,
        "canonicalreset": 0.0967741935483871
      },
      "development_mean": 0.10483870967741934,
      "training_examples": 3072,
      "label_group_counts": [
        301,
        307,
        309,
        284,
        297,
        277,
        340,
        342,
        615
      ],
      "neural_updates": 14809,
      "neural_ticks": 151764,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.9214344785262448,
      "first256_loss": 2.0503555103871642,
      "last256_loss": 0.5577952484327555,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/surface_expanded-1823.zip",
      "candidate_sha256": "1b60fadf97846405479909c019b7ad2c5620468c9ae3909740a213281fbe39b0",
      "seconds": 21.944360200002848,
      "resources": {
        "peak_rss_bytes": 54243328,
        "peak_commit_or_vms_bytes": 39120896,
        "os_process_peak_working_set_bytes": 58421248,
        "os_process_peak_pagefile_commit_bytes": 43171840,
        "samples": 113,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.9341940000013,
        "hard_cap_enforced": false,
        "igpu_measured": false,
        "energy_joules": null,
        "note": "Tree sums may double-count shared pages; Windows pagefile is commit, not bytes swapped. No iGPU allocated."
      },
      "cap": {
        "enforced": true,
        "limit_bytes": 2000000000,
        "kind": "Windows per-process committed memory; shared pages monitored separately"
      }
    }
  ],
  "conditions": [
    {
      "condition": "surface_expanded",
      "curriculum_order_seed": 1801,
      "development_by_state": {
        "savedstate": 0.12903225806451613,
        "silent_idle": 0.12903225806451613,
        "public_idle_probes": 0.0967741935483871,
        "canonicalreset": 0.12903225806451613
      },
      "development_mean": 0.12096774193548387,
      "training_examples": 3072,
      "label_group_counts": [
        304,
        330,
        292,
        302,
        340,
        286,
        288,
        315,
        615
      ],
      "neural_updates": 14647,
      "neural_ticks": 151395,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.9357922605297163,
      "first256_loss": 2.3049465118680907,
      "last256_loss": 0.5963089889878994,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/surface_expanded-1801.zip",
      "candidate_sha256": "596d3b1ac43858f3116c2ab1d6eb6fd941619769d77efe9a242133fd0f3172e5",
      "seconds": 21.830107900001167,
      "resources": {
        "peak_rss_bytes": 54018048,
        "peak_commit_or_vms_bytes": 39120896,
        "os_process_peak_working_set_bytes": 58421248,
        "os_process_peak_pagefile_commit_bytes": 43171840,
        "samples": 112,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.82134049999877,
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
      "states": {
        "savedstate": {
          "known": 0.6507936507936508,
          "antigravity": 0.22916666666666666,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 0,
          "proof_accuracy": 1.0,
          "pass": false
        },
        "silent_idle": {
          "known": 0.6984126984126984,
          "antigravity": 0.125,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 0,
          "proof_accuracy": 1.0,
          "pass": false
        },
        "public_idle_probes": {
          "known": 0.6507936507936508,
          "antigravity": 0.2708333333333333,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 0,
          "proof_accuracy": 1.0,
          "pass": false
        },
        "canonicalreset": {
          "known": 0.6825396825396826,
          "antigravity": 0.125,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 0,
          "proof_accuracy": 1.0,
          "pass": false
        }
      },
      "screen_pass": false,
      "lost_old_correct_digits": 0,
      "development_pass": false
    },
    {
      "condition": "surface_expanded",
      "curriculum_order_seed": 1811,
      "development_by_state": {
        "savedstate": 0.0967741935483871,
        "silent_idle": 0.12903225806451613,
        "public_idle_probes": 0.12903225806451613,
        "canonicalreset": 0.12903225806451613
      },
      "development_mean": 0.12096774193548387,
      "training_examples": 3072,
      "label_group_counts": [
        321,
        316,
        322,
        333,
        294,
        277,
        299,
        295,
        615
      ],
      "neural_updates": 14693,
      "neural_ticks": 151491,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.9192227189524161,
      "first256_loss": 2.1865625784527136,
      "last256_loss": 0.49895825587602843,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/surface_expanded-1811.zip",
      "candidate_sha256": "8e2adf6cf5bfff89caa91c349eecd5a874beb0fffeb6e9e4140c3cfbbd42d1e3",
      "seconds": 21.739546700002393,
      "resources": {
        "peak_rss_bytes": 53440512,
        "peak_commit_or_vms_bytes": 37859328,
        "os_process_peak_working_set_bytes": 57753600,
        "os_process_peak_pagefile_commit_bytes": 42332160,
        "samples": 112,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.731122299999697,
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
      "states": {
        "savedstate": {
          "known": 0.6349206349206349,
          "antigravity": 0.2708333333333333,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 2,
          "proof_accuracy": 0.6666666666666666,
          "pass": false
        },
        "silent_idle": {
          "known": 0.6349206349206349,
          "antigravity": 0.16666666666666666,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 0,
          "proof_accuracy": 1.0,
          "pass": false
        },
        "public_idle_probes": {
          "known": 0.5873015873015873,
          "antigravity": 0.20833333333333334,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 0,
          "proof_accuracy": 1.0,
          "pass": false
        },
        "canonicalreset": {
          "known": 0.6349206349206349,
          "antigravity": 0.14583333333333334,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 0,
          "proof_accuracy": 1.0,
          "pass": false
        }
      },
      "screen_pass": false,
      "lost_old_correct_digits": 0,
      "development_pass": false
    },
    {
      "condition": "surface_expanded",
      "curriculum_order_seed": 1823,
      "development_by_state": {
        "savedstate": 0.0967741935483871,
        "silent_idle": 0.12903225806451613,
        "public_idle_probes": 0.0967741935483871,
        "canonicalreset": 0.0967741935483871
      },
      "development_mean": 0.10483870967741934,
      "training_examples": 3072,
      "label_group_counts": [
        301,
        307,
        309,
        284,
        297,
        277,
        340,
        342,
        615
      ],
      "neural_updates": 14809,
      "neural_ticks": 151764,
      "hidden_weights_unchanged": true,
      "allocation_unchanged": true,
      "mean_loss": 0.9214344785262448,
      "first256_loss": 2.0503555103871642,
      "last256_loss": 0.5577952484327555,
      "candidate_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/surface_expanded-1823.zip",
      "candidate_sha256": "1b60fadf97846405479909c019b7ad2c5620468c9ae3909740a213281fbe39b0",
      "seconds": 21.944360200002848,
      "resources": {
        "peak_rss_bytes": 54243328,
        "peak_commit_or_vms_bytes": 39120896,
        "os_process_peak_working_set_bytes": 58421248,
        "os_process_peak_pagefile_commit_bytes": 43171840,
        "samples": 113,
        "interval_seconds": 0.2,
        "limit_bytes": 2000000000,
        "observed_over_limit": false,
        "elapsed_seconds": 21.9341940000013,
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
      "states": {
        "savedstate": {
          "known": 0.5238095238095238,
          "antigravity": 0.16666666666666666,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 0,
          "proof_accuracy": 1.0,
          "pass": false
        },
        "silent_idle": {
          "known": 0.5873015873015873,
          "antigravity": 0.14583333333333334,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 0,
          "proof_accuracy": 1.0,
          "pass": false
        },
        "public_idle_probes": {
          "known": 0.5873015873015873,
          "antigravity": 0.16666666666666666,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 0,
          "proof_accuracy": 1.0,
          "pass": false
        },
        "canonicalreset": {
          "known": 0.5873015873015873,
          "antigravity": 0.14583333333333334,
          "source_antigravity": 0.0,
          "lost_personal_runtime_guards": 0,
          "proof_accuracy": 1.0,
          "pass": false
        }
      },
      "screen_pass": false,
      "lost_old_correct_digits": 0,
      "development_pass": false
    }
  ],
  "selected": null,
  "all_order_passes": {
    "surface_expanded": false
  },
  "original_source_sha256": "58f965b67134f5a03a1f0eed8fe318fa478c539582e515ba6a910dd46907ccf6",
  "original_source_private_path": "local/g1-curated-curriculum/20261109T164331Z-g1-curated-curriculum-v1-d1489c/source-model.zip",
  "curated_train_prompts": 117,
  "curated_development_prompts": 31,
  "personal_guards": 5,
  "proof_replay_types": 3,
  "deployed": false,
  "new_hidden_rule": false,
  "normal_chat_or_vision_proven": false,
  "lifelong_P2_pass": false,
  "resources": {
    "peak_rss_bytes": 184143872,
    "peak_commit_or_vms_bytes": 147959808,
    "os_process_peak_working_set_bytes": 196620288,
    "os_process_peak_pagefile_commit_bytes": 160419840,
    "samples": 1570,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 164.287499500002,
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
  "limits": [
    "Same sourceinitialization,3curriculumorders, not independentmodel confirmation.",
    "Closed8intent stylematching, not generalbilingualconversation or normalphotos.",
    "Privateweights/rawguardresponses staylocal; priorAGY48 diagnostic was alreadyinspected, nevertrained."
  ]
}
