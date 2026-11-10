# Bounded generic neural-token search

Weights shared/unchanged, no target/template access, no deployment or learninggain claim.

{
  "run": "runs/20261109T173540Z-g1-bounded-beam-diagnostic-v1-f972ce",
  "source_run": "runs/20261109T164331Z-g1-curated-curriculum-v1-d1489c",
  "decision": "STOP_G1_WORD_DECODER_SEARCH",
  "selection": {
    "width": 4,
    "alpha": 0.6,
    "development_scores": {
      "(1, 0.0)": 0.12903225806451613,
      "(1, 0.6)": 0.12903225806451613,
      "(4, 0.0)": 0.18279569892473116,
      "(4, 0.6)": 0.21505376344086022,
      "(8, 0.0)": 0.18279569892473116,
      "(8, 0.6)": 0.21505376344086022
    },
    "prior_opened": false
  },
  "development": [
    {
      "order": 1801,
      "width": 1,
      "alpha": 0.0,
      "summary": {
        "strict_style_exact": 0.12903225806451613,
        "normalized_style_exact": 0.16129032258064516,
        "first_token_correct": 0.3548387096774194,
        "stopped": 1.0,
        "mean_seconds": 0.0013078064526311091,
        "p95_seconds": 0.0019451000043773092,
        "max_seconds": 0.0020185000030323863,
        "mean_computation_ticks": 27.774193548387096,
        "peak_search_state_bytes": 16280,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1801,
      "width": 1,
      "alpha": 0.6,
      "summary": {
        "strict_style_exact": 0.12903225806451613,
        "normalized_style_exact": 0.16129032258064516,
        "first_token_correct": 0.3548387096774194,
        "stopped": 1.0,
        "mean_seconds": 0.0013098258064166012,
        "p95_seconds": 0.0018623500000103377,
        "max_seconds": 0.0021074999967822805,
        "mean_computation_ticks": 27.774193548387096,
        "peak_search_state_bytes": 16280,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1801,
      "width": 4,
      "alpha": 0.0,
      "summary": {
        "strict_style_exact": 0.16129032258064516,
        "normalized_style_exact": 0.1935483870967742,
        "first_token_correct": 0.2903225806451613,
        "stopped": 1.0,
        "mean_seconds": 0.0025385483867496313,
        "p95_seconds": 0.00415104999410687,
        "max_seconds": 0.004193400003714487,
        "mean_computation_ticks": 62.225806451612904,
        "peak_search_state_bytes": 130240,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1801,
      "width": 4,
      "alpha": 0.6,
      "summary": {
        "strict_style_exact": 0.16129032258064516,
        "normalized_style_exact": 0.1935483870967742,
        "first_token_correct": 0.3225806451612903,
        "stopped": 1.0,
        "mean_seconds": 0.003594883872034599,
        "p95_seconds": 0.0076977499993517995,
        "max_seconds": 0.01407160000235308,
        "mean_computation_ticks": 76.93548387096774,
        "peak_search_state_bytes": 130240,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1801,
      "width": 8,
      "alpha": 0.0,
      "summary": {
        "strict_style_exact": 0.16129032258064516,
        "normalized_style_exact": 0.1935483870967742,
        "first_token_correct": 0.2903225806451613,
        "stopped": 1.0,
        "mean_seconds": 0.0045801290346594945,
        "p95_seconds": 0.00834430000395514,
        "max_seconds": 0.012595599997439422,
        "mean_computation_ticks": 109.45161290322581,
        "peak_search_state_bytes": 260480,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1801,
      "width": 8,
      "alpha": 0.6,
      "summary": {
        "strict_style_exact": 0.16129032258064516,
        "normalized_style_exact": 0.1935483870967742,
        "first_token_correct": 0.3225806451612903,
        "stopped": 1.0,
        "mean_seconds": 0.005929545161404436,
        "p95_seconds": 0.01343895000172779,
        "max_seconds": 0.019431900000199676,
        "mean_computation_ticks": 139.6451612903226,
        "peak_search_state_bytes": 260480,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1811,
      "width": 1,
      "alpha": 0.0,
      "summary": {
        "strict_style_exact": 0.12903225806451613,
        "normalized_style_exact": 0.12903225806451613,
        "first_token_correct": 0.3225806451612903,
        "stopped": 0.967741935483871,
        "mean_seconds": 0.0016829516121839743,
        "p95_seconds": 0.0027801499963970855,
        "max_seconds": 0.009435599989956245,
        "mean_computation_ticks": 31.93548387096774,
        "peak_search_state_bytes": 16280,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1811,
      "width": 1,
      "alpha": 0.6,
      "summary": {
        "strict_style_exact": 0.12903225806451613,
        "normalized_style_exact": 0.12903225806451613,
        "first_token_correct": 0.3225806451612903,
        "stopped": 0.967741935483871,
        "mean_seconds": 0.0014398967731751562,
        "p95_seconds": 0.0022552499940502457,
        "max_seconds": 0.0032091000030050054,
        "mean_computation_ticks": 31.93548387096774,
        "peak_search_state_bytes": 16280,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1811,
      "width": 4,
      "alpha": 0.0,
      "summary": {
        "strict_style_exact": 0.1935483870967742,
        "normalized_style_exact": 0.1935483870967742,
        "first_token_correct": 0.3870967741935484,
        "stopped": 1.0,
        "mean_seconds": 0.0028164645177199536,
        "p95_seconds": 0.004704300008597784,
        "max_seconds": 0.012546800004201941,
        "mean_computation_ticks": 61.83870967741935,
        "peak_search_state_bytes": 130240,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1811,
      "width": 4,
      "alpha": 0.6,
      "summary": {
        "strict_style_exact": 0.25806451612903225,
        "normalized_style_exact": 0.25806451612903225,
        "first_token_correct": 0.3870967741935484,
        "stopped": 1.0,
        "mean_seconds": 0.003935051611587105,
        "p95_seconds": 0.008652750002511311,
        "max_seconds": 0.012246599988429807,
        "mean_computation_ticks": 93.96774193548387,
        "peak_search_state_bytes": 130240,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1811,
      "width": 8,
      "alpha": 0.0,
      "summary": {
        "strict_style_exact": 0.1935483870967742,
        "normalized_style_exact": 0.1935483870967742,
        "first_token_correct": 0.3870967741935484,
        "stopped": 1.0,
        "mean_seconds": 0.004242429032831663,
        "p95_seconds": 0.006157399999210611,
        "max_seconds": 0.0112892000033753,
        "mean_computation_ticks": 107.90322580645162,
        "peak_search_state_bytes": 260480,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1811,
      "width": 8,
      "alpha": 0.6,
      "summary": {
        "strict_style_exact": 0.25806451612903225,
        "normalized_style_exact": 0.25806451612903225,
        "first_token_correct": 0.3870967741935484,
        "stopped": 1.0,
        "mean_seconds": 0.0070018999995813975,
        "p95_seconds": 0.013050649999058805,
        "max_seconds": 0.02026109999860637,
        "mean_computation_ticks": 175.25806451612902,
        "peak_search_state_bytes": 260480,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1823,
      "width": 1,
      "alpha": 0.0,
      "summary": {
        "strict_style_exact": 0.0967741935483871,
        "normalized_style_exact": 0.0967741935483871,
        "first_token_correct": 0.3548387096774194,
        "stopped": 1.0,
        "mean_seconds": 0.0016985612897007095,
        "p95_seconds": 0.0025040000036824495,
        "max_seconds": 0.010399499995401129,
        "mean_computation_ticks": 30.967741935483872,
        "peak_search_state_bytes": 16280,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1823,
      "width": 1,
      "alpha": 0.6,
      "summary": {
        "strict_style_exact": 0.0967741935483871,
        "normalized_style_exact": 0.0967741935483871,
        "first_token_correct": 0.3548387096774194,
        "stopped": 1.0,
        "mean_seconds": 0.0013983999998609146,
        "p95_seconds": 0.0020321499978308566,
        "max_seconds": 0.002846000003046356,
        "mean_computation_ticks": 30.967741935483872,
        "peak_search_state_bytes": 16280,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1823,
      "width": 4,
      "alpha": 0.0,
      "summary": {
        "strict_style_exact": 0.16129032258064516,
        "normalized_style_exact": 0.16129032258064516,
        "first_token_correct": 0.3225806451612903,
        "stopped": 1.0,
        "mean_seconds": 0.002936809675321133,
        "p95_seconds": 0.007013699992967304,
        "max_seconds": 0.012536099995486438,
        "mean_computation_ticks": 59.12903225806452,
        "peak_search_state_bytes": 130240,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1823,
      "width": 4,
      "alpha": 0.6,
      "summary": {
        "strict_style_exact": 0.1935483870967742,
        "normalized_style_exact": 0.1935483870967742,
        "first_token_correct": 0.3225806451612903,
        "stopped": 1.0,
        "mean_seconds": 0.0032833612893873286,
        "p95_seconds": 0.006330650001473259,
        "max_seconds": 0.0072576999955344945,
        "mean_computation_ticks": 81.96774193548387,
        "peak_search_state_bytes": 130240,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1823,
      "width": 8,
      "alpha": 0.0,
      "summary": {
        "strict_style_exact": 0.16129032258064516,
        "normalized_style_exact": 0.16129032258064516,
        "first_token_correct": 0.3225806451612903,
        "stopped": 1.0,
        "mean_seconds": 0.004323090322413331,
        "p95_seconds": 0.007787549991917331,
        "max_seconds": 0.013432299994747154,
        "mean_computation_ticks": 103.25806451612904,
        "peak_search_state_bytes": 260480,
        "budget_exhausted": 0
      }
    },
    {
      "order": 1823,
      "width": 8,
      "alpha": 0.6,
      "summary": {
        "strict_style_exact": 0.1935483870967742,
        "normalized_style_exact": 0.1935483870967742,
        "first_token_correct": 0.3225806451612903,
        "stopped": 1.0,
        "mean_seconds": 0.005968864518499393,
        "p95_seconds": 0.012735099997371435,
        "max_seconds": 0.02148019999731332,
        "mean_computation_ticks": 148.93548387096774,
        "peak_search_state_bytes": 260480,
        "budget_exhausted": 0
      }
    }
  ],
  "prior_comparisons": [
    {
      "order": 1801,
      "summary": {
        "greedy": {
          "strict_style_exact": 0.125,
          "normalized_style_exact": 0.1875,
          "first_token_correct": 0.4583333333333333,
          "stopped": 0.9791666666666666,
          "mean_seconds": 0.0013369562481481505,
          "p95_seconds": 0.0023645950030186207,
          "max_seconds": 0.003161600005114451,
          "mean_computation_ticks": 29.875,
          "peak_search_state_bytes": 16280,
          "budget_exhausted": 0
        },
        "selected": {
          "strict_style_exact": 0.1875,
          "normalized_style_exact": 0.25,
          "first_token_correct": 0.4791666666666667,
          "stopped": 1.0,
          "mean_seconds": 0.0036087312500967528,
          "p95_seconds": 0.00941579000136698,
          "max_seconds": 0.012475900002755225,
          "mean_computation_ticks": 81.75,
          "peak_search_state_bytes": 130240,
          "budget_exhausted": 0
        }
      },
      "prior_normalized_gain": 0.0625,
      "dev_normalized_gain": 0.03225806451612903,
      "lost_old_correct_prior": 0
    },
    {
      "order": 1811,
      "summary": {
        "greedy": {
          "strict_style_exact": 0.14583333333333334,
          "normalized_style_exact": 0.20833333333333334,
          "first_token_correct": 0.4583333333333333,
          "stopped": 0.9791666666666666,
          "mean_seconds": 0.001521149999765233,
          "p95_seconds": 0.0026508499970077535,
          "max_seconds": 0.009646899998188019,
          "mean_computation_ticks": 30.5625,
          "peak_search_state_bytes": 16280,
          "budget_exhausted": 0
        },
        "selected": {
          "strict_style_exact": 0.14583333333333334,
          "normalized_style_exact": 0.20833333333333334,
          "first_token_correct": 0.4375,
          "stopped": 1.0,
          "mean_seconds": 0.003634716667572017,
          "p95_seconds": 0.007824744996469232,
          "max_seconds": 0.015630699999746867,
          "mean_computation_ticks": 89.75,
          "peak_search_state_bytes": 130240,
          "budget_exhausted": 0
        }
      },
      "prior_normalized_gain": 0.0,
      "dev_normalized_gain": 0.12903225806451613,
      "lost_old_correct_prior": 0
    },
    {
      "order": 1823,
      "summary": {
        "greedy": {
          "strict_style_exact": 0.14583333333333334,
          "normalized_style_exact": 0.1875,
          "first_token_correct": 0.4166666666666667,
          "stopped": 0.9583333333333334,
          "mean_seconds": 0.0013188104157961789,
          "p95_seconds": 0.0021231150029052516,
          "max_seconds": 0.003315399997518398,
          "mean_computation_ticks": 28.75,
          "peak_search_state_bytes": 16280,
          "budget_exhausted": 0
        },
        "selected": {
          "strict_style_exact": 0.20833333333333334,
          "normalized_style_exact": 0.25,
          "first_token_correct": 0.4375,
          "stopped": 1.0,
          "mean_seconds": 0.003783145833949675,
          "p95_seconds": 0.009925399994972394,
          "max_seconds": 0.01130100000591483,
          "mean_computation_ticks": 83.75,
          "peak_search_state_bytes": 130240,
          "budget_exhausted": 0
        }
      },
      "prior_normalized_gain": 0.0625,
      "dev_normalized_gain": 0.0967741935483871,
      "lost_old_correct_prior": 1
    }
  ],
  "informative_screen": false,
  "summary": {
    "dev_greedy_normalized": 0.12903225806451613,
    "dev_selected_normalized": 0.21505376344086022,
    "mean_prior_normalized_gain": 0.041666666666666664
  },
  "greedy_state_parity": true,
  "weights_unchanged": true,
  "new_training": false,
  "deployed": false,
  "normal_chat_or_vision_proven": false,
  "retention_confirmed": false,
  "resources": {
    "peak_rss_bytes": 54177792,
    "peak_commit_or_vms_bytes": 38879232,
    "os_process_peak_working_set_bytes": 55201792,
    "os_process_peak_pagefile_commit_bytes": 39673856,
    "samples": 81,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 6.983164300007047,
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
    "Diagnosticvirtualclock not compatiblewith live monotonicclock yet.",
    "Search cannot create missing language/visualknowledge or demonstrate new synapselearning.",
    "Sharedsourceinitialization, known8intent styles, alreadyinspectedcases; no independentnormalchat or researchgatepass."
  ]
}
