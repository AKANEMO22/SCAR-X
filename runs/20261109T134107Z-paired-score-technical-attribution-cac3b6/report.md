# Paired-score technical/attribution audit

Gate inherited unchanged; no new candidate training or deployment.

{
  "run": "runs/20261109T134107Z-paired-score-technical-attribution-cac3b6",
  "source_run": "runs/20261109T131841Z-p1-paired-score-v2-0dc401",
  "decision": "GO_P2_RESEARCH_ONLY",
  "selection": {
    "selected_condition": "raw-independent",
    "selected_lr": 0.002,
    "tasks": {
      "order": {
        "probe_gain": 0.18229166666666666,
        "direct": 0.90234375,
        "initial_probe": 0.7213541666666666
      },
      "xor": {
        "probe_gain": 0.25390625,
        "direct": 0.7630208333333334,
        "initial_probe": 0.5494791666666666
      },
      "strict_binding": {
        "probe_gain": 0.010416666666666666,
        "direct": 0.4986979166666667,
        "initial_probe": 0.4752604166666667
      }
    },
    "screen_pass": true,
    "confirmation_opened": true
  },
  "primary_representation_gate": true,
  "source_confirmation_statistics": {
    "order": {
      "probe_gain": {
        "mean": 0.183349609375,
        "lower95": 0.146484375,
        "upper95": 0.225341796875,
        "n_seeds": 8
      },
      "direct": {
        "mean": 0.919921875,
        "lower95": 0.88232421875,
        "upper95": 0.956298828125,
        "n_seeds": 8
      },
      "vs_frozen": {
        "mean": 0.183349609375,
        "lower95": 0.146484375,
        "upper95": 0.225341796875,
        "n_seeds": 8
      },
      "gain_p": 0.0078125,
      "vs_frozen_p": 0.0078125,
      "holm_gain_p": 0.0234375,
      "holm_vs_frozen_p": 0.0234375,
      "pass": true
    },
    "xor": {
      "probe_gain": {
        "mean": 0.199951171875,
        "lower95": 0.150634765625,
        "upper95": 0.246826171875,
        "n_seeds": 8
      },
      "direct": {
        "mean": 0.74560546875,
        "lower95": 0.69775390625,
        "upper95": 0.793701171875,
        "n_seeds": 8
      },
      "vs_frozen": {
        "mean": 0.199951171875,
        "lower95": 0.150634765625,
        "upper95": 0.246826171875,
        "n_seeds": 8
      },
      "gain_p": 0.0078125,
      "vs_frozen_p": 0.0078125,
      "holm_gain_p": 0.0234375,
      "holm_vs_frozen_p": 0.0234375,
      "pass": true
    },
    "strict_binding": {
      "probe_gain": {
        "mean": -0.012939453125,
        "lower95": -0.023681640625,
        "upper95": -0.0009765625,
        "n_seeds": 8
      },
      "direct": {
        "mean": 0.49560546875,
        "lower95": 0.484375,
        "upper95": 0.50732421875,
        "n_seeds": 8
      },
      "vs_frozen": {
        "mean": -0.012939453125,
        "lower95": -0.023681640625,
        "upper95": -0.0009765625,
        "n_seeds": 8
      },
      "gain_p": 0.09375,
      "vs_frozen_p": 0.09375,
      "holm_gain_p": 0.09375,
      "holm_vs_frozen_p": 0.09375,
      "pass": false
    }
  },
  "control_summary": {
    "order": {
      "clipped-adam-independent": {
        "direct": 0.910888671875,
        "probe_gain": 0.198486328125
      },
      "clipped-sgd-independent": {
        "direct": 0.820068359375,
        "probe_gain": 0.2392578125
      },
      "frozen-independent": {
        "direct": 0.815673828125,
        "probe_gain": 0.0
      },
      "raw-independent": {
        "direct": 0.919921875,
        "probe_gain": 0.183349609375
      },
      "shuffle-independent": {
        "direct": 0.753173828125,
        "probe_gain": 0.097412109375
      }
    },
    "xor": {
      "clipped-adam-independent": {
        "direct": 0.671875,
        "probe_gain": 0.133544921875
      },
      "clipped-sgd-independent": {
        "direct": 0.66552734375,
        "probe_gain": 0.171142578125
      },
      "frozen-independent": {
        "direct": 0.56396484375,
        "probe_gain": 0.0
      },
      "raw-independent": {
        "direct": 0.74560546875,
        "probe_gain": 0.199951171875
      },
      "shuffle-independent": {
        "direct": 0.613037109375,
        "probe_gain": 0.087646484375
      }
    },
    "strict_binding": {
      "clipped-adam-independent": {
        "direct": 0.5009765625,
        "probe_gain": -0.019287109375
      },
      "clipped-sgd-independent": {
        "direct": 0.500732421875,
        "probe_gain": 0.00146484375
      },
      "frozen-independent": {
        "direct": 0.510986328125,
        "probe_gain": 0.0
      },
      "raw-independent": {
        "direct": 0.49560546875,
        "probe_gain": -0.012939453125
      },
      "shuffle-independent": {
        "direct": 0.50390625,
        "probe_gain": -0.010009765625
      }
    }
  },
  "secondary_contrasts": {
    "order": {
      "frozen-independent/probe": {
        "paired_values": [
          0.166015625,
          0.169921875,
          0.23828125,
          0.107421875,
          0.126953125,
          0.296875,
          0.16015625,
          0.201171875
        ],
        "bootstrap95": {
          "mean": 0.183349609375,
          "lower95": 0.146484375,
          "upper95": 0.225341796875,
          "n_seeds": 8
        },
        "signflip_p": 0.0078125,
        "secondary_exploratory": false
      },
      "frozen-independent/direct": {
        "paired_values": [
          0.087890625,
          0.1171875,
          0.029296875,
          0.078125,
          0.103515625,
          0.169921875,
          0.140625,
          0.107421875
        ],
        "bootstrap95": {
          "mean": 0.104248046875,
          "lower95": 0.076171875,
          "upper95": 0.1313537597656249,
          "n_seeds": 8
        },
        "signflip_p": 0.0078125,
        "secondary_exploratory": false
      },
      "shuffle-independent/probe": {
        "paired_values": [
          0.130859375,
          0.205078125,
          0.126953125,
          -0.056640625,
          0.037109375,
          0.0625,
          0.10546875,
          0.076171875
        ],
        "bootstrap95": {
          "mean": 0.0859375,
          "lower95": 0.033447265625,
          "upper95": 0.1337890625,
          "n_seeds": 8
        },
        "signflip_p": 0.0234375,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.046875
      },
      "shuffle-independent/direct": {
        "paired_values": [
          0.19921875,
          0.2578125,
          0.0703125,
          0.146484375,
          0.099609375,
          0.283203125,
          0.119140625,
          0.158203125
        ],
        "bootstrap95": {
          "mean": 0.166748046875,
          "lower95": 0.1201171875,
          "upper95": 0.2168029785156249,
          "n_seeds": 8
        },
        "signflip_p": 0.0078125,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.0234375
      },
      "clipped-sgd-independent/probe": {
        "paired_values": [
          -0.041015625,
          -0.041015625,
          -0.083984375,
          -0.123046875,
          -0.0859375,
          0.009765625,
          -0.015625,
          -0.06640625
        ],
        "bootstrap95": {
          "mean": -0.055908203125,
          "lower95": -0.083740234375,
          "upper95": -0.0283203125,
          "n_seeds": 8
        },
        "signflip_p": 0.015625,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.046875
      },
      "clipped-sgd-independent/direct": {
        "paired_values": [
          -0.029296875,
          0.0078125,
          0.30859375,
          -0.0234375,
          0.00390625,
          0.37109375,
          -0.001953125,
          0.162109375
        ],
        "bootstrap95": {
          "mean": 0.099853515625,
          "lower95": 0.007080078125,
          "upper95": 0.21192626953124982,
          "n_seeds": 8
        },
        "signflip_p": 0.203125,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.40625
      },
      "clipped-adam-independent/probe": {
        "paired_values": [
          -0.00390625,
          0.02734375,
          -0.01171875,
          -0.080078125,
          -0.0546875,
          0.013671875,
          0.001953125,
          -0.013671875
        ],
        "bootstrap95": {
          "mean": -0.01513671875,
          "lower95": -0.039794921875,
          "upper95": 0.006103515625,
          "n_seeds": 8
        },
        "signflip_p": 0.328125,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.65625
      },
      "clipped-adam-independent/direct": {
        "paired_values": [
          -0.01171875,
          0.005859375,
          0.015625,
          0.001953125,
          -0.0078125,
          0.068359375,
          0.0078125,
          -0.0078125
        ],
        "bootstrap95": {
          "mean": 0.009033203125,
          "lower95": -0.004150390625,
          "upper95": 0.028076171875,
          "n_seeds": 8
        },
        "signflip_p": 0.46875,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.71875
      }
    },
    "xor": {
      "frozen-independent/probe": {
        "paired_values": [
          0.150390625,
          0.240234375,
          0.2265625,
          0.16015625,
          0.072265625,
          0.3046875,
          0.177734375,
          0.267578125
        ],
        "bootstrap95": {
          "mean": 0.199951171875,
          "lower95": 0.150634765625,
          "upper95": 0.246826171875,
          "n_seeds": 8
        },
        "signflip_p": 0.0078125,
        "secondary_exploratory": false
      },
      "frozen-independent/direct": {
        "paired_values": [
          0.150390625,
          0.15625,
          0.2578125,
          0.162109375,
          0.068359375,
          0.337890625,
          0.1640625,
          0.15625
        ],
        "bootstrap95": {
          "mean": 0.181640625,
          "lower95": 0.134521484375,
          "upper95": 0.238525390625,
          "n_seeds": 8
        },
        "signflip_p": 0.0078125,
        "secondary_exploratory": false
      },
      "shuffle-independent/probe": {
        "paired_values": [
          0.033203125,
          0.177734375,
          0.197265625,
          0.162109375,
          0.005859375,
          0.158203125,
          0.015625,
          0.1484375
        ],
        "bootstrap95": {
          "mean": 0.1123046875,
          "lower95": 0.056884765625,
          "upper95": 0.160400390625,
          "n_seeds": 8
        },
        "signflip_p": 0.0078125,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.0234375
      },
      "shuffle-independent/direct": {
        "paired_values": [
          0.115234375,
          0.189453125,
          0.2578125,
          0.119140625,
          -0.044921875,
          0.275390625,
          0.015625,
          0.1328125
        ],
        "bootstrap95": {
          "mean": 0.132568359375,
          "lower95": 0.06005859375,
          "upper95": 0.202392578125,
          "n_seeds": 8
        },
        "signflip_p": 0.0234375,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.046875
      },
      "clipped-sgd-independent/probe": {
        "paired_values": [
          -0.021484375,
          0.142578125,
          0.021484375,
          0.04296875,
          -0.11328125,
          0.119140625,
          -0.033203125,
          0.072265625
        ],
        "bootstrap95": {
          "mean": 0.02880859375,
          "lower95": -0.027587890625,
          "upper95": 0.08203125,
          "n_seeds": 8
        },
        "signflip_p": 0.359375,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.71875
      },
      "clipped-sgd-independent/direct": {
        "paired_values": [
          0.041015625,
          0.18359375,
          0.130859375,
          0.076171875,
          -0.08984375,
          0.228515625,
          0.0078125,
          0.0625
        ],
        "bootstrap95": {
          "mean": 0.080078125,
          "lower95": 0.012939453125,
          "upper95": 0.14453125,
          "n_seeds": 8
        },
        "signflip_p": 0.0703125,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.2109375
      },
      "clipped-adam-independent/probe": {
        "paired_values": [
          -0.07421875,
          0.16015625,
          0.1015625,
          0.0,
          0.046875,
          0.1015625,
          0.07421875,
          0.12109375
        ],
        "bootstrap95": {
          "mean": 0.06640625,
          "lower95": 0.01513671875,
          "upper95": 0.11083984375,
          "n_seeds": 8
        },
        "signflip_p": 0.0625,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.1875
      },
      "clipped-adam-independent/direct": {
        "paired_values": [
          0.00390625,
          0.109375,
          0.091796875,
          -0.04296875,
          0.056640625,
          0.15234375,
          0.09375,
          0.125
        ],
        "bootstrap95": {
          "mean": 0.07373046875,
          "lower95": 0.02880859375,
          "upper95": 0.11328735351562491,
          "n_seeds": 8
        },
        "signflip_p": 0.0234375,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.0703125
      }
    },
    "strict_binding": {
      "frozen-independent/probe": {
        "paired_values": [
          -0.025390625,
          -0.01953125,
          -0.01171875,
          0.001953125,
          -0.02734375,
          0.01953125,
          -0.0078125,
          -0.033203125
        ],
        "bootstrap95": {
          "mean": -0.012939453125,
          "lower95": -0.023681640625,
          "upper95": -0.0009765625,
          "n_seeds": 8
        },
        "signflip_p": 0.09375,
        "secondary_exploratory": false
      },
      "frozen-independent/direct": {
        "paired_values": [
          -0.005859375,
          -0.03515625,
          0.025390625,
          -0.0390625,
          0.0,
          -0.044921875,
          -0.013671875,
          -0.009765625
        ],
        "bootstrap95": {
          "mean": -0.015380859375,
          "lower95": -0.0302734375,
          "upper95": 0.000244140625,
          "n_seeds": 8
        },
        "signflip_p": 0.125,
        "secondary_exploratory": false
      },
      "shuffle-independent/probe": {
        "paired_values": [
          -0.025390625,
          -0.005859375,
          0.015625,
          0.01953125,
          0.001953125,
          0.025390625,
          -0.009765625,
          -0.044921875
        ],
        "bootstrap95": {
          "mean": -0.0029296875,
          "lower95": -0.01953125,
          "upper95": 0.011962890625,
          "n_seeds": 8
        },
        "signflip_p": 0.78125,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.78125
      },
      "shuffle-independent/direct": {
        "paired_values": [
          -0.0078125,
          0.00390625,
          -0.01953125,
          0.0,
          -0.033203125,
          0.0,
          -0.0234375,
          0.013671875
        ],
        "bootstrap95": {
          "mean": -0.00830078125,
          "lower95": -0.018798828125,
          "upper95": 0.0017150878906249112,
          "n_seeds": 8
        },
        "signflip_p": 0.1875,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.1875
      },
      "clipped-sgd-independent/probe": {
        "paired_values": [
          -0.048828125,
          -0.064453125,
          0.080078125,
          -0.005859375,
          0.0078125,
          -0.00390625,
          -0.029296875,
          -0.05078125
        ],
        "bootstrap95": {
          "mean": -0.014404296875,
          "lower95": -0.041748046875,
          "upper95": 0.01831665039062491,
          "n_seeds": 8
        },
        "signflip_p": 0.4140625,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.71875
      },
      "clipped-sgd-independent/direct": {
        "paired_values": [
          -0.009765625,
          0.001953125,
          0.025390625,
          -0.00390625,
          -0.02734375,
          0.0,
          -0.0390625,
          0.01171875
        ],
        "bootstrap95": {
          "mean": -0.005126953125,
          "lower95": -0.018798828125,
          "upper95": 0.0078125,
          "n_seeds": 8
        },
        "signflip_p": 0.515625,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.515625
      },
      "clipped-adam-independent/probe": {
        "paired_values": [
          -0.02734375,
          0.025390625,
          0.01953125,
          -0.013671875,
          0.037109375,
          -0.01953125,
          0.02734375,
          0.001953125
        ],
        "bootstrap95": {
          "mean": 0.00634765625,
          "lower95": -0.009521484375,
          "upper95": 0.022216796875,
          "n_seeds": 8
        },
        "signflip_p": 0.484375,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.65625
      },
      "clipped-adam-independent/direct": {
        "paired_values": [
          -0.005859375,
          0.001953125,
          0.017578125,
          -0.009765625,
          -0.009765625,
          0.0,
          -0.029296875,
          -0.0078125
        ],
        "bootstrap95": {
          "mean": -0.00537109375,
          "lower95": -0.013916015625,
          "upper95": 0.003173828125,
          "n_seeds": 8
        },
        "signflip_p": 0.359375,
        "secondary_exploratory": true,
        "holm_3tasks_p": 0.71875
      }
    }
  },
  "cold_checks": [
    {
      "task": "order",
      "seed": 14503,
      "exact": true,
      "checkpoint_sha256": "f15a06a1965be68382e6eea86c2fc88489a270a7eed2e52076141be1a6db2340"
    },
    {
      "task": "order",
      "seed": 14519,
      "exact": true,
      "checkpoint_sha256": "de526a6e6e5f8d3b9563137a4c39ec333c36eb922757b9eddb81c5b5f262d6bd"
    },
    {
      "task": "order",
      "seed": 14533,
      "exact": true,
      "checkpoint_sha256": "0a9ca4471c8970312af9d99c2f9d20bc290dfd0806a508387efc373bc745db70"
    },
    {
      "task": "order",
      "seed": 14537,
      "exact": true,
      "checkpoint_sha256": "e6cd594aa1f83f78306dad1980781360fa1a907fe5705548c29361a7cbf5dc5c"
    },
    {
      "task": "order",
      "seed": 14543,
      "exact": true,
      "checkpoint_sha256": "7bf2fb975c929220c71d210b3aea47bb7efc812f346b5574ce3e9d2b3b84bcb9"
    },
    {
      "task": "order",
      "seed": 14549,
      "exact": true,
      "checkpoint_sha256": "2f1f479c0eaca8f0849eea26a02b15327711be81d99bd54ae4d60b06b70d00ad"
    },
    {
      "task": "order",
      "seed": 14551,
      "exact": true,
      "checkpoint_sha256": "4dd0968c79021e13120efe8bdd7526602d1f31c4f84b879377db51dfa3b93419"
    },
    {
      "task": "order",
      "seed": 14557,
      "exact": true,
      "checkpoint_sha256": "61c7b379515e63c6a9f2a0d2b193ed99c091134716a30cc3dc404977f27db5d1"
    },
    {
      "task": "xor",
      "seed": 14503,
      "exact": true,
      "checkpoint_sha256": "fab3e43bf2f05e1580c3fc769ff8b357b3037daa3125c3e4975f0b72692178cb"
    },
    {
      "task": "xor",
      "seed": 14519,
      "exact": true,
      "checkpoint_sha256": "d5d3696b4b3181f59e5261ff9ee161797928e00b836a3bd5ee6d283594a1cb1b"
    },
    {
      "task": "xor",
      "seed": 14533,
      "exact": true,
      "checkpoint_sha256": "adf95d8652c1e46d5f88747ff4895cd4263fb80e59db458cba85c15ccfd8f629"
    },
    {
      "task": "xor",
      "seed": 14537,
      "exact": true,
      "checkpoint_sha256": "3184e8323c54049d133e78ce66c81e3667799cddd729e9d888ec0fb1d9de69c5"
    },
    {
      "task": "xor",
      "seed": 14543,
      "exact": true,
      "checkpoint_sha256": "d509a314bc9c94ccb2829e66c115635d1ac3d06398ae8d61e9b7f1332f03cf97"
    },
    {
      "task": "xor",
      "seed": 14549,
      "exact": true,
      "checkpoint_sha256": "9948a003f66b9a8f3c0983e159aad2fb1d4e60944bf05ab56d4c05b53105a27d"
    },
    {
      "task": "xor",
      "seed": 14551,
      "exact": true,
      "checkpoint_sha256": "0a57c433258bf51282d2a026e21aac00c782380747e3a8124501c48c4f068cf3"
    },
    {
      "task": "xor",
      "seed": 14557,
      "exact": true,
      "checkpoint_sha256": "616d78da909c540c8e8ee58679b57e8d8c7dea2af0b9657ea6a6eb75cbdbe09f"
    },
    {
      "task": "strict_binding",
      "seed": 14503,
      "exact": true,
      "checkpoint_sha256": "94c783bb9d8e2927f8597dcb0455b1b806418c0bf57bd9544bb7ea06eef6c0cc"
    },
    {
      "task": "strict_binding",
      "seed": 14519,
      "exact": true,
      "checkpoint_sha256": "31c8f2d085501333a7ca20944da3dd82d1ada86a3a336094fd47294b0619b40d"
    },
    {
      "task": "strict_binding",
      "seed": 14533,
      "exact": true,
      "checkpoint_sha256": "a6e6331c37d6225b1ea03200f477e6d467d2b2db789e7fb4e2f02417847c8bd2"
    },
    {
      "task": "strict_binding",
      "seed": 14537,
      "exact": true,
      "checkpoint_sha256": "f1e30f404b2e36771b86ff835295c53fc018c4daa38ebb1aabfb5a819f859c21"
    },
    {
      "task": "strict_binding",
      "seed": 14543,
      "exact": true,
      "checkpoint_sha256": "35eb518e410d381a14ea337765f3a936598ba10dc7cbcb74b3f2ad67ec0f5c04"
    },
    {
      "task": "strict_binding",
      "seed": 14549,
      "exact": true,
      "checkpoint_sha256": "48902f2674f56d9a6e5b7f30b3ff4dc795e1a336a6c90df0885ec1428f4f74b0"
    },
    {
      "task": "strict_binding",
      "seed": 14551,
      "exact": true,
      "checkpoint_sha256": "1fae0e374f1af9f363fa3c1d4d6a8412714f9661a7b7ea3b4a36f288fb0ad427"
    },
    {
      "task": "strict_binding",
      "seed": 14557,
      "exact": true,
      "checkpoint_sha256": "475a8d3006babcc3629707fb697bbdcaee9863f37fca949bd93017dfeb9b02d0"
    }
  ],
  "all_cold_exact": true,
  "head_diagnostics": [
    {
      "task": "order",
      "seed": 14503,
      "normalized_feature_clipped_fraction": 0.0003255208333333333,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 9.704153060913086,
      "readout_gap_to_refit": 0.015625,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "order",
      "seed": 14519,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 15.697465896606445,
      "readout_gap_to_refit": -0.044921875,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "order",
      "seed": 14533,
      "normalized_feature_clipped_fraction": 0.0032552083333333335,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 11.70408821105957,
      "readout_gap_to_refit": 0.083984375,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "order",
      "seed": 14537,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 12.447514533996582,
      "readout_gap_to_refit": -0.05078125,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "order",
      "seed": 14543,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 14.699723243713379,
      "readout_gap_to_refit": -0.037109375,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "order",
      "seed": 14549,
      "normalized_feature_clipped_fraction": 0.0003255208333333333,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 14.197476387023926,
      "readout_gap_to_refit": 0.0234375,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "order",
      "seed": 14551,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 12.609031677246094,
      "readout_gap_to_refit": -0.021484375,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "order",
      "seed": 14557,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 8.913569450378418,
      "readout_gap_to_refit": 0.037109375,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "xor",
      "seed": 14503,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 3.405054807662964,
      "readout_gap_to_refit": -0.021484375,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "xor",
      "seed": 14519,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 4.221127986907959,
      "readout_gap_to_refit": 0.04296875,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "xor",
      "seed": 14533,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 4.450220584869385,
      "readout_gap_to_refit": -0.0390625,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "xor",
      "seed": 14537,
      "normalized_feature_clipped_fraction": 0.0013020833333333333,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 3.4605767726898193,
      "readout_gap_to_refit": 0.01171875,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "xor",
      "seed": 14543,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 4.130136013031006,
      "readout_gap_to_refit": -0.017578125,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "xor",
      "seed": 14549,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 3.5387465953826904,
      "readout_gap_to_refit": -0.041015625,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "xor",
      "seed": 14551,
      "normalized_feature_clipped_fraction": 0.010416666666666666,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 4.126240253448486,
      "readout_gap_to_refit": 0.015625,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "xor",
      "seed": 14557,
      "normalized_feature_clipped_fraction": 0.0003255208333333333,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 3.643446922302246,
      "readout_gap_to_refit": -0.001953125,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "strict_binding",
      "seed": 14503,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 4.6368489265441895,
      "readout_gap_to_refit": 0.0,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "strict_binding",
      "seed": 14519,
      "normalized_feature_clipped_fraction": 0.020182291666666668,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 3.7939889430999756,
      "readout_gap_to_refit": 0.005859375,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "strict_binding",
      "seed": 14533,
      "normalized_feature_clipped_fraction": 0.0035807291666666665,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 6.77462100982666,
      "readout_gap_to_refit": 0.001953125,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "strict_binding",
      "seed": 14537,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 6.460618495941162,
      "readout_gap_to_refit": 0.021484375,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "strict_binding",
      "seed": 14543,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 4.476771831512451,
      "readout_gap_to_refit": 0.046875,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "strict_binding",
      "seed": 14549,
      "normalized_feature_clipped_fraction": 0.0003255208333333333,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 6.492486000061035,
      "readout_gap_to_refit": 0.01953125,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "strict_binding",
      "seed": 14551,
      "normalized_feature_clipped_fraction": 0.0,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 3.734175443649292,
      "readout_gap_to_refit": 0.04296875,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    },
    {
      "task": "strict_binding",
      "seed": 14557,
      "normalized_feature_clipped_fraction": 0.0016276041666666667,
      "near_constant_neuron_fraction": 0.0,
      "covariance_effective_rank": 5.376439094543457,
      "readout_gap_to_refit": -0.02734375,
      "interpretation": "posthoc heldout feature statistics; no retraining or causal retention claim"
    }
  ],
  "deployed": false,
  "normal_chat_or_vision_proven": false,
  "retention_proven": false,
  "limits": [
    "Gate/selection inherited unchanged; secondary contrasts and head statistics are exploratory.",
    "Existing bit-tasks with heldout noise/timing do not prove broad novel-combination or language generalization.",
    "No claim paired sampling/Adam uniquely explains gains vs compute-matched clipped baselines."
  ]
}
