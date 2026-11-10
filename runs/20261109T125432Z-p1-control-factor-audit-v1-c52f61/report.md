# P1 control factors

Isolated diagnostic only; stopped local family stays stopped.

{
  "run": "runs/20261109T125432Z-p1-control-factor-audit-v1-c52f61",
  "decision": "DIAGNOSTIC_ONLY",
  "primary_representation_gate": false,
  "control_summary": {
    "order": {
      "frozen": {
        "direct": 0.9518229166666666,
        "probe_gain": 0.0
      },
      "adam-all": {
        "direct": 1.0,
        "probe_gain": 0.05078125
      },
      "adam-rotating": {
        "direct": 1.0,
        "probe_gain": 0.049479166666666664
      },
      "sgd-all": {
        "direct": 0.9778645833333334,
        "probe_gain": 0.03515625
      },
      "sgd-rotating": {
        "direct": 0.9661458333333334,
        "probe_gain": 0.020833333333333332
      }
    },
    "xor": {
      "frozen": {
        "direct": 0.8776041666666666,
        "probe_gain": 0.0
      },
      "adam-all": {
        "direct": 1.0,
        "probe_gain": 0.18359375
      },
      "adam-rotating": {
        "direct": 0.9986979166666666,
        "probe_gain": 0.17447916666666666
      },
      "sgd-all": {
        "direct": 0.9375,
        "probe_gain": 0.08854166666666667
      },
      "sgd-rotating": {
        "direct": 0.89453125,
        "probe_gain": 0.03125
      }
    },
    "strict_binding": {
      "frozen": {
        "direct": 0.6510416666666666,
        "probe_gain": 0.0
      },
      "adam-all": {
        "direct": 0.82421875,
        "probe_gain": 0.22005208333333334
      },
      "adam-rotating": {
        "direct": 0.6328125,
        "probe_gain": 0.024739583333333332
      },
      "sgd-all": {
        "direct": 0.6106770833333334,
        "probe_gain": 0.0078125
      },
      "sgd-rotating": {
        "direct": 0.6276041666666666,
        "probe_gain": 0.028645833333333332
      }
    }
  },
  "comparisons": {
    "order": {
      "adam-all minus sgd-all": {
        "direct": {
          "values": [
            0.0546875,
            0.0078125,
            0.00390625
          ],
          "bootstrap95": {
            "mean": 0.022135416666666668,
            "lower95": 0.00390625,
            "upper95": 0.0546875,
            "n_seeds": 3
          },
          "signflip_p": 0.25,
          "operationally_material_mean_5pp": false,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            0.01171875,
            0.03125,
            0.00390625
          ],
          "bootstrap95": {
            "mean": 0.015625,
            "lower95": 0.00390625,
            "upper95": 0.03125,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "mean_loss": {
          "values": [
            -0.19039912749998228,
            -0.06054660272249636,
            -0.02944565831611673
          ],
          "bootstrap95": {
            "mean": -0.09346379617953178,
            "lower95": -0.19039912749998225,
            "upper95": -0.02944565831611673,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "train_seconds": {
          "values": [
            1.1683808999950998,
            -0.24409100000048056,
            0.007198400002380367
          ],
          "bootstrap95": {
            "mean": 0.31049609999899985,
            "lower95": -0.24409100000048056,
            "upper95": 1.1683808999950998,
            "n_seeds": 3
          },
          "signflip_p": 0.75
        },
        "hidden_delta_in": {
          "values": [
            1.3493325263261795,
            1.5112853646278381,
            1.0324829295277596
          ],
          "bootstrap95": {
            "mean": 1.2977002734939258,
            "lower95": 1.0324829295277596,
            "upper95": 1.5112853646278381,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_rec": {
          "values": [
            3.0576978474855423,
            2.853886052966118,
            2.738566592335701
          ],
          "bootstrap95": {
            "mean": 2.883383497595787,
            "lower95": 2.738566592335701,
            "upper95": 3.0576978474855423,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      },
      "adam-rotating minus sgd-rotating": {
        "direct": {
          "values": [
            0.0546875,
            0.0390625,
            0.0078125
          ],
          "bootstrap95": {
            "mean": 0.033854166666666664,
            "lower95": 0.0078125,
            "upper95": 0.0546875,
            "n_seeds": 3
          },
          "signflip_p": 0.25,
          "operationally_material_mean_5pp": false,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            0.01171875,
            0.0625,
            0.01171875
          ],
          "bootstrap95": {
            "mean": 0.028645833333333332,
            "lower95": 0.01171875,
            "upper95": 0.0625,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "mean_loss": {
          "values": [
            -0.1629753496965783,
            -0.0753511404621814,
            -0.024558956123668707
          ],
          "bootstrap95": {
            "mean": -0.0876284820941428,
            "lower95": -0.1629753496965783,
            "upper95": -0.024558956123668707,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "train_seconds": {
          "values": [
            1.0491957999911392,
            0.7927998000013758,
            0.5727953000096022
          ],
          "bootstrap95": {
            "mean": 0.8049303000007058,
            "lower95": 0.5727953000096022,
            "upper95": 1.0491957999911392,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_in": {
          "values": [
            1.2957140449434519,
            1.296576227992773,
            1.1313113756477833
          ],
          "bootstrap95": {
            "mean": 1.2412005495280027,
            "lower95": 1.1313113756477833,
            "upper95": 1.296576227992773,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_rec": {
          "values": [
            2.1342648677527905,
            1.807097926735878,
            2.138817060738802
          ],
          "bootstrap95": {
            "mean": 2.026726618409157,
            "lower95": 1.807097926735878,
            "upper95": 2.138817060738802,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      },
      "adam-all minus adam-rotating": {
        "direct": {
          "values": [
            0.0,
            0.0,
            0.0
          ],
          "bootstrap95": {
            "mean": 0.0,
            "lower95": 0.0,
            "upper95": 0.0,
            "n_seeds": 3
          },
          "signflip_p": 1.0,
          "operationally_material_mean_5pp": false,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            0.0,
            0.0,
            0.00390625
          ],
          "bootstrap95": {
            "mean": 0.0013020833333333333,
            "lower95": 0.0,
            "upper95": 0.00390625,
            "n_seeds": 3
          },
          "signflip_p": 1.0
        },
        "mean_loss": {
          "values": [
            -0.034278715415030685,
            -0.011403446037622236,
            -0.011140950174750502
          ],
          "bootstrap95": {
            "mean": -0.018941037209134474,
            "lower95": -0.034278715415030685,
            "upper95": -0.011140950174750502,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "train_seconds": {
          "values": [
            0.014807999999902677,
            -1.0463440000021365,
            -0.5837635999996564
          ],
          "bootstrap95": {
            "mean": -0.5384332000006301,
            "lower95": -1.0463440000021365,
            "upper95": 0.014807999999902677,
            "n_seeds": 3
          },
          "signflip_p": 0.5
        },
        "hidden_delta_in": {
          "values": [
            0.12331056594848633,
            0.2853635549545288,
            -0.0327763557434082
          ],
          "bootstrap95": {
            "mean": 0.1252992550532023,
            "lower95": -0.0327763557434082,
            "upper95": 0.2853635549545288,
            "n_seeds": 3
          },
          "signflip_p": 0.5
        },
        "hidden_delta_rec": {
          "values": [
            1.0244169235229492,
            1.1811057329177856,
            0.6849510669708252
          ],
          "bootstrap95": {
            "mean": 0.9634912411371866,
            "lower95": 0.6849510669708252,
            "upper95": 1.1811057329177856,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      },
      "sgd-all minus sgd-rotating": {
        "direct": {
          "values": [
            0.0,
            0.03125,
            0.00390625
          ],
          "bootstrap95": {
            "mean": 0.01171875,
            "lower95": 0.0,
            "upper95": 0.03125,
            "n_seeds": 3
          },
          "signflip_p": 0.5,
          "operationally_material_mean_5pp": false,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            0.0,
            0.03125,
            0.01171875
          ],
          "bootstrap95": {
            "mean": 0.014322916666666666,
            "lower95": 0.0,
            "upper95": 0.03125,
            "n_seeds": 3
          },
          "signflip_p": 0.5
        },
        "mean_loss": {
          "values": [
            -0.006854937611626699,
            -0.026207983777307278,
            -0.00625424798230248
          ],
          "bootstrap95": {
            "mean": -0.013105723123745486,
            "lower95": -0.026207983777307278,
            "upper95": -0.00625424798230248,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "train_seconds": {
          "values": [
            -0.10437710000405787,
            -0.009453200000280049,
            -0.018166699992434587
          ],
          "bootstrap95": {
            "mean": -0.04399899999892417,
            "lower95": -0.10437710000405787,
            "upper95": -0.009453200000280049,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_in": {
          "values": [
            0.0696920845657587,
            0.07065441831946373,
            0.06605209037661552
          ],
          "bootstrap95": {
            "mean": 0.06879953108727932,
            "lower95": 0.06605209037661552,
            "upper95": 0.07065441831946373,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_rec": {
          "values": [
            0.10098394379019737,
            0.13431760668754578,
            0.08520153537392616
          ],
          "bootstrap95": {
            "mean": 0.10683436195055644,
            "lower95": 0.08520153537392616,
            "upper95": 0.13431760668754578,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      }
    },
    "xor": {
      "adam-all minus sgd-all": {
        "direct": {
          "values": [
            0.04296875,
            0.05859375,
            0.0859375
          ],
          "bootstrap95": {
            "mean": 0.0625,
            "lower95": 0.04296875,
            "upper95": 0.0859375,
            "n_seeds": 3
          },
          "signflip_p": 0.25,
          "operationally_material_mean_5pp": true,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            0.09375,
            0.0625,
            0.12890625
          ],
          "bootstrap95": {
            "mean": 0.09505208333333333,
            "lower95": 0.0625,
            "upper95": 0.12890625,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "mean_loss": {
          "values": [
            -0.19974741351378705,
            -0.1525146215879907,
            -0.25510817285052634
          ],
          "bootstrap95": {
            "mean": -0.20245673598410138,
            "lower95": -0.25510817285052634,
            "upper95": -0.1525146215879907,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "train_seconds": {
          "values": [
            0.6487579999957234,
            0.35438170000270475,
            0.1616113000054611
          ],
          "bootstrap95": {
            "mean": 0.3882503333346297,
            "lower95": 0.1616113000054611,
            "upper95": 0.6487579999957234,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_in": {
          "values": [
            1.6276209354400635,
            1.7740359753370285,
            1.8064657747745514
          ],
          "bootstrap95": {
            "mean": 1.736040895183881,
            "lower95": 1.6276209354400635,
            "upper95": 1.8064657747745514,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_rec": {
          "values": [
            2.8092264533042908,
            2.7413662374019623,
            2.9642707109451294
          ],
          "bootstrap95": {
            "mean": 2.838287800550461,
            "lower95": 2.7413662374019623,
            "upper95": 2.9642707109451294,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      },
      "adam-rotating minus sgd-rotating": {
        "direct": {
          "values": [
            0.09765625,
            0.08203125,
            0.1328125
          ],
          "bootstrap95": {
            "mean": 0.10416666666666667,
            "lower95": 0.08203125,
            "upper95": 0.1328125,
            "n_seeds": 3
          },
          "signflip_p": 0.25,
          "operationally_material_mean_5pp": true,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            0.15625,
            0.0703125,
            0.203125
          ],
          "bootstrap95": {
            "mean": 0.14322916666666666,
            "lower95": 0.0703125,
            "upper95": 0.203125,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "mean_loss": {
          "values": [
            -0.24340328991072724,
            -0.17106578266946457,
            -0.25533635913046915
          ],
          "bootstrap95": {
            "mean": -0.223268477236887,
            "lower95": -0.25533635913046915,
            "upper95": -0.17106578266946457,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "train_seconds": {
          "values": [
            0.7699478000067757,
            0.6208913999944343,
            -0.07903950000036275
          ],
          "bootstrap95": {
            "mean": 0.4372665666669491,
            "lower95": -0.07903950000036275,
            "upper95": 0.7699478000067757,
            "n_seeds": 3
          },
          "signflip_p": 0.5
        },
        "hidden_delta_in": {
          "values": [
            1.4931955747306347,
            1.501266971230507,
            1.4501976296305656
          ],
          "bootstrap95": {
            "mean": 1.4815533918639023,
            "lower95": 1.4501976296305656,
            "upper95": 1.501266971230507,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_rec": {
          "values": [
            1.989266812801361,
            2.0895542055368423,
            1.867856465280056
          ],
          "bootstrap95": {
            "mean": 1.9822258278727531,
            "lower95": 1.867856465280056,
            "upper95": 2.0895542055368423,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      },
      "adam-all minus adam-rotating": {
        "direct": {
          "values": [
            0.0,
            0.00390625,
            0.0
          ],
          "bootstrap95": {
            "mean": 0.0013020833333333333,
            "lower95": 0.0,
            "upper95": 0.00390625,
            "n_seeds": 3
          },
          "signflip_p": 1.0,
          "operationally_material_mean_5pp": false,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            0.00390625,
            0.01171875,
            0.01171875
          ],
          "bootstrap95": {
            "mean": 0.009114583333333334,
            "lower95": 0.00390625,
            "upper95": 0.01171875,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "mean_loss": {
          "values": [
            -0.044144703854960665,
            -0.030063001454830034,
            -0.047709886748705896
          ],
          "bootstrap95": {
            "mean": -0.0406391973528322,
            "lower95": -0.04770988674870589,
            "upper95": -0.03006300145483003,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "train_seconds": {
          "values": [
            -0.15956460000597872,
            -0.6543485000001965,
            -0.2570612999916193
          ],
          "bootstrap95": {
            "mean": -0.3569914666659315,
            "lower95": -0.6543485000001965,
            "upper95": -0.15956460000597872,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_in": {
          "values": [
            0.2665376663208008,
            0.3777233362197876,
            0.49274349212646484
          ],
          "bootstrap95": {
            "mean": 0.3790014982223511,
            "lower95": 0.2665376663208008,
            "upper95": 0.49274349212646484,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_rec": {
          "values": [
            0.9784736633300781,
            0.8177101612091064,
            1.2475323677062988
          ],
          "bootstrap95": {
            "mean": 1.0145720640818279,
            "lower95": 0.8177101612091064,
            "upper95": 1.2475323677062988,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      },
      "sgd-all minus sgd-rotating": {
        "direct": {
          "values": [
            0.0546875,
            0.02734375,
            0.046875
          ],
          "bootstrap95": {
            "mean": 0.04296875,
            "lower95": 0.02734375,
            "upper95": 0.0546875,
            "n_seeds": 3
          },
          "signflip_p": 0.25,
          "operationally_material_mean_5pp": false,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            0.06640625,
            0.01953125,
            0.0859375
          ],
          "bootstrap95": {
            "mean": 0.057291666666666664,
            "lower95": 0.01953125,
            "upper95": 0.0859375,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "mean_loss": {
          "values": [
            -0.08780058025190085,
            -0.048614162536303906,
            -0.047938073028648664
          ],
          "bootstrap95": {
            "mean": -0.06145093860561781,
            "lower95": -0.08780058025190085,
            "upper95": -0.047938073028648664,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "train_seconds": {
          "values": [
            -0.038374799994926434,
            -0.38783880000846693,
            -0.49771209999744315
          ],
          "bootstrap95": {
            "mean": -0.30797523333361215,
            "lower95": -0.49771209999744315,
            "upper95": -0.038374799994926434,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_in": {
          "values": [
            0.132112305611372,
            0.10495433211326599,
            0.1364753469824791
          ],
          "bootstrap95": {
            "mean": 0.12451399490237236,
            "lower95": 0.10495433211326599,
            "upper95": 0.1364753469824791,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_rec": {
          "values": [
            0.15851402282714844,
            0.1658981293439865,
            0.15111812204122543
          ],
          "bootstrap95": {
            "mean": 0.15851009140412012,
            "lower95": 0.15111812204122543,
            "upper95": 0.1658981293439865,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      }
    },
    "strict_binding": {
      "adam-all minus sgd-all": {
        "direct": {
          "values": [
            0.20703125,
            0.296875,
            0.13671875
          ],
          "bootstrap95": {
            "mean": 0.21354166666666666,
            "lower95": 0.13671875,
            "upper95": 0.296875,
            "n_seeds": 3
          },
          "signflip_p": 0.25,
          "operationally_material_mean_5pp": true,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            0.19140625,
            0.32421875,
            0.12109375
          ],
          "bootstrap95": {
            "mean": 0.21223958333333334,
            "lower95": 0.12109375,
            "upper95": 0.32421875,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "mean_loss": {
          "values": [
            -0.22550528623231925,
            -0.38951373046294413,
            -0.03022295741344927
          ],
          "bootstrap95": {
            "mean": -0.21508065803623755,
            "lower95": -0.38951373046294413,
            "upper95": -0.03022295741344927,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "train_seconds": {
          "values": [
            0.17067120000137947,
            -0.06707129999267636,
            0.03810830000293208
          ],
          "bootstrap95": {
            "mean": 0.047236066670545064,
            "lower95": -0.06707129999267636,
            "upper95": 0.17067120000137947,
            "n_seeds": 3
          },
          "signflip_p": 0.75
        },
        "hidden_delta_in": {
          "values": [
            1.470839984714985,
            1.2569165229797363,
            1.3403631150722504
          ],
          "bootstrap95": {
            "mean": 1.3560398742556572,
            "lower95": 1.2569165229797363,
            "upper95": 1.470839984714985,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_rec": {
          "values": [
            2.069783493876457,
            2.5966357588768005,
            1.3350759595632553
          ],
          "bootstrap95": {
            "mean": 2.0004984041055045,
            "lower95": 1.3350759595632553,
            "upper95": 2.5966357588768005,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      },
      "adam-rotating minus sgd-rotating": {
        "direct": {
          "values": [
            -0.07421875,
            0.1015625,
            -0.01171875
          ],
          "bootstrap95": {
            "mean": 0.005208333333333333,
            "lower95": -0.07421875,
            "upper95": 0.1015625,
            "n_seeds": 3
          },
          "signflip_p": 1.0,
          "operationally_material_mean_5pp": false,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            -0.01953125,
            0.0859375,
            -0.078125
          ],
          "bootstrap95": {
            "mean": -0.00390625,
            "lower95": -0.078125,
            "upper95": 0.0859375,
            "n_seeds": 3
          },
          "signflip_p": 1.0
        },
        "mean_loss": {
          "values": [
            0.18497569565984828,
            0.08377116764617276,
            0.09759708560749258
          ],
          "bootstrap95": {
            "mean": 0.12211464963783787,
            "lower95": 0.08377116764617276,
            "upper95": 0.18497569565984828,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "train_seconds": {
          "values": [
            0.21823799999401672,
            0.3453910999960499,
            0.5685829000067315
          ],
          "bootstrap95": {
            "mean": 0.3774039999989327,
            "lower95": 0.21823799999401672,
            "upper95": 0.5685829000067315,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_in": {
          "values": [
            1.0923509448766708,
            1.321076126769185,
            1.1378259155899286
          ],
          "bootstrap95": {
            "mean": 1.1837509957452614,
            "lower95": 1.0923509448766708,
            "upper95": 1.321076126769185,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_rec": {
          "values": [
            1.2299187779426575,
            1.2030881717801094,
            0.9675583597272635
          ],
          "bootstrap95": {
            "mean": 1.1335217698166768,
            "lower95": 0.9675583597272635,
            "upper95": 1.2299187779426575,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      },
      "adam-all minus adam-rotating": {
        "direct": {
          "values": [
            0.2109375,
            0.1875,
            0.17578125
          ],
          "bootstrap95": {
            "mean": 0.19140625,
            "lower95": 0.17578125,
            "upper95": 0.2109375,
            "n_seeds": 3
          },
          "signflip_p": 0.25,
          "operationally_material_mean_5pp": true,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            0.1953125,
            0.2421875,
            0.1484375
          ],
          "bootstrap95": {
            "mean": 0.1953125,
            "lower95": 0.1484375,
            "upper95": 0.2421875,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "mean_loss": {
          "values": [
            -0.38812097475519836,
            -0.45378940652939415,
            -0.16846137259393035
          ],
          "bootstrap95": {
            "mean": -0.3367905846261743,
            "lower95": -0.45378940652939415,
            "upper95": -0.16846137259393035,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "train_seconds": {
          "values": [
            -0.20653139999922132,
            -0.23573499999474734,
            -0.7205434000061359
          ],
          "bootstrap95": {
            "mean": -0.38760326666670153,
            "lower95": -0.7205434000061359,
            "upper95": -0.20653139999922132,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_in": {
          "values": [
            0.430985689163208,
            -0.03117656707763672,
            0.23809432983398438
          ],
          "bootstrap95": {
            "mean": 0.2126344839731852,
            "lower95": -0.03117656707763672,
            "upper95": 0.430985689163208,
            "n_seeds": 3
          },
          "signflip_p": 0.5
        },
        "hidden_delta_rec": {
          "values": [
            0.8940106630325317,
            1.4375208616256714,
            0.40696191787719727
          ],
          "bootstrap95": {
            "mean": 0.9128311475118002,
            "lower95": 0.40696191787719727,
            "upper95": 1.4375208616256714,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      },
      "sgd-all minus sgd-rotating": {
        "direct": {
          "values": [
            -0.0703125,
            -0.0078125,
            0.02734375
          ],
          "bootstrap95": {
            "mean": -0.016927083333333332,
            "lower95": -0.0703125,
            "upper95": 0.02734375,
            "n_seeds": 3
          },
          "signflip_p": 0.75,
          "operationally_material_mean_5pp": false,
          "holm_12contrasts_p": 1.0
        },
        "probe_gain": {
          "values": [
            -0.015625,
            0.00390625,
            -0.05078125
          ],
          "bootstrap95": {
            "mean": -0.020833333333333332,
            "lower95": -0.05078125,
            "upper95": 0.00390625,
            "n_seeds": 3
          },
          "signflip_p": 0.5
        },
        "mean_loss": {
          "values": [
            0.022360007136969173,
            0.019495491579722746,
            -0.040641329572988494
          ],
          "bootstrap95": {
            "mean": 0.0004047230479011417,
            "lower95": -0.040641329572988494,
            "upper95": 0.022360007136969173,
            "n_seeds": 3
          },
          "signflip_p": 1.0
        },
        "train_seconds": {
          "values": [
            -0.15896460000658408,
            0.1767273999939789,
            -0.1900688000023365
          ],
          "bootstrap95": {
            "mean": -0.05743533333831389,
            "lower95": -0.1900688000023365,
            "upper95": 0.1767273999939789,
            "n_seeds": 3
          },
          "signflip_p": 0.75
        },
        "hidden_delta_in": {
          "values": [
            0.05249664932489395,
            0.03298303671181202,
            0.035557130351662636
          ],
          "bootstrap95": {
            "mean": 0.040345605462789536,
            "lower95": 0.03298303671181202,
            "upper95": 0.05249664932489395,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        },
        "hidden_delta_rec": {
          "values": [
            0.054145947098731995,
            0.043973274528980255,
            0.039444318041205406
          ],
          "bootstrap95": {
            "mean": 0.04585451322297255,
            "lower95": 0.039444318041205406,
            "upper95": 0.054145947098731995,
            "n_seeds": 3
          },
          "signflip_p": 0.25
        }
      }
    }
  },
  "control_trials": 45,
  "resources": {
    "peak_rss_bytes": 751308800,
    "peak_commit_or_vms_bytes": 1251741696,
    "os_process_peak_working_set_bytes": 754135040,
    "os_process_peak_pagefile_commit_bytes": 1255206912,
    "samples": 2903,
    "interval_seconds": 0.1,
    "limit_bytes": 2000000000,
    "observed_over_limit": false,
    "elapsed_seconds": 301.82072110000445,
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
  "torch_version": "2.13.0+cu130",
  "family_reopened": false,
  "new_local_candidate": false,
  "deployed": false,
  "normal_chat_or_vision_proven": false,
  "retention_proven": false,
  "limitations": [
    "Three development seeds; no confirmatory efficacy claim.",
    "Global Adam clock; inactive weights/vector moments frozen.",
    "Full BPTT graph computed for rotating controls; no sparse efficiency claim.",
    "Equal LR is not equal effective step between optimizers."
  ]
}
