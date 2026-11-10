# MTB screen run

Scope: Registered synthetic tasks at H=48. No chat/vision/reasoning claim.

## Per-task acquisition (dev_direct, %)

| condition | order | xor | strict_binding |
|---|---|---|---|
| prod | 51.43 | 50.65 | 52.47 |
| mtb | 68.88 | 48.05 | 50.39 |
| prod_x240 | 55.21 | 51.69 | 49.61 |
| mtb_one_decay | 53.39 | 49.61 | 49.61 |
| prod_frozen | 51.43 | 50.00 | 51.04 |
| mtb_frozen | 66.54 | 49.61 | 49.87 |
| prod_readout | 51.43 | 50.00 | 51.04 |
| mtb_readout | 66.54 | 49.61 | 49.87 |
| mtb_shuffle | 66.15 | 48.83 | 50.39 |
| mtb_only_fast | 48.57 | 53.12 | 50.00 |
| mtb_only_slow | 65.89 | 49.87 | 50.00 |

## Attribution contrasts (direct accuracy, pp)

| contrast | mean [95% CI] |
|---|---|
| mtb_minus_prod_x240 | +3.60pp [-1.30, +8.77] |
| mtb_minus_mtb_one_decay | +4.90pp [-0.22, +10.42] |
| mtb_minus_prod | +4.25pp [-1.65, +11.07] |
| mtb_minus_mtb_shuffle | +0.65pp [-0.95, +2.30] |
| mtb_minus_mtb_frozen | +0.43pp [-0.91, +1.87] |
| mtb_minus_mtb_readout | +0.43pp [-0.91, +1.87] |
| mtb_frozen_minus_prod_frozen | +4.51pp [-0.09, +10.07] |
| prod_minus_prod_frozen | +0.69pp [-1.17, +2.39] |
| mtb_minus_mtb_only_slow | +0.52pp [-1.13, +2.08] |

## Screen gates

```json
{
  "G5_mtb_beats_dim_matched": false,
  "G6_mtb_beats_single_timescale": false,
  "G7_signal_matters": false,
  "spike_identity_holds": false,
  "spike_identity_within_rule_groups": {
    "learning_score": false,
    "frozen": true,
    "readout_only": true,
    "shuffled_signal": true
  },
  "mtb_frozen_beats_prod_frozen": false
}
```

**screen_pass: False**

Spike-train identity across arms: VIOLATED
