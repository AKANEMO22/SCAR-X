# deepseek screen run

Scope: Registered synthetic tasks at H=48. No chat/vision/reasoning claim.

## Acquisition (direct %, mean of dev seeds)

| arm | pair_binding | strict_binding | order |
|---|---|---|---|
| score | 48.96 | 48.70 | 55.86 |
| frozen | 51.43 | 49.61 | 51.04 |
| readout | 48.57 | 49.09 | 51.69 |
| shuffle | 49.48 | 51.30 | 52.60 |

## Contrasts per task (pp)

| task | contrast | mean [95% CI] |
|---|---|---|
| pair_binding | score_minus_frozen | -2.47 [-2.73, -1.95] |
| pair_binding | score_minus_readout | +0.39 [-1.56, +1.56] |
| pair_binding | score_minus_shuffle | -0.52 [-1.95, +0.39] |
| pair_binding | gain_score_minus_frozen | +1.82 [-0.39, +5.86] |
| strict_binding | score_minus_frozen | -0.91 [-3.91, +2.73] |
| strict_binding | score_minus_readout | -0.39 [-5.08, +3.91] |
| strict_binding | score_minus_shuffle | -2.60 [-3.52, -1.17] |
| strict_binding | gain_score_minus_frozen | -0.52 [-2.34, +1.95] |
| order | score_minus_frozen | +4.82 [-1.17, +9.38] |
| order | score_minus_readout | +4.17 [+2.34, +5.86] |
| order | score_minus_shuffle | +3.26 [+1.95, +5.47] |
| order | gain_score_minus_frozen | -2.21 [-5.08, +0.00] |

## Gates

```json
{
  "C1_task_valid": true,
  "C2_frozen_below_gate": true,
  "C3_learning_measurable": false,
  "score_beats_readout": false,
  "score_beats_shuffle": false,
  "frozen_hidden_untouched": true
}
```

**screen_pass: False**

unknown gates: ['C1_task_valid']
