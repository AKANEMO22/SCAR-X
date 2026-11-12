# Protocol v1 — online readout mechanism

Branch `research/grok/r4-online-readout-mechanism`. Base `main` `495dab2f4bac0180e4d1cd28674391f055ccf09d`.
Sources inspected, not modified: DeepSeek R3A head in `r3_screen.py` at `42ed574`; DeepSeek R2A head in `r2_screen.py` (named commit `7d590f7`); Grok R3B `211380f`; Grok R2B STOP `3bcf609`.
Production readout is `UnifiedCore.learn`: no feature normalization, learning rate 0.025, per-step clip 0.025, then weight clip 20. The R2A/R3B normalized head uses training-batch z-score, standard-deviation floor 0.05, feature clip [-5, 5], learning rate 0.08, per-step clip 0.025, weight clip 3, initial weights normal(0, 0.03) from seed+991. This protocol gives every arm the same initial weights (salt 991). The raw arm still uses the production rate and clips. Hidden weights stay frozen. `src/scarx/neural/` is not edited.

## Hypotheses, locked before scores

H1. The label is not linearly present. Falsified on a task if ridge paired-flip or ridge accuracy is at least 0.80.
H2. The label is present, but scale, sparsity or conditioning blocks the delta rule. Falsified if removing only the standard-deviation floor and the feature clip does not raise held-out accuracy by 0.10.
H3. Per-step clipping blocks the update. Falsified if removing only the per-step clip does not raise held-out accuracy by 0.10. Clipping saturation is recorded and is not, by itself, the conclusion.
H4. Fit and generalization are different. A near-chance training accuracy is a fit failure. A high training accuracy with a collapsed held-out accuracy is a generalization failure. Loss is not the classifier.
H5. Order and the timing of normalization matter. The three orders are the generator order, one permutation from rng 96001, and class alternation. Causal normalization uses only past examples. Falsified if those three held-out accuracies differ by less than 0.05 and the causal arm does not beat the batch arm by 0.10.

## Reproduction bar

Seeds 94011, 94017, 94023. Order pairs from `order_pairs(94102)` and `order_pairs(94104)`. Decay-0.25 ridge paired-flip mean at least 0.80, normalized-head paired-flip mean at most 0.40, and bank5 normalized-head paired-flip mean at least 0.90. If this bar fails, stop.

## One candidate rule

Arms, one change from `norm` except `raw`:

| arm | change |
|---|---|
| norm | batch z-score, floor 0.05, feature clip 5, lr 0.08, step clip 0.025, weight clip 3 |
| raw | no z-score, lr 0.025, step clip 0.025, weight clip 20 |
| noclip | step clip removed |
| nofloor | floor 0 and feature clip removed |
| nonorm | z-score removed, lr and clips kept |
| causal | z-score from examples already seen, then the current example is absorbed |

On the three reproduction seeds, look only at `noclip`, `nofloor`, `nonorm`, and `causal` against `norm` on order-reversal decay features. Promote the arm with the largest mean held-out paired-flip gain if that gain is at least 0.10, the original training accuracy is below 0.65, and the same arm does not lower simultaneous-XOR decay held-out accuracy by more than 0.03. Otherwise there is no candidate and the verdict is STOP.

## Screen

New weight seeds 96011, 96017, 96023. Data seeds 96102 and 96104. Tasks: simultaneous XOR, separated XOR (bits on ticks 2 and 6), order reversal, query retrieval, query swap, delay at tick 11 with a head trained at tick 6, and one paired distractor. One epoch. The second order is the permutation from rng 96001.

Interval is `phase21.interval`. GO only if all hold:

1. On at least two tasks whose ridge is at least 0.80, the candidate's held-out accuracy exceeds `norm` by at least 0.10.
2. The paired gain on those tasks has lower bound above 0.
3. Order paired-flip also gains at least 0.10 with lower bound above 0.
4. No task mean falls by more than 0.03.
5. The same two-task gain holds under the second training order.
6. Test rows are not used to form batch statistics. A 32-draw shuffled-label null is reported; one draw is not called leakage.
7. Update count matches `norm`.
8. Sampled RSS stays under 2e9 and the screen under 600 seconds.

Confirmation seeds 97011, 97017, 97023, 97029, 97031, 97037, 97041, 97043 run only after GO. Query swap labels are `1 - y` because the generator stores complementary values.
