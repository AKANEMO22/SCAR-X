# Protocol r3b-v1 — temporal stress

Branch `research/grok/r3-temporal-stress`, base `main` `495dab2f4bac0180e4d1cd28674391f055ccf09d`.
DC-ALIF round 2B stays stopped. This protocol does not change a neuron, a dendritic gate, or a hidden learning rule. Hidden weights stay frozen in every measured cell.

The forward is the ALIF step in `src/scarx/neural/core.py`: leaky voltage, hard threshold, reset by the limit, adaptation, then `features = 0.25 * features + 0.75 * spikes`. Weights come from `SpikingCore` at the Generation-1 constants `leak=0.65`, `threshold=0.5`, `adaptation_strength=0.02`, `adaptation_decay=0.95`, `feature_decay=0.25`, H=48, fan-in=8. Feature decay is not retuned per sequence length.

Substrates, all from that frozen trajectory, windowed to the current question only:

- `decay025`: end feature
- `spikesum`: sum of spikes inside the question window
- `bank5`: decays 0.10, 0.25, 0.50, 0.75, 0.90 concatenated

Readers:

- `raw`: serving-style delta rule, learning rate 0.025, update clip 0.025, no feature normalization (`unified.py`)
- `norm`: z-score from training features only, floor 0.05, clip to [-5, 5], learning rate 0.08, update clip 0.025
- `ridge`: `tasks21.ridge_probe`, a linear diagnostic, not an information ceiling

One epoch. Normalization statistics are frozen before evaluation. Test features are never passed to `observe`. Labels are never used to build features or to decide a reset.

## Stage 0 — reproduction

Seeds `81001`, `81007`, `81013`. `sim_xor`: two bits on one tick in {1,2,3}, label = XOR, marker at tick 10. Calibration offset +100000 (128), train +200000 (1024), test +300000 (512).

The primary contrast reproduces only if, on the normalized reader:

- mean ridge(`spikesum`) ≥ 0.80
- mean ridge(`spikesum`) − mean ridge(`decay025`) ≥ 0.15
- mean norm head(`decay025`) < 0.65
- hidden weights unchanged

If this fails, stop. Do not run Stage 1 as a replacement result.

## Stage 1 — counterfactuals

Screen seeds `94011`, `94017`, `94023`. Train seed `94102` (256), test seed `94104` (128), calibration seed `94103` (128), shared across weight seeds.

- Order reversal: ticks 2 and 6, channels 0 and 1, identical noise, identical marker. The twin swaps the two pulses and flips the label. Metric: fraction of pairs where both predictions match both labels (`correct_flip`).
- Delay: pulse at tick 1, query at tick 3, 6, or 11. One head trained on the mixture. Score each delay with that head.
- Distractors: order task trained with no extra pulse, tested with 0, 1, or 2 extra pulses.
- Query: two values that always differ, query balanced, label = value of the queried key. Report accuracy of the policy "always answer the first key".
- Length: train only length 12 with end padding. Test lengths 8, 12, 16 and both end and start padding.

## Stage 2 — continuous state

Hidden stays frozen. The readout is trained on reset-mode data and then frozen. The question window is 12 ticks. Persistent mode carries voltage, adaptation, and spikes into the next question. It does not accumulate an unbounded spike sum. Mode C inserts 4 noise ticks and one extra pulse between questions. The stream has 1000 episodes, cycling `sim_xor`, order reversal, and medium delay. Stream seed `94180`.

A question boundary is an observable segment of 12 ticks, the same segment length the serving core already resets between. Mode B and Mode C deliberately do not reset. That assumption is stated here, not discovered from labels.

## Gates

Interval is `phase21.interval` (10 000 bootstrap resamples, rng 2101). The normalized `spikesum` head is the reader under the gate. `raw` and `ridge` are reported and cannot be substituted after the fact.

The screen is GO only if every line holds:

1. Stage 0 contrast reproduces.
2. Calibration checksum: predicting on test features does not change the stored mean or count. A deliberately leaky observer fails this check.
3. Order-reversal `correct_flip` mean ≥ 0.65 and lower bound > 0.55.
4. Query-permutation accuracy mean ≥ 0.65 and lower bound > 0.55, and the first-key policy is ≤ 0.55.
5. Delay: short-delay accuracy ≥ 0.65 and long-delay accuracy ≥ short − 0.05.
6. Streaming: on `sim_xor` and order reversal, persistent accuracy ≥ reset accuracy − 0.05, and the same for the distractor-insert mode.
7. Feature width is constant across the 1000 episodes, and RSS sampled immediately before versus after that loop grows by at most 50 MB.
8. Rollout matches `SpikingCore.step` within 1e-4, and resuming the carried state matches an uninterrupted rollout.

Otherwise STOP. Confirmation seeds `95011, 95017, 95023, 95029, 95031, 95037, 95041, 95043` run only after a GO, with the same gates, and their own 600 second cap. A random-label head at or above 0.60, or a no-input head at or above 0.60, is a leakage stop. Wall clock cap for the development screen is 600 seconds.

## Round 2A audit, fixed before this screen

Their screen sets `S5_no_leakage` to the literal `True`. Their ridge parity test calls `tasks21.ridge_probe` on data it just built. This protocol does not treat 0.9719 as a mathematical information ceiling. The metamorphic tests here must fail if labels are permuted, if test columns are permuted, or if test features update the calibration.
