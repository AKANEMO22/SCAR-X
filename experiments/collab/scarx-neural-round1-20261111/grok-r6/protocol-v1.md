# Protocol v1 — case-A causal readout repair

Branch `research/grok/r6-causal-readout-repair`. Base `main` `495dab2`. R5B source `d6c609df6f90fc8df96fb3de424dd54388f8dc59`. The copies are `r5b_probe.py` and `r5b_independent.py`. Historical confirmation seeds are a `REPRODUCTION`, not a new confirmation.

## Hypotheses

H1. R5 features cannot represent case A. Falsified if a ridge on those exact features has both-correct at least 0.75.
H2. The solution exists and one clipped pass does not reach it. Prediction: ridge is high, the online head's training both-correct stays near its held-out both-correct, and removing only the step clip moves held-out both-correct.
H3. Query B is learned worse than query A when the values differ. Prediction: the two conditional accuracies differ by at least 0.10.
H4. The fixed projection draw is the bottleneck. Prediction: projection seeds 1234, 7 and 99, with the same network and data, have a both-correct range above 0.15. The best draw is reported and is not a candidate.
H5. Three development seeds overstated the result. Prediction: the eight confirmation seeds sit below the screen. This is checked by the reproduction itself.
H6. A scorer or cue bug creates case A. Prediction: siblings do not share facts, or the query tick is wrong, or labels disagree with the oracle. Ruled out before a repair.

## One candidate rule

The candidate exists only if the reproduction has ridge both-correct at least 0.75 and the published head both-correct below 0.70. It is the same R5 head with the per-step clip removed. Learning rate 0.08, z-score floor 0.05, feature clip 5, and weight clip 3 stay. No other factor changes. If the ridge is below 0.75, there is no candidate.

The unclipped step can move farther along a consistent coordinate. It is predicted to fail when a coordinate is huge: the same update then takes a step the clip would have stopped. Those two toy predictions are unit tests, written before the screen.

## Screen

Seeds `120011, 120017, 120023`. Train offset `+200000`, held-out `+300000`, pairs `+400000`, 1024 training rows, one pass. Tasks: clean case A, clean direct, case B, reversed writes, query swap, in-distribution shift, distractors, noise, simultaneous XOR, and order reversal. The second training order is the permutation from rng 96001.

GO only if case-A both-correct is at least 0.75, the gain over the clipped head is at least 0.05 with a seed-level lower bound above 0, case B is at least 0.70, clean direct is at least 0.85, neither protected task falls by more than 0.03, the gain remains under the second order, features stay finite, and the cue tick matches the data. Otherwise STOP. Confirmation seeds `121011, 121017, 121023, 121029, 121031, 121037, 121041, 121043` run only after GO.
