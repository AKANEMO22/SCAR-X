# Argument audit — Round 4b

## Hypothesis ledger

H1, information absent. Prediction: ridge paired-flip below 0.80. Evidence against: decay ridge is 1.000, 0.984, 0.949. Falsified for this order task. It remains the right description of query retrieval in Round 3b, where every ridge was 0.50–0.57. That task was not re-fit here because the reproduction bar already selected the task where information is present.

H2, geometry. Prediction: ridge succeeds, the head fails, and a scale change fixes the head. Evidence for the first two clauses: the table in the report, median standard deviation 0.0042, and 14 of 15 strong coordinates below the floor. Evidence against the registered fix: `nofloor` and `nonorm` both lose about 0.15 paired-flip. Remaining uncertainty: an untested rule that z-scores only coordinates above the floor, and zeros the rest, might separate the dead coordinates from the small signal coordinates. It was not run. It is not a result.

H3, clipping. Prediction: saturation is high and removing the clip fixes the fit. Evidence against: clip fraction is 0.035–0.100, and `noclip` lowers paired-flip by 0.138 and also drops simultaneous XOR. Falsified as the primary cause. Clipping can still interact with a bad direction; it is not what creates the gap.

H4, fit versus generalization. Prediction: train and held-out come apart if the failure is overfitting. Evidence: train accuracy 0.523–0.578 and held-out accuracy 0.523–0.574. The failure is fit. Falsified as a generalization story.

H5, order and causal statistics. Prediction: order or causal normalization changes held-out flip by at least 0.05, or causal beats batch by 0.10. Evidence for order: permutation and class alternation move flip from 0.172 to 0.344. Evidence against causal normalization: gain −0.003. Order is a real, incomplete effect. It does not reach the ridge and was not eligible to become the candidate.

## Mathematical audit

The head probability is softmax of `weights @ basis(x) + bias`. The loss is cross-entropy. Its weight derivative is `-error[:, None] * basis[None, :]` with `error = one_hot - probability`. The implemented unclipped step is `learning_rate * error[:, None] * basis[None, :]`. A float32 finite difference at `1e-4` matches that derivative within `4e-4`. A wrong sign would miss by order 1. The same test does not differentiate a hard spike.

`norm` then clips each coordinate to 0.025 and the weights to 3. `raw` uses learning rate 0.025, the same step clip, weight clip 20, and `basis(x) = x`, matching `UnifiedCore.learn` on the readout and not on the hidden synapses. Every arm here starts from normal(0, 0.03) with salt 991 so the initial matrix is the same.

## Causal audit

`noclip` changes only the per-step clip. Result rejects clipping as the remedy.

`nofloor` changes the floor from 0.05 to 0 and removes the feature clip. Dead coordinates then divide by zero. Result rejects "any more accurate scale" as an automatic fix.

`nonorm` changes only whether a z-score is applied, keeping the 0.08 rate and the clips. Result is chance on the original order.

`causal` changes only when the mean and second moment are updated. Result matches `norm`.

`raw` changes rate, weight clip, and normalization together. It is the production readout, not a one-factor arm, and it does not move. Its clip fraction is 0, so the failure is a small step, not a saturated one.

No arm changed the frozen ALIF weights, the labels, or the update count (256 order rows, one pass).

## Statistical audit

The reproduction unit is a weight seed. There are three, the Round 3b seeds, not a new sample. The interval method was not needed for the selection because no arm's mean gain was positive. The screen seeds were not used. Selection bias: the task was chosen because Round 3b already showed a ridge-head gap. The arms and the 0.10 rule were committed in `0399578` before this run. The geometry counts were computed after the run, on the same frozen features, and were not used to promote an arm.

## Alternative explanations

The head might need more than one epoch. That was not varied. It would be a budget change, and Round 2a already found that more data did not make the decay head reach its ridge on simultaneous XOR.

The 0.34 flip from a new order might grow into a solution under a different loss. Only three orders were registered. None was promoted.

Bank5 might be easier only because it has 240 coordinates. Its label-shuffle ridge in Round 3b stayed near 0.5, and 14 of its strong coordinates sit above the floor while decay has one. Dimension count alone is not the measurement that separates them.

## Rebuttal

A reviewer can say the floor story was fit after seeing the failure. Response: the floor was a pre-registered arm. It failed. The count of strong coordinates below the floor is an explanation of that failure, not a successful new learner.

A reviewer can say three seeds and one order task are too small to reject every local update. Response: the claim is narrower. These four registered changes do not close this gap. An untested fifth change is not a result.

A reviewer can say training accuracy near 0.57 with a two-class balanced set is not chance, so the head learned something. Response: it did. Held-out accuracy equals that partial fit, and the paired flip stays near 0.17 against a ridge flip near 0.98. Partial movement is not the accessible linear map.

## Conclusion

Demonstrated: on this order task the decay feature is linearly readable; the normalized head does not fit it; clipping is infrequent and harmful to remove; the variance floor crushes 14 of 15 strong coordinates; bank5 leaves 14 strong coordinates above the floor and the same head succeeds; causal normalization does not matter; training order moves the result and does not finish it.

Suggested, untested: zero coordinates below the floor and z-score the rest.

Falsified: the gap is missing information; the gap is only clipping; the gap is held-out overfitting; removing the floor is a sufficient fix.

Unmeasured: a screened candidate, confirmation seeds, query retrieval under these arms, and any production change.
