# Argument audit — Round 6b

## Why the failure looks large

Claim: 0.90 accuracy and 0.67 both-correct describe different events.
Prediction: both-correct is approximately the product of the two query-conditional accuracies, and case B is near 1.
Evidence: means 0.812 and 0.804, product 0.652, both-correct 0.668, both-B from 0.929 to 1.000.
Counterevidence: the product is 0.016 below both-correct, so the errors are slightly less independent than a pure product.
Alternative: a scorer drops pairs. The oracle matches every label, siblings share the first ten ticks, and the query tick is 10.
Falsification: a seed whose both-correct is far from the product of its two conditionals. None of the eight is.
Verdict: the headline gap is the conjunction, not a separate collapse.

## Capacity

Claim: the R5 features cannot represent case A.
Prediction: ridge both-correct stays below 0.75.
Evidence against: the eight-seed mean is 0.986, and the worst seed in the table is 0.959.
Verdict: H1 is false for these frozen features. The linear decoder is not an information ceiling; it is enough to reject "the representation cannot do this."

## The clip

Claim: deleting the per-step clip reaches the ridge solution.
Prediction: both-correct rises by at least 0.05 with a positive seed-level lower bound, and protected tasks do not fall by more than 0.03.
Evidence against: gain 0.010, lower bound −0.071, order paired-flip falls by 0.187.
Alternative: more than one epoch would climb the rest of the way. That budget was not changed, because a longer budget would not be the locked one-factor comparison.
Verdict: the clip is not a sufficient repair. The toy tests still hold: on a consistent coordinate the unclipped weight moves farther, and on a huge contradictory coordinate it overshoots. The R5 features are the second situation often enough that the screen fails.

## First key and projection

Claim: query B is systematically worse.
Prediction: the conditional gap is at least 0.10.
Evidence against: the mean gap is 0.008 and several seeds have B above A.
Claim: one projection draw explains the miss.
Evidence: on seed 98001 only, draws 1234, 7 and 99 score 0.590, 0.746 and 0.705.
Counterevidence: this is one hidden initialization. The other seven seeds were not crossed with the other draws.
Verdict: H3 is not supported. H4 is a real sensitivity on one seed and not a chosen solution.

## What remains unidentified

The constraint that stops one clipped pass at about 80% per query, given a linear solution at 99%, was not isolated. Candidates that remain confounded with each other are the weight clip at 3, the single pass, the z-score floor, and the 192-to-32 projection. The experiment separated the step clip from those and found it is not enough. It did not separate the rest.

## Decision

STOP, from the preregistered gates. The historical both-correct number is reproduced. The representation is not incapable. The unclipped local update does not repair case A across new seeds and it regresses order.
