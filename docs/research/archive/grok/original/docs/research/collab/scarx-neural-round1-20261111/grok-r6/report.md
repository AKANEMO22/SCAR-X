# Report — Round 6b

Protocol commit `6a640fd`. Reproduction class `REPRODUCTION` on the eight R5B confirmation seeds. Command: `PYTHONPATH=src`, one numerical thread, `python scripts/grok_r6_causal.py`. Toy tests: `python -m pytest -q tests/test_r6_toys.py` (2 passed before the run). Compute 3.33 s, sampled RSS is in `result.json`. Verdict: **STOP**. No confirmation was opened.

## Reproduction

The online head's case-A both-correct mean is **0.6676**, the published figure. Ridge on the same R5 features, same rows, mean both-correct **0.9858**. Hidden weights do not change. The oracle matches the labels. Siblings share the first ten ticks. The query tick is 10.

| seed | head both-A | ridge both-A | correct q=A | correct q=B | train accuracy | both-B |
|---|---:|---:|---:|---:|---:|---:|
| 98001 | 0.590 | 0.959 | 0.721 | 0.836 | 0.904 | 0.963 |
| 98007 | 0.777 | 0.992 | 0.900 | 0.846 | 0.925 | 1.000 |
| 98013 | 0.631 | 0.992 | 0.815 | 0.754 | 0.912 | 0.929 |
| 98019 | 0.758 | 0.992 | 0.850 | 0.908 | 0.936 | 0.978 |
| 98023 | 0.558 | 1.000 | 0.733 | 0.692 | 0.872 | 0.993 |
| 98029 | 0.797 | 0.968 | 0.846 | 0.943 | 0.959 | 1.000 |
| 98031 | 0.542 | 0.992 | 0.742 | 0.667 | 0.864 | 0.985 |
| 98037 | 0.688 | 0.992 | 0.888 | 0.784 | 0.938 | 1.000 |

Mean query-A accuracy when the values differ is 0.812. Mean query-B accuracy is 0.804. Their product is 0.652, and both-correct is 0.668. The hard metric is the conjunction of two approximately independent 80% events. Overall accuracy stays near 0.90 because case B, where the values match, is 0.96–1.00, and a single-query hit does not require both siblings.

## What was ruled out

H1 is false here: the linear decoder of the same features both-corrects case A at 0.99. H6 is false on the checks above. H3's predicted 10-point query gap is false; the mean gap is 0.008 and the sign flips across seeds. H4, on seed 98001 only, projection seeds 1234, 7 and 99 give both-correct 0.590, 0.746 and 0.705. The range is 0.156. The historical draw is the worst of the three on that seed. The best draw was not selected.

## The candidate

Because ridge both-correct is above 0.75 and the head is below 0.70, the locked rule created one candidate: the same head with the per-step clip removed. On fresh seeds the both-correct gain is 0.010, and the seed-level interval includes negative values. Under the second training order the mean gain is 0.113 and the lower bound is still negative. The unclipped head lowers the protected order paired-flip by 0.187. The screen fails. The product readout is unchanged.

## Answer

Case A stays at 0.67 because each query is only about 80% correct and the pair demands both. The representation can do 0.99. Removing the step clip does not reliably climb from 80% to that solution, and it damages the order task. The projection draw moves one seed by about 15 points, which is a real sensitivity and not a repair.
