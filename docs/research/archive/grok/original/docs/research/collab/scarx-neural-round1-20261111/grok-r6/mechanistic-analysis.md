# Mechanistic analysis — Round 6b

The published head's case-A both-correct is the fraction of differing-value pairs in which both the A-query sibling and the B-query sibling are correct. It is not the fraction of individual rows.

On the eight confirmation seeds the two conditional accuracies average 0.812 and 0.804. If those errors were independent, both-correct would be about 0.652. The measured both-correct is 0.668. The gap between 0.90 overall and 0.67 both-correct is therefore mostly arithmetic: case B is nearly perfect, and case A multiplies two imperfect queries.

A ridge using the identical R5 matrix, the identical labels and the identical pairs reaches 0.986 both-correct. The rows are linearly separable enough for a closed-form linear decoder. The one-pass clipped delta rule stops at a marginal solution near 80% per query. Its training accuracy on unpaired rows is 0.86–0.96, so it does fit the marginal task it was trained on. It does not fit the joint solution the ridge finds.

Removing only the step clip was the pre-registered attempt to continue that optimization. On three new seeds the both-correct gain is 0.010 and the lower bound is −0.071. The same change cuts order paired-flip by 0.187. The clip is not the lever that turns the marginal solution into the joint one.

The projection check on seed 98001 shows the fixed draw 1234 is 0.156 below the better of two other locked draws. That can contribute to seed variance. It was not turned into a selected projection.
