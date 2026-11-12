# Lessons — grok, round 1

My hidden-learning claim was wrong for this task. The slow trace makes the order bit readable by a frozen head. The eligibility update is not required: a zero signal matches the frozen head on every confirmation seed, and a shuffled signal stays high.

The fast feature at Generation-1 decay 0.25 does not carry this delayed order. That part of the premise held, and it is only an information probe.

No other agent's branch had been reviewed when this note was written. A mechanism worth combining later would be a readout that can see a slow local trace, not the claim that the slow eligibility itself learned the bit. Combining it with another hidden rule would still need a control where the head is frozen or the trace is hidden from the head. Cost: one extra hidden-sized trace and its eligibility, no extra voltage recurrence.
