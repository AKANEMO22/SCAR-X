# Argument audit — Round 5a

## Claim ledger

Claim: the published shift accuracy 0.6394 is the accuracy of a correctly read shifted query.
Prediction: replaying capture at tick 11 and the cue at tick 10 matches 0.6394, and the cue is key 0 on every row.
Evidence: eight seeds match the published per-seed values to the recorded rounding, and the key-1 fraction is 0.
Counterevidence: none on the replay identity.
Alternative: the match could be a coincidence of means. It is not. Each seed matches.
Falsification: any seed whose replay differs in the fourth decimal from the published file.
Verdict: the published number is the buggy cue. Historical STOP stays on the record and is not converted to GO.

Claim: R5 cannot handle a one-tick shift.
Prediction: after the cue is moved to tick 11, accuracy stays below 0.65.
Evidence against: F2 mean 0.982, F3 mean 0.885, F4 mean 0.910. The lowest F3 seed is 0.746.
Alternative: F2 still trains on the shifted distribution, so it is not transfer. F4 is the transfer test and still clears 0.65.
Falsification: a pure shift, with background retained and cue at tick 11, whose seed-level accuracy falls below 0.65.
Verdict: the temporal-alignment claim, as measured, is withdrawn. A new confirmation would be required before calling R5 shift-robust. This replay is not that confirmation.

Claim: the 0.05 floor causes the order underfit.
Prediction: changing only the scale rule, keeping the feature clip, rate, and step clip, raises paired-flip by at least 0.10.
Evidence: historical seeds move from about 0.15 to about 0.99. Fresh seeds move by 0.802 with lower bound 0.695. The second order does the same.
Counterevidence: simultaneous XOR does not move, and query falls by 0.042. The floor is not the only obstacle on every task.
Alternative: the safe rule also zeros coordinates below 1e-4, so it is not a pure rescaling of every coordinate. That zeroing was the protection against division by zero. It is one locked rule, not a second tuned factor.
Falsification: a repeat on new generator families where the strong coordinates are already above 0.05 and the gap remains. That was not run.
Verdict: causal for this order task. Not a general repair. Screen STOP.

## Mathematical audit

The cue is `(1 - h, h)` with `h = 1` when input channel 5 at the chosen tick exceeds 0.5. Key 1 is a pulse on channel 5. Key 0 is a pulse on channel 4 and a quiet channel 5. Reading tick 10 after the pulse has moved to tick 11 makes `h = 0` for every row.

The safe basis is 0 when the training standard deviation is at most 1e-4. Otherwise it is the centered coordinate divided by that standard deviation, then clipped to [-5, 5]. The update remains `clip(0.08 * error * basis, -0.025, 0.025)` and the weights remain clipped to 3. No label enters the scale.

## Statistical audit

Forensic units are the eight old confirmation seeds. They are dependent on the published run by construction. The screen units are three new weight seeds. The interval resamples those three seeds, not the rows inside a seed. Query's loss is a mean over three seeds and clears the 0.03 regression line; with three seeds that line is coarse.

## Objections

The replay might have copied the published numbers. Response: F1 was recomputed from the generators and the unchanged cue function. It landed on the published value. F2 used the same weights and a different tick.

F3 could still be easy because the complementary-value shortcut remains. Response: the label is the stored bit XOR the query. A correct cue is required to choose which stored bit. F1, with a constant cue, scores 0.639. F2, with a balanced cue, scores 0.982. The cue is doing the work.

The safe head might have been tuned. Response: `1e-4` was fixed from the earlier dead-coordinate definition before this screen. No other rate or clip was changed. The screen still failed, so there was no post-hoc rescue.

## Withdrawn or corrected

Withdrawn: reading the published 0.6394 as evidence that R5 fails a genuine one-tick shift.
Corrected: Round 4b's `nofloor` arm was not a clean test of the floor, because it also removed the feature clip and divided by zero. The safe arm is the clean test. It supports the floor account on order and rejects it as a general fix.
Unchanged: the historical confirmation verdict remains STOP. The product readout is unchanged.
