# Protocol v1 — independent confirmation repair and one readout factor

Branch `research/grok/r5-independent-audit`. Base `main` `495dab2f4bac0180e4d1cd28674391f055ccf09d`.
R4A sources are copies of `10e1c7df9a585566db04e754af9ba7b419d8bab1`, hashes recorded in `r5_audit.py`. They are not edited. The historical confirmation mean 0.6394 is not rewritten.

Forensic rows are `CORRECTED_REPLAY` on seeds `96001, 96007, 96013, 96019, 96023, 96029, 96031, 96037`. They are not a fresh confirmation and cannot change the historical STOP into a GO.

F1 is the published runner: capture at tick 11, cue at tick 10. F2 reads the cue at tick 11. F3 moves only pulse coordinates and keeps background. F4 trains on clean tick 10 and tests F3 without another fit. F0 is the published aggregate, not a rerun.

## Safe normalization, locked before its scored screen

This changes one factor relative to the normalized head: a coordinate with training standard deviation at or below `1e-4` is set to 0, and every other coordinate is divided by its own standard deviation. The feature clip stays [-5, 5]. The learning rate stays 0.08. The step clip stays 0.025. The weight clip stays 3. The initial weights stay normal(0, 0.03) with salt 991. `1e-4` is the dead-coordinate cutoff already used to describe these features. It is not chosen from a scored screen.

Prediction: on order reversal, paired flip rises by at least 0.10 because the fourteen small-variance signal coordinates are no longer divided by 0.05. Falsification: the gain is below 0.10, or simultaneous XOR falls by more than 0.03, or any coordinate produces a non-finite basis.

Screen seeds `98011, 98017, 98023`. Tasks: order reversal, simultaneous XOR, separated XOR, query retrieval, delay at tick 11 after training at tick 6, and one added pulse on an otherwise identical order row. One epoch. Second order is the permutation from rng 96001. Interval is `phase21.interval` on the three seeds.

GO only if the candidate's order paired-flip gain is at least 0.10 with lower bound above 0, at least one other task with ridge at least 0.80 gains at least 0.10 with lower bound above 0, no task falls by more than 0.03, the second order still shows the order gain, the basis stays finite, and test rows are not observed. Otherwise STOP. Confirmation seeds `98111, 98117, 98123, 98129, 98131, 98137, 98141, 98143` run only after GO.
