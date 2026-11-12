# Report — Round 4b — online readout mechanism

Code that measured: the commit that adds this report, on `research/grok/r4-online-readout-mechanism`, parent protocol commit `0399578`, base `495dab2f4bac0180e4d1cd28674391f055ccf09d`.
Command: `PYTHONPATH=src`, one numerical thread, `python scripts/grok_r4_readout.py`.
Tests: `python -m pytest -q tests/test_readout_mechanism.py` (4 passed before the run).
Compute 1.37 s, sampled RSS 43,229,184 bytes. Verdict: **STOP**. No candidate was promoted. Confirmation was not opened. DC-ALIF was not run. `src/scarx/neural/` was not edited.

## What was reproduced

Order-reversal, seeds 94011, 94017, 94023, one epoch of the normalized head.

| seed | decay ridge | decay head flip | decay train accuracy | bank5 head flip | clip fraction |
|---|---:|---:|---:|---:|---:|
| 94011 | 1.000 | 0.172 | 0.578 | 0.992 | 0.100 |
| 94017 | 0.984 | 0.180 | 0.578 | 1.000 | 0.055 |
| 94023 | 0.949 | 0.102 | 0.523 | 1.000 | 0.035 |

Training accuracy and held-out accuracy match. The head does not fit the rows it just trained on. This is not a generalization gap.

## What the single-factor arms did

Mean paired-flip change versus `norm` on order, and the simultaneous-XOR held-out drop:

| arm | order gain | XOR drop |
|---|---:|---:|
| noclip | −0.138 | +0.036 |
| nofloor | −0.151 | +0.044 |
| nonorm | −0.151 | +0.023 |
| causal | −0.003 | −0.001 |

Removing the per-step clip, removing the variance floor, or removing normalization made the order fit worse. Causal normalization matched the batch head. The raw serving-rate head stayed at a paired flip of 0 with a clip fraction of 0: its steps are too small to saturate and too small to move.

## Geometry that the arms did not fix

On seed 94011, decay features have median standard deviation 0.0042, effective rank 9.3, sparsity 0.67, and a covariance condition number about 1.5e7. Fifteen coordinates have a class separation above 0.5. Fourteen of those fifteen have standard deviation below the 0.05 floor (median 0.00095). Coordinates that are not strongly separated have median standard deviation 0.113. The floor therefore leaves one strong coordinate at its true scale and compresses the other fourteen, while full-scaling the larger noise coordinates. Ridge uses `std + 1e-4` and reads the small coordinates. Bank5 keeps 14 strong coordinates above that floor, and the same head reaches a paired flip of about 0.99. Spike-sum has only four strong coordinates, all above the floor, and a weaker ridge.

`nofloor` divides by exact standard deviations, including zeros. The run emitted divide-by-zero warnings and the head fell to chance. The floor is load-bearing for dead coordinates and harmful for the small coordinates that carry the order bit. No registered arm separates those two jobs.

## Training order

On seed 94011 the original order gives a paired flip of 0.172. One fixed permutation and class alternation each reach 0.344, with training accuracy 0.672 and 0.707. Held-out accuracy rises with training accuracy. Order changes the result by more than the locked 0.05 band and still leaves the head far from the ridge. It was not a promotable arm.

## Answer

The frozen decay feature contains a linear order code. The normalized head misses it because one pass of the delta rule, with this floor, does not move along the small-variance coordinates that carry the label. Clipping is uncommon (3–10% of coordinates) and removing it hurts. No registered local change closed the gap by 10 points without a simultaneous-XOR regression, so no candidate was screened.
