# Report — Round 5a

Protocol commit `46e2023`, before these scores. Base `main` `495dab2`. R4A copy is `10e1c7df9a585566db04e754af9ba7b419d8bab1`. Command: `PYTHONPATH=src`, one numerical thread, `python scripts/grok_r5_audit.py`. Tests: `python -m pytest -q tests/test_r5_harness.py` (6 passed before the run). Compute 5.17 s, sampled RSS 44,175,360 bytes. Verdict: **STOP**. The historical R4A STOP is not rewritten. Confirmation was not opened.

## Problem A — the shift result was a cue bug

The published shift mean is 0.6394. Replaying the same eight seeds with capture at tick 11 and the cue still read at tick 10 reproduces that mean exactly: 0.639404296875. On every seed the cue's key-1 fraction is 0. The published shift also zeros 75% of ticks, so it changes the noise distribution as well as the time.

| seed | published | F1 cue at 10 | F2 cue at 11 | F3 pure shift | F4 train clean, test pure |
|---|---:|---:|---:|---:|---:|
| 96001 | 0.613 | 0.613 | 0.938 | 0.801 | 0.799 |
| 96007 | 0.705 | 0.705 | 1.000 | 0.969 | 0.994 |
| 96013 | 0.678 | 0.678 | 0.992 | 0.957 | 0.914 |
| 96019 | 0.594 | 0.594 | 0.992 | 0.949 | 0.994 |
| 96023 | 0.652 | 0.652 | 0.998 | 0.811 | 0.953 |
| 96029 | 0.629 | 0.629 | 0.998 | 0.879 | 0.842 |
| 96031 | 0.678 | 0.678 | 0.938 | 0.746 | 0.889 |
| 96037 | 0.566 | 0.566 | 0.998 | 0.965 | 0.895 |
| mean | 0.639 | 0.639 | 0.982 | 0.885 | 0.910 |

F2 trains and tests inside the shifted distribution with the cue on the query. F4 trains on clean tick 10 and tests the pure shift with no second fit. Both stay above 0.65. The corrected measurement does not show the temporal failure that the published number reported. It is labeled `CORRECTED_REPLAY`. It does not turn the historical confirmation into a GO.

Six harness tests fail the old cue and pass the tick-11 cue. They also show that a sibling pair differs only at tick 10, that labels flip, and that capture does not change hidden weights.

## Problem B — the floor is causal for this order task, and not a general fix

On the three Round 4b seeds the normalized head's order paired-flip is 0.172, 0.180, 0.102. The safe head, which changes only the scale rule, reaches 0.984, 1.000, 1.000. Ridge is 1.000, 0.984, 0.949. The raw serving head stays at 0. Training accuracy of the normalized head remains 0.52–0.58, so the old failure is a fit failure.

Fresh seeds 98011, 98017, 98023, original order:

| task | ridge | norm | safe |
|---|---:|---:|---:|
| order paired-flip | 1.000, 0.934, 0.988 | 0.305, 0.164, 0.117 | 1.000, 1.000, 0.992 |
| simultaneous XOR | 0.742, 0.562, 0.766 | 0.531, 0.531, 0.523 | 0.539, 0.531, 0.523 |
| separated XOR | 0.547, 0.523, 0.531 | 0.531, 0.453, 0.523 | 0.539, 0.484, 0.484 |
| delay, trained at 6 and tested at 11 | 0.539, 0.516, 0.492 | 0.500, 0.500, 0.500 | 0.586, 0.477, 0.508 |
| query | 0.602, 0.516, 0.586 | 0.633, 0.633, 0.578 | 0.625, 0.562, 0.531 |
| paired distractor | 0.812, 0.883, 0.688 | 0.414, 0.344, 0.352 | 1.000, 0.992, 0.977 |

The order gain mean is 0.802 and its lower bound is 0.695. The same gain holds under the locked permutation. Query falls by 0.042, which is more than 0.03. No other task has ridge at least 0.80 and a 10-point gain: the distractor ridge mean is 0.794, just short of that line, and simultaneous XOR stays near 0.53. The screen therefore fails. The product readout is unchanged.

## Answer

The published R5 shift failure is a harness failure: the cue was read one tick before the query and was key 0 on every row. With the cue on the query tick, the same map scores 0.982 inside the shifted distribution and 0.910 when trained on clean data and tested on a pure shift. The variance floor is a cause of the order underfit, because changing only that scale rule closes the order gap. It is not a general readout repair: query gets worse, and simultaneous XOR does not move.
