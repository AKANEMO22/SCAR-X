# Round 3b — temporal stress

Code: `04ce511` on `research/grok/r3-temporal-stress`, base `495dab2f4bac0180e4d1cd28674391f055ccf09d`.
Protocol, committed before the run: `experiments/collab/scarx-neural-round1-20261111/grok/protocol-r3b-v1.md`.
Numbers: `runs/grok-r3b-temporal-stress-v1/result.json`.
Command: `PYTHONPATH=src` and one numerical thread, then `python scripts/grok_r3b_temporal_stress.py`.
Compute 7.15 s, sampled RSS 41,795,584 bytes, RSS growth across the 1000-episode loop 5,124,096 bytes. Verdict: **STOP**. Confirmation was not opened. DC-ALIF was not run.

## Stage 0

Independent rollout against `SpikingCore.step` on their seeds `81001`, `81007`, `81013`, hard threshold, frozen hidden, 1024 train / 512 test.

| seed | decay ridge | decay head | spike-sum ridge | spike-sum head |
|---|---:|---:|---:|---:|
| 81001 | 0.711 | 0.467 | 0.996 | 0.984 |
| 81007 | 0.793 | 0.498 | 0.994 | 1.000 |
| 81013 | 0.775 | 0.490 | 0.951 | 0.965 |
| mean | 0.760 | 0.485 | 0.980 | 0.983 |

Feature gap versus `SpikingCore.step` is 0. Resume gap is 0. Hidden weights do not change. This matches Round 2A's development screen (decay head 0.485, spike-sum head 0.983, spike-sum ridge 0.980) to the reported rounding. The same-tick contrast is reproduced. It is not a bit-level claim about their process, and it is not a ceiling on information.

## What fails

Normalized spike-sum head, three fresh seeds:

| test | result |
|---|---|
| Order-reversal paired flip | 0.500, 0.578, 0.516. Mean 0.531, lower bound 0.500 |
| Same test, decay 0.25 | 0.320, 0.461, 0.211 |
| Same test, five-decay bank | 0.992, 1.000, 1.000 |
| Query permutation | 0.586, 0.547, 0.555. First-key policy 0.500 |
| Query, five-decay bank | 0.617, 0.570, 0.523 |
| Delay ticks 3, 6 and 11, spike-sum | 1.000 on every seed |
| Delay, decay 0.25 | about 0.50 at every lag, including lag 3 |
| Stream sim_xor, reset / persistent / inserted gap | 0.964 / 0.953 / 0.955 |
| Stream order, those three modes | 0.637 / 0.675 / 0.629 |

Spike counts do change when event order is swapped (mean absolute feature gap 0.18 to 0.26). The normalized head still does not use that change. The five-decay bank does, on this order test only. None of the three substrates answers a swapped query.

Persistent voltage does not remove the same-tick XOR from the spike sum. Decay 0.25 was already at chance before persistence.

A head trained on shuffled `sim_xor` labels scores 0.621 on the true test labels. The locked line was 0.60, so that control fails. Calibration mean and count do not change when test features are transformed. The no-input head scores 0.500.

## Round 2A audit

`S5_no_leakage` in their screen is the literal `True`. Their ridge parity check calls the repository probe on features it just built. The tests in `tests/test_temporal_stress.py` fail if labels are permuted, if test columns are permuted, or if calibration keeps accepting samples after it is frozen. Six of those tests passed before this screen.

## Gates

G1, G2, G5, G6, G7 and G8 pass. G3 and G4 fail. The random-label control fails its locked line. Status STOP.

## Answers

1. Spike-sum does not keep enough order information for a paired flip above the locked bound. A five-decay bank does, on this one order test.
2. Decay 0.25 fails a channel identity that must wait even three ticks. Spike-sum survives that delay and fails order reversal, query permutation, and is weaker at length 16 than at length 8.
3. Not on these tasks. Persistent state changed spike-sum `sim_xor` by about one point.
4. Yes. The same-tick ridge and online-head contrast reproduces.
5. No. The new tasks limit the scope of that contrast. They do not contradict it.
6. Shown: same-tick XOR is in the spike counts and this normalized head can read those counts; the decay-0.25 end feature leaves that head at chance; order and query are not solved by spike-sum; persistence of the ALIF state across 12-tick windows does not erase `sim_xor`. Not shown: a transferable temporal code, a product change, or a reason to reopen DC-ALIF.
