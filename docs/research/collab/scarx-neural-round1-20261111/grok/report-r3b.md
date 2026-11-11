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

## Careful pass — ridge on the same splits

The screen above scored the normalized head and treated a low score as a missing code. That was incomplete. `scripts/grok_r3b_careful.py` keeps the gates unchanged and adds the ridge reader, a label-shuffle null, a paired distractor, and a delay time-shift. Numbers: `runs/grok-r3b-temporal-stress-v1-careful/result.json`.

Order reversal, held-out pairs, three weight seeds. Label-shuffle ridge stays near one half, so these are not fit-to-noise.

| substrate | ridge accuracy | ridge paired flip | head paired flip |
|---|---:|---:|---:|
| decay 0.25 | 1.000, 0.984, 0.949 | 1.000, 0.969, 0.898 | 0.172, 0.180, 0.102 |
| spike-sum | 0.734, 0.824, 0.801 | 0.547, 0.688, 0.625 | 0.438, 0.398, 0.227 |
| five-decay bank | 0.996, 1.000, 0.984 | 0.992, 1.000, 0.969 | 0.992, 1.000, 1.000 |

The decay-0.25 end feature already separates this order. The normalized head does not read that separation. Spike-sum has a weaker linear trace of order, and its head is worse than its own ridge. The bank is the one substrate where the head matches the ridge. Its label-shuffle ridge is 0.512, 0.422, 0.480, so the order score is not a 240-dimensional fit to noise.

Query permutation, seed 94011, 256 held-out rows. First-key policy 0.500.

| substrate | ridge | head | ridge with shuffled train labels |
|---|---:|---:|---:|
| decay 0.25 | 0.574 | 0.500 | 0.520 |
| spike-sum | 0.504 | 0.582 | 0.496 |
| five-decay bank | 0.574 | 0.570 | 0.445 |

No linear code of the queried value is present. This failure belongs to the substrate, not to the optimizer.

Delay, pulse on one channel. Decay ridge is 0.992 on the original time and 0.977 after the same pulse is moved from tick 1 to tick 8. The normalized head stays at 0.512 and 0.473. Spike-sum ridge and head stay at 1.000 in both placements. Spike-sum succeeds here because the channel identity is still in the count, not because it stored the delay.

Paired distractors use one frozen sequence and add pulses to it. The earlier table drew new noise for every distractor count, so those shifts were not an interference effect. On the paired sequences the decay head stays at 0.891, 0.922, 0.898 and the spike-sum head stays at 0.586, 0.570, 0.641.

The single shuffled-label cell at 0.621 crossed the locked 0.60 line. Thirty-two new shuffles of the same split, whose test labels are exactly half ones, have mean 0.485, maximum 0.590, and none reach 0.60. That cell was one draw. It is not evidence that test features entered calibration. The original gate record is left as a fail. The null does not support calling it leakage.

## Answers

1. Spike-sum keeps a partial linear trace of this order (ridge accuracy 0.73–0.82, paired flip 0.55–0.69). That misses the locked head gate. The decay-0.25 end feature keeps the order linearly (ridge paired flip 0.90–1.00) and the normalized head does not read it. The five-decay bank is where the head matches the ridge.
2. Moving a one-channel pulse from tick 1 to tick 8 leaves the decay ridge at 0.977 and the spike-sum ridge at 1.000. The normalized head reads only the spike-sum. Query permutation has no linear code in any of the three substrates (ridge 0.50–0.57).
3. Not on these tasks. Persistent state changed spike-sum `sim_xor` by about one point.
4. Yes. The same-tick ridge and online-head contrast reproduces.
5. No. The same-tick result stands. The careful pass limits where a head score can be called a missing code.
6. Shown: same-tick XOR is in the spike counts and this head can read those counts; on the order task the decay feature is linearly readable while the same head is not; query binding is absent from all three linear substrates; persistence across 12-tick windows does not erase `sim_xor`. The single 0.621 shuffled-label cell is one draw from a null whose maximum over 32 shuffles is 0.590. Not shown: a product change, or a reason to reopen DC-ALIF.
