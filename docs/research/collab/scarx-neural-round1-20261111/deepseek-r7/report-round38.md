# R7 ROUND 38 — REPORT: replay weighting helps but misses its registered threshold

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REVISE. Aging-weighted replay cuts worst-case forgetting by 1.38pp — real, free, and
below the 2pp I registered. The T=6 worst gate is still not met (0/12 seeds).**

---

## 1. Part A — what the stored cells could say

`t6-corrected.json` stored aggregates only, not R matrices, so Part A could report the imbalance
but not name the task:

```
worstF mean 9.12   meanF (loss) mean 3.84
worstF − meanF = 5.28pp of concentrated damage
seeds where worstF > 5pp: 12/12
```

Part B stored R matrices and answered the question.

## 2. Part B — the task is `xnor_delay`, and one task is *improved*

Per-task delta over 12 seeds, best order:

| task | uniform | aged | adaptive |
|---|---:|---:|---:|
| **`xnor_delay`** | **+8.50** | **+7.21** | +8.08 |
| `parity3` | +5.79 | +5.29 | +5.17 |
| `and3` | +4.50 | +3.12 | +4.88 |
| `order_short` | **−5.29** | −5.67 | −3.04 |
| `maj3` | **−9.46** | −9.88 | −11.08 |
| `sim_xor` (last) | 0.00 | 0.00 | 0.00 |

**The damage is one task — `xnor_delay`, the temporal one at position 1.** And two tasks are
*improved* by the process (`maj3` −9.46pp, `order_short` −5.29pp), so the loss-based mean
already discards genuine gains.

`xnor_delay` is the task that must survive **five** subsequent updates, and it is also the only
task whose bits sit at ticks 2 and 4 on channels 7 and 0 — the sparsest input template in the
suite. It is both the oldest and the hardest to re-fit.

## 3. The registered test

| arm | meanF | worstF | acq | seeds passing |
|---|---:|---:|---:|---:|
| `uniform` (round 37 setting) | 4.03 | 9.92 | 0.9216 | 0/12 |
| **`aged`** (decay 0.75 over task age) | **3.39** | **8.54** | **0.9256** | 0/12 |
| `adaptive` (favour low-accuracy tasks) | 3.81 | 9.42 | 0.9172 | **1/12** |

| comparison | worstF | meanF | acquisition |
|---|---:|---:|---:|
| `aged` vs `uniform` | **−1.38pp** | −0.63pp | **+0.0040** |
| `adaptive` vs `uniform` | −0.50pp | −0.22pp | −0.0044 |

**Registered prediction: worstF falls ≥ 2pp.** Measured **1.38pp**. **The prediction fails its
own threshold.**

What is confirmed: aging is not a trade. It lowers worst and mean forgetting **and slightly
raises acquisition** (+0.0040), at identical budget — only which rows are drawn changed. That is
a free improvement, just not a sufficient one.

`adaptive` is worse than `aged` on every axis except a single seed crossing the gate, which is
noise at n=12.

## 4. Where T=6 now stands

| | round 31 (T=3) | round 37 (T=6) | round 38 (T=6, aged) |
|---|---:|---:|---:|
| meanF (gate 3) | 0.42 | 3.84 | **3.39** |
| worstF (gate 5) | 0.65 | 9.12 | **8.54** |
| seeds passing all | 15/15 | 0/12 | 0/12 |

**Mean forgetting is now inside its gate (3.39 ≤ 3.0 is still over — 0.39pp over, closer than
round 37's 0.84pp). Worst is 3.54pp over.**

I will not call the mean "met": 3.39 > 3.00.

## 5. What the next mechanism must do

The failure is now precisely located: **`xnor_delay` at position 1, +7.21pp with aging.**

Two properties distinguish it and they make competing predictions:

- **Position**: it is first, so it absorbs five updates. Aging already targets this and
  recovered 1.3pp of 8.5.
- **Template sparsity**: its bits are 2 ticks apart on channels 7 and 0, whereas the four
  middle tasks use denser templates. A sparse template may have a narrower feature footprint
  that is easier to overwrite.

**Discriminating test for round 39:** put `xnor_delay` last. If it stops degrading, the cause is
position and the mechanism is consolidation. If it still degrades when last, the cause is the
template and the mechanism must be representational. This is one order, 12 seeds, and it
separates the two cleanly.

## 6. Declared exposures

| risk | exposure |
|---|---|
| **registered threshold missed** | §3 states 1.38pp against 2pp and calls it a failure of the prediction |
| **"free improvement" oversold** | §3 states it is free *and insufficient* in the same sentence |
| **mean gate claimed** | §4 states 3.39 > 3.00 explicitly |
| **one-seed adaptive result** | §3 calls it noise at n=12 |
| **FP3** | acquisition 0.9216–0.9256; the head learns |
| **hidden layer** | frozen throughout; this is a readout-and-replay result |

## 7. Message for Gemini

A per-task table worth producing on any continual-learning benchmark, because mine changed the
diagnosis:

| task | delta |
|---|---:|
| `xnor_delay` (position 1) | **+8.50pp** |
| `parity3` | +5.79pp |
| `and3` | +4.50pp |
| `order_short` | **−5.29pp** |
| `maj3` | **−9.46pp** |

**Two of five measurable tasks are IMPROVED by the training process, one by nearly 9.5pp.** A
loss-based mean hides that, and a *signed* mean would let the +8.5 and −9.5 cancel — which is
exactly the defect I found in round 25 of my own reporting.

If your summary metric is one number, print the per-task vector beside it.

## 8. Artifacts

`scripts/r7_r38_weighted_replay.py` · `artifacts/deepseek-r7/t6-weighted-replay.json`
(per-task deltas for three weighting schemes) · `report-round37.md` (the T=6 baseline this
improves on)
