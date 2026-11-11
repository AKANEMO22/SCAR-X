# R7 ROUND 16 — REPORT: replay was rejected on my own label bug; corrected, it cuts forgetting 3.4×

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REVISE. Round 4's replay rejection was an artefact of a bug in my code.
Corrected replay is the best retention result of this session — and still fails the gate.**

---

## 1. The bug

Re-reading `r7_r4b_rmatrix.py` (line 170–177):

```python
buf.append(trx[i].copy())                                    # sequence, NO label
...
RF = readout_feats(core, np.stack(buf))
Ry = np.array([try_[k % N_TR] for k in range(len(buf))], np.int32)   # <-- CURRENT task's labels
head.train(RF, Ry)
```

`try_` is the **current** task's label array. So stored sequences from **previous** tasks
were trained against labels drawn from the **current** task, applied positionally. The
replay arm was learning near-random label assignments.

Round 4 therefore reported:

```
frozen_replay  acq 0.4818  meanF -0.04pp      ->  I declared FP3
learned_replay acq 0.4688  meanF  0.26pp
```

and I wrote that replay "bought retention by not learning". **That was my bug, not
replay's behaviour.**

## 2. Corrected replay

Buffer now stores `(sequence, label)` pairs together.

| arm | acq | final | meanF | worstF | gate |
|---|---:|---:|---:|---:|---|
| `frozen_shared` | 0.8819 | 0.6693 | **31.90pp** | 54.30pp | fail |
| **`frozen_replay`** | **0.8607** | **0.7982** | **9.38pp** | 18.75pp | fail |
| `learned_replay` | 0.8594 | 0.7908 | 10.29pp | 24.22pp | fail |
| `frozen_pertask` (instrument) | 0.9123 | 0.9123 | **0.00pp** | 0.00pp | check ✓ |

**Acquisition is preserved (0.8607 vs 0.8819, −2.1pp, not the 0.48 collapse) and mean
forgetting falls from 31.90pp to 9.38pp — a 3.4× reduction.** Final accuracy rises from
0.6693 to 0.7982.

The instrument check passes: `frozen_pertask` reads exactly **0.00pp**.

## 3. What this changes in the record

| round-4 claim | status |
|---|---|
| "replay flattens forgetting to ~0pp but acquisition collapses to 0.48" | **RETRACTED** — bug |
| "replay buys retention by not learning (FP3)" | **RETRACTED** — the decline was mislabelled data |
| "no retention mechanism works" | **REVISED** — replay works better than any mechanism I built |

This is the **fifth** harness defect in this session, and the most consequential: it caused
me to reject the one technique that actually helps.

## 4. What still does not pass

The mission's gates are acquisition ≥ 0.65, final ≥ 0.65, **mean forgetting ≤ 3pp**,
**worst ≤ 5pp**, on every seed.

| requirement | best achieved | gap |
|---|---:|---|
| acquisition ≥ 0.65 | 0.8607 ✓ | met |
| final ≥ 0.65 | 0.7982 ✓ | met |
| mean forgetting ≤ 3pp | **9.38pp** | **3.1×** over |
| worst forgetting ≤ 5pp | **18.75pp** | **3.75×** over |

So replay clears the two accuracy gates and misses the two forgetting gates by ~3–4×.
That is a far better position than "31.90pp and nothing works", and it is honest progress
that I could only see after finding my own error.

## 5. Why replay helps, mechanistically

The round-4 decomposition already showed the cause: **hidden plasticity contributes
−1.17pp; readout overwriting contributes +31.90pp.** Replay attacks exactly that —
it keeps the head exposed to previously-seen `(feature, label)` pairs, so the head's
decision boundaries for old tasks are not overwritten.

With labels correct, the mechanism does what the decomposition predicted. That is the first
time in sixteen rounds that a prediction from my own causal analysis was confirmed by a
mechanism I then built.

## 6. Why it still fails, and what is left

9.38pp of forgetting remains with replay. The buffer holds 128 sequences per seen task
(384 total at the end) against 384 fresh examples for the current task — a 50/50 mix. Two
untested levers:

1. **Retention budget** — a larger buffer would raise the old/new ratio. This is a
   *capacity* knob (memory, not neurons), bounded by the 2 GB process-tree limit which is
   nowhere near binding for 384×48 floats (74 KB).
2. **Replay weighting** — training the head on replayed examples more than once per task.

Both are within the constraints: no extra neurons, no new topology, no BPTT, no future
data. A replay buffer is explicitly listed in the brief's section X as a permitted
direction ("Adaptive replay"), and it is *not* the FP9 case ("replay buffer answers instead
of the neuron") because here the buffer trains the readout that the neurons feed — the
neurons still do the work.

I am registering this for the next round rather than claiming it will close a 3.1× gap.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **FP3 was declared on a bug** | retracted in §1 and §3 |
| **replay could be read as FP9** | §6 explains the distinction: the buffer trains the readout, the neurons compute the features |
| **buffer size is a capacity knob** | §6 states this plainly; it is memory, not neurons, and it is on the brief's permitted list |
| **gates still failed** | headline says "still fails the gate"; no threshold relaxed |

## 8. Message for Gemini

A defect pattern worth checking in any replay implementation:

**Does your buffer store the label with the sequence?**

Mine did not. It stored sequences and then applied the *current* task's labels
positionally, so replayed examples were trained on random labels. That produced
`acq 0.4818, meanF −0.04pp`, which I misread as "replay destroys learning". With labels
stored correctly:

| | acq | meanF |
|---|---:|---:|
| buggy | 0.4818 | −0.04pp |
| **correct** | **0.8607** | **9.38pp** |

The clickbait version of my own result ("replay gives near-zero forgetting") would have
been exactly right for the wrong reason. The corrected version is 9.38pp — real, 3.4×
better than baseline, and still above a 3pp gate.

## 9. Artifacts

`scripts/r7_r16_replay_fixed.py` · `artifacts/deepseek-r7/replay-corrected.json` ·
`scripts/r7_r4b_rmatrix.py` (the buggy original, retained) · `report-round4.md`
