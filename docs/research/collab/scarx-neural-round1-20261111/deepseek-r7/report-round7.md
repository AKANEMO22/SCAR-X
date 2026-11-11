# R7 ROUND 7 — REPORT: both protection mechanisms fail, and the shuffled control wins

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REJECT CPS. REJECT OGH-mean on the honest metric. The interference line is
closed by the evidence, not by preference.**

---

## 1. Result — honest metric (T−1 measurable tasks)

3 dev seeds, frozen hidden, spike-sum channel, identical data:

| arm | acq | final | meanF_honest | worstF | gate | per-seed meanF |
|---|---:|---:|---:|---:|---|---|
| `frozen_shared` | 0.7504 | 0.5382 | 31.84pp | 54.69pp | fail | 29.10 / 28.91 / 37.50 |
| `ogh_mean` (round 5) | 0.7539 | 0.5265 | **34.11pp** | 67.19pp | fail | 28.32 / 26.37 / 47.66 |
| **`cps`** (top-r eigen-subspace) | 0.7600 | 0.5299 | **34.51pp** | 49.22pp | fail | 32.03 / 38.67 / 32.81 |
| `cps_shuffled` (control) | 0.7014 | 0.5373 | **24.61pp** | 47.66pp | fail | 23.24 / 28.91 / 21.68 |

**The control beats both mechanisms.** `cps_shuffled` — random orthonormal projection, no
task information whatsoever — has the **lowest** mean forgetting of any arm (24.61pp vs
CPS's 34.51pp and the unprotected baseline's 31.84pp).

## 2. What this means, stated plainly

On the honest metric:

- **OGH-mean (34.11pp) is worse than doing nothing (31.84pp).** Round 5's apparent
  +3.43pp improvement over its control was an artefact of the diluted metric — recomputed
  honestly, the round-5 mechanism is *harmful*.
- **CPS (34.51pp) is worse than doing nothing** and worse than its own shuffled control.
- **Only the shuffled control improves on baseline** (24.61pp), and it does so by
  constraining the update into a random low-dimensional subspace — i.e. **learning less**,
  which is exactly the False-Progress-3 failure mode. Its acquisition is also the lowest
  (0.7014).

**The mechanism I designed does not work. The direction I believed in is not supported.**

## 3. Why the mechanism failed — diagnosed, not hand-waved

Two specific reasons, both visible in the matrices:

**3a. The projection removes directions the NEXT task needs.** Look at seed 140023:

```
frozen_shared R = [[0.992, 0.719, 0.426],
                   [0.660, 0.629, 0.488],
                   [0.445, 0.426, 0.766]]
cps           R = [[0.992, 0.719, 0.426],
                   [0.602, 0.633, 0.488],
                   [0.520, 0.449, 0.879]]
```

CPS raises final `sim_xor` 0.445 → 0.520 (+7.5pp) and `order_short` 0.766 → 0.879, but
drops `parity3` acquisition `R[1][1]` 0.629 → 0.633 — essentially flat, and the diagonal
mean barely moves. The protection redistributes rather than preserves.

**3b. The protected subspace is computed from a gradient that changes sign within a task.**
`gdiff = gw[1] − gw[0]` is the class-discriminating direction at one minibatch. It flips
as the head moves. Summing outer products over minibatches (my covariance) therefore
accumulates a subspace that mixes opposing directions, and the top eigenvectors do not
correspond to "the decision subspace the task depends on". That is a design error in CPS,
not a tuning issue.

## 4. What the round-6 joint control now tells us, combined with this failure

| fact | source |
|---|---|
| a linear head **can** hold all three tasks (0.8620, worst 0.7734) | round 6 |
| sequential training loses that ability (31.84pp forgetting) | round 7 |
| protecting gradient directions does not help (CPS, OGH both ≥ baseline) | round 7 |
| constraining the update does help slightly (shuffled 24.61pp) but costs acquisition | round 7 |

**The representational capacity exists; the protection schemes I can build from local
gradient statistics do not recover it.** That is a real, if negative, finding: the
interference is not of the kind that gradient-subspace projection addresses, at least not
at this dimension and with these per-task estimates.

## 5. Honest conclusion on the mission's second goal

Across five mechanisms now tested on this benchmark:

| mechanism | honest meanF | vs baseline 31.84pp |
|---|---:|---|
| frozen_shared (baseline) | 31.84pp | — |
| OGH-mean | 34.11pp | **worse** |
| CPS top-r subspace | 34.51pp | **worse** |
| shuffled projection | 24.61pp | better, but acquisition 0.7014 and it learns less |
| replay (round 4) | ~0pp but acquisition 0.48 | **FP3** |
| nonlinear head (round 5b) | 4.30pp | best, but a diagnostic and still fails worstF 9.38pp |

**No mechanism passes the gates: acquisition ≥ 0.65, final ≥ 0.65, mean forgetting ≤ 3pp,
worst ≤ 5pp.** The closest, at 4.30pp mean / 9.38pp worst, is a nonlinear head that is a
diagnostic rather than a candidate.

**I am not going to keep inventing protection schemes.** Three have failed in the same
direction, and the pattern (shuffled ≥ structured) says the benefit available in this
family is "perturb the update less", not "protect the right directions".

## 6. Where I would go next, and what it requires

The evidence points away from gradient-subspace protection and toward the one thing the
joint control proves is possible: **the head can hold the union**. Two legitimate options
remain, and both need a decision I will not make alone:

1. **A nonlinear readout as an actual candidate.** Round 5b shows it reaches 4.30pp mean /
   9.38pp worst with acquisition 0.8381. It fails worstF, but it is the only arm in the
   nearest order of magnitude. Using it as a candidate conflicts with "no capacity
   expansion to mask a learning-rule failure" — unless the mission owner judges that a
   nonlinear **readout** is not the neural learning rule and is in scope.
2. **Accept the gate is out of reach in this configuration** and record it as a measured
   bound on what a linear readout over H48 spike-sum features can retain across three
   tasks.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **FP3** (low forgetting because it learned nothing) | shuffled control has the lowest forgetting AND lowest acquisition (0.7014); flagged |
| **FP7** (optimizer change called mechanism) | `hidden_lr`, `trace_decay`, `update_clip` untouched; hidden frozen |
| **my own metric error propagating** | round 5's "+3.43pp over control" was measured on the diluted metric; on the honest metric OGH is **worse than baseline** and that is stated here |
| **selecting a favourable arm** | all four arms reported per seed, including the one that beats my mechanism |

## 8. Message for Gemini

Three numbers that should change your next experiment:

1. **On the honest metric (dividing by T−1, not T), my round-5 result reversed sign.** OGH
   went from "+3.43pp over its control" to **34.11pp versus a 31.84pp baseline — worse than
   doing nothing.** If your forgetting average includes the last task, recompute it.
2. **A random orthonormal projection beat both of my structured protection mechanisms**
   (24.61pp vs 34.51pp and 34.11pp). If your retention gain also appears under a shuffled
   basis control, the mechanism is "perturb less", not "protect the right thing".
3. **A linear head trained jointly on all three tasks reaches 0.8620.** Capacity is not the
   blocker; sequential update is. Any retention mechanism should be compared against that
   ceiling, not against a shared-head baseline.

## 9. Artifacts

`scripts/r7_r7_cps.py` · `runs/20261111T121149Z-r7-r7-cps-v1-721bde/result.json` ·
`report-round6.md` (joint control + honest metric)
