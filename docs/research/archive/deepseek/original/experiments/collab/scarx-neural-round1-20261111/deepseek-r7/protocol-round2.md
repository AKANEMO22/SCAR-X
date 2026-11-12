# R7 ROUND 2 — PROTOCOL: a testbed with real headroom

**Committed before any round-2 scored run.**
**Branch:** `research/deepseek/r7-causal-hidden`

## 0. Why round 2 exists

Round 1 established that the inherited suite has **no headroom** for hidden learning:
`frozen + common refit` already reaches 0.8268 (sim_xor), 0.9688 (order), 0.6107
(parity3), and a second-order reader adds **nothing** (−0.042, +0.007, −0.044).

A hidden-learning mechanism cannot be validated on a task frozen already answers.
Round 2 therefore first builds a testbed where hidden plasticity is *required*, verifies
that by measurement, and only then compares mechanisms.

## 1. The testbed

**Task:** `delayed_xor` — two bits separated by 7 silent ticks.
```
x[1, 0] += 1.5·a        # first bit at tick 1
x[8, 1] += 1.5·b        # second bit at tick 8
x[10, 7] += 1.5         # readout gate at tick 10
label = a XOR b
```
**Readout channel:** spike sum over the window (not the production decay).

**Measured headroom (round-1 probe 5 + round-2 step 1, 3 seeds):**

| channel | linear refit | quadratic refit | headroom |
|---|---:|---:|---:|
| decay-0.25 | 0.5365 | — | — |
| **spike sum** | **0.5404** | **0.9948** | **+0.4544** |

The information needed to form `a XOR b` **is present** in the spike sum of the frozen
population, but no linear reader extracts it. Only a representation change — or a
higher-order reader — can. That is exactly the condition a hidden-learning test needs.

**Why this is the right testbed and not a rigged one:** the task is order-2 in the raw
tape and requires holding a value across silent ticks, which is what a recurrent
population is supposed to do. The frozen population already has 99.5% of the information
nonlinearly; the question is whether **hidden plasticity can make it linearly available**,
which is a strictly harder and more meaningful test than the inherited ones.

## 2. Registered prediction (falsifiable, before running)

If the bottleneck were credit assignment (C1), then a mechanism that genuinely shapes
credit should raise the **linear** readout accuracy on this task substantially.

**Prediction:** SECS and CTS both raise `spike-sum linear refit` from ~0.54 toward the
0.99 ceiling.

**Falsifier:** if neither exceeds frozen by ≥ 5pp on the linear refit, then on a task with
45pp of proven headroom the trace mechanisms still cannot move the representation, and
cause C1 is not merely "wrong direction" but "no usable magnitude at all". That would
point the next round at the **hard threshold (C2)** rather than at credit rules.

## 3. Arms (identical population, identical data, identical refit)

| arm | mechanism |
|---|---|
| `frozen` | no hidden update |
| `production` | production rule verbatim: scalar advantage × unshaped trace |
| `secs` | per-neuron credit gating (round-1 mechanism, re-tested on a valid testbed) |
| `cts` | contrastive trace selection (mechanism B, first run) |
| `shuffled` | SECS with credit permuted across neurons |
| `zero` | advantage forced to 0 |

## 4. Mechanism B (CTS) — first definition

```
Keep the last K eligibility traces and the error magnitude at the time each was made.
At update time:
    δ_k = |e_now| − |e_then(k)|            improvement attributable to trace k
    w_k = max(0, δ_k) / (Σ_j max(0, δ_j) + eps)
    elig_effective = Σ_k w_k · trace_k
    Δw = hidden_lr · advantage · elig_effective − anchor·(w − stable)
```
`K = 8`, all streaming, state = `K × 96` floats = 3 KB. No future ticks: `e_now` is known
when the update is applied.

**Falsifier:** if feeding `δ_k` from a **shuffled** error history matches CTS, then the
selection carries no causal information and CTS is rejected.

## 5. Gates for DEV_PROMISING

All of:

1. `spike-sum linear refit` for some arm ≥ **frozen + 5pp**, mean of 3 dev seeds.
2. Paired CI95% over seeds excludes 0.
3. The corresponding **shuffled** control does not reproduce it (gap ≥ 3pp).
4. `zero` control does not reproduce it.
5. **Weight-restore intervention**: restoring an arm's learned hidden into the frozen
   pathway reproduces its refit gain; restoring frozen weights removes it.
6. No regression on the protected `order` task > 3pp.

Failing (1) anywhere on a task with 45pp of headroom is a **strong** negative: it would
mean the available local signals cannot produce a useful representation even when a
useful representation provably exists and is worth +45pp.

## 6. Budget

DEV only. 3 seeds. No confirmation. Peak process-tree commit ≤ 2 GB. Wall clock bounded
per arm; if exceeded, report the completed prefix as `INCOMPLETE_RESOURCE`.

## 7. Declared risks

| risk | guard |
|---|---|
| testbed is rigged (I chose it after seeing headroom) | disclosed in §1; the task generator is fixed and unused before round 1 |
| refit head undertrained (round-1 bug) | refit now z-scales and reports TRAIN accuracy alongside test |
| comparing channels not mechanisms | every arm uses the same spike-sum channel |
| false precision from 3 seeds | paired per-seed values reported, not only means |
