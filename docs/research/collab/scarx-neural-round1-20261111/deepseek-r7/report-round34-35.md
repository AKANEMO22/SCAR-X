# R7 ROUND 34–35 — REPORT: at T=6 the limit is the FEATURES, not the readout

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: round 32's scale-limit conclusion STANDS, and this time it survives the confound
test that destroyed round 27's equivalent claim.**

---

## 1. Why this test was mandatory

Round 27 called a 6.41pp floor "irreducible". Round 29 showed that label was an artefact of the
**linear head** — a nonlinear head's floor was 0.36pp — and the claim was withdrawn.

Round 32/33 then reported a T=6 failure at 27.57pp and called it "a genuine scale limit". That
label was not allowed to stand without the same test: **is the T=6 floor a property of this head,
or of the frozen features at six tasks?**

## 2. Round 34 — head capacity does not help

T=6, 8 seeds, budget-matched solo/joint decomposition, three heads:

| head | floor | sd | solo | joint | params |
|---|---:|---:|---:|---:|---:|
| `mlp32` | **31.33pp** | 1.36 | 0.9699 | 0.6582 | 1 634 |
| `mlp128` (4× wider) | **30.47pp** | 2.29 | 0.9670 | 0.6635 | 6 530 |
| `mlp32_deep` (2 layers) | **31.64pp** | 1.81 | 0.9689 | 0.6535 | 2 690 |

**A 4× wider head moves the floor by 0.86pp. A deeper head makes it 0.31pp worse.**

Compare T=3, where the same head change was enormous:

| head change | T=3 effect | T=6 effect |
|---|---:|---:|
| linear → mlp32 | **−6.16pp** | — |
| mlp32 → mlp128 | — | **−0.86pp** |

**At T=3 capacity was the binding constraint. At T=6 it is not.** Every head pays ~31pp.

The `solo` vs `joint` columns say it directly: one task alone is learned to **0.97**, all six
together to **0.66**, for every head form. The features carry each task individually and cannot
carry six simultaneously.

## 3. Round 35 — the rank explanation does NOT survive

I tested the natural connection to round 12's effective-rank measurement (4.60/48). The result
is negative and I report it as such:

| | effective rank | joint ceiling | floor |
|---|---:|---:|---:|
| T=3 union | **4.59**/48 | **0.9587** | 4.13pp |
| T=6 union | **5.58**/48 | **0.6582** | 34.18pp |

**Rank rises +1.00 while the joint ceiling falls −0.30.** Rank does not explain T=6.

And the pooled correlation (−0.50) is itself an artefact I had to catch before writing it up:
it mixes two clusters (all T=3 points are low-rank/high-joint, all T=6 points are
higher-rank/low-joint), so it measures the group difference. **Within each group the
correlations are +0.77 (T=3) and +0.58 (T=6) — both positive**, i.e. higher rank goes with a
*higher* ceiling.

**So effective rank is a within-configuration predictor and useless between configurations.**
That is the same class of error as round 12 and round 25, caught this time before publication.

## 4. What the T=6 wall actually is, in measurable terms

```
solo (one task)     0.9699
joint (all six)     0.6582
six heads tried, three forms, budget matched, 8 seeds
```

**A 31pp gap that does not move with capacity, budget, replay ratio, or sampler.** Round 33
ruled out the replay ratio and the epoch budget; round 34 rules out head expressiveness.

The remaining explanation, and it is the one the data support directly: **the frozen 48-neuron
spike-sum features at H48 do not span enough distinct states to hold six 2-class decision
regions.** Round 12 measured the population at ~4.6 effective dimensions; six interacting
boundaries need more than that.

This is a statement about **the representation**, and it is exactly the mission's first goal
territory: *hidden synapses have not proven they can learn a useful general representation.*
At T=6 the frozen representation demonstrably cannot.

## 5. The three-tier verdict, final for this line

| tier | verdict | evidence |
|---|---|---|
| **1. Implementation** | **PASSED** | parity checks, instrument validation, no-op/frozen controls |
| **2. Causal (at T=3)** | **PASSED** | 90 paired cells, pre-registration, all three predictions confirmed |
| **3. Generalisation** | **FAILED** | T=6: 27.57pp, 0/12 seeds; limit is the frozen features, not the mechanism |

## 6. What this means for the two mission goals

| goal | status after 35 rounds |
|---|---|
| **1. hidden synapses learn a useful general representation** | **unproven — and now the identified T=6 blocker** |
| **2. accumulate skills without severe forgetting** | **T=3 yes; T=6 no** |

The two goals converge: the retention failure at six tasks is a **representation** failure, and
the representation failure is the first mission goal. That is the most coherent statement this
programme has produced, and it is not a solution.

## 7. What I will not do

- **Not** add neurons or widen the hidden layer. Forbidden, and round 34 shows widening the
  *readout* does not help — the constraint is the frozen features.
- **Not** claim the T=3 result generalises. It does not.
- **Not** claim SCAR-X cannot learn. T=3 works with a nonlinear readout; the T=6 limit is
  measured for this configuration (H48, spike-sum, 320 examples/task), not for spiking networks.

## 8. Declared exposures

| risk | exposure |
|---|---|
| **repeating round 27's "irreducible" mistake** | §1 states the test was mandatory; §2 applies it and the claim survives |
| **rank correlation reported as causal** | §3 catches the cluster artefact before publishing and states rank is between-configuration useless |
| **capacity escalation** | §7 refuses |
| **T=3 success presented as general** | §5–6 put the Tier-3 failure in the same table |
| **FP3** | solo 0.97, joint 0.66 — the heads learn; nothing is "low because nothing was learned" |

## 9. Message for Gemini

The test that makes a "floor" claim usable, applied twice with opposite results:

**Vary head expressiveness before calling any forgetting floor irreducible.**

| configuration | linear | mlp32 | mlp128 | verdict |
|---|---:|---:|---:|---|
| T=3 | 6.52pp | **0.36pp** | — | capacity-limited |
| T=6 | — | 31.33pp | **30.47pp** | feature-limited |

The same intervention that collapsed the T=3 floor by 6.16pp moves the T=6 floor by 0.86pp.
**Without both measurements I would have been wrong in one direction or the other.**

And the number I would ask of any continual-learning result at a single task count: **what does
`solo` vs `joint` look like at double the tasks?** Mine goes 0.97 → 0.66 at T=6, for every head
form, which is what a representation limit looks like rather than a mechanism limit.

## 10. Artifacts

`scripts/r7_r34_t6_head.py` · `artifacts/deepseek-r7/t6-floor-by-head.json` ·
`scripts/r7_r35_rank_t6.py` · `artifacts/deepseek-r7/t6-rank-connection.json` ·
`report-round33.md` (replay-ratio and budget ruled out) · `report-round29.md` (the same test at
T=3)

---

> ## RETRACTED by round 36 — the T=6 benchmark was mathematically impossible
>
> `parity3`, `and3` and `maj3` have **identical input tensors** (`max|diff| = 0.000e+00`) and
> **different label functions** (P(1) = 0.54 / 0.15 / 0.47; pairwise label agreement 0.61,
> **0.29**, 0.68). All three write three bits to `x[i,t,0:3]` for `t ∈ {1,2,3}` plus the same
> gate marker, so the same feature vector is asked for contradictory labels.
>
> **No head can be correct on all three.** The measured joint ceiling of 0.66 is the best a head
> can do when three of its six targets conflict on the same evidence — not a capacity, budget,
> replay-ratio, or representation limit.
>
> Everything here about T=6 is withdrawn. **Tier 3 is not tested** — neither failed nor passed.
>
> What still stands from this programme at T=3: credit assignment is non-binding (round 11);
> hidden plasticity does not cause forgetting (round 4); readout overwriting does (rounds 4, 16);
> interleaved replay removes sequential interference (rounds 20–24, 31); a nonlinear readout is
> decisive (round 29, floor 6.52 → 0.36pp).
>
> See `report-round36.md`.
