# R7 ROUND 19 — REPORT: the mission's 3pp forgetting gate is not measurable with 3 seeds

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: the gate cannot be resolved by this protocol. That is a protocol finding, and
it outranks any mechanism result in this session.**

---

## 1. Design

3 seeds × all 6 task orders = **18 cells**, identical mechanism (replay, buffer 128,
weight 1), identical data, identical gates. This separates order-driven variance from
seed-driven variance, and it answers whether a 3pp gate is resolvable at all.

## 2. The headline number

```
pooled mean 7.71pp    pooled sd 5.36pp    min −0.59    max 18.75
gate: 3.00pp
```

**The pooled standard deviation (5.36pp) is 1.8× the gate (3.00pp).**

A 3pp gate can only be resolved if the observed mean sits below
`3.00 − 2×5.36 = −7.72pp`. **A negative mean forgetting is not attainable**, so under this
protocol a 3pp gate cannot be confirmed by any result — including a genuinely perfect one.

This is not a failure of a mechanism. It is a statement that **the measurement's noise
floor is above the threshold the mission set**, and it applies to every retention claim made
on three seeds, mine and anyone else's.

## 3. Where the variance lives

| source | variance of means |
|---|---:|
| across seeds | 16.47 |
| across orders | 10.00 |
| **mean within-seed (across orders)** | **19.40** |

**Within-seed variance is the largest of the three.** One seed, six orders:

| seed | mean | range | sd |
|---|---:|---|---:|
| 140011 | 3.16pp | [−0.59, 7.62] | 3.10 |
| 140017 | 9.05pp | [1.95, **18.75**] | 5.74 |
| 140023 | 10.94pp | [5.08, 16.80] | 3.95 |

**Seed 140023 — the "stationary outlier" of round 18 — reaches 5.08pp under one order and
16.80pp under another.** It was never stationary. Round 18 measured it under a single order,
`sim_xor > parity3 > order_short`, and I built a whole diagnostic round on the assumption
that its 11.52pp was a property of its initialisation.

**That assumption was wrong.** The 11.52pp was largely an order effect.

Order means span 4.23pp to 11.85pp:

| order | meanF |
|---|---:|
| order_short > parity3 > sim_xor | **4.23pp** |
| parity3 > sim_xor > order_short | 5.34pp |
| parity3 > order_short > sim_xor | 5.99pp |
| sim_xor > parity3 > order_short | 7.68pp |
| order_short > sim_xor > parity3 | 11.20pp |
| sim_xor > order_short > parity3 | 11.85pp |

Putting `sim_xor` first costs ~3.5pp on the mean versus putting it last. That is a task-order
effect of the same size as the gate.

## 4. What this retracts

| claim | round | status |
|---|---|---|
| "seed 140023 is stationary at 11.52pp across every buffer and weight" | 17 | **RETRACTED** — it varies 5.08–16.80pp across orders |
| "the worst seed's representation cannot hold the union (H1)" | 18 | **VOID** — the premise that the seed was an outlier does not hold |
| "spread is seed-driven" | 18 (framing) | corrected: within-seed order variance (19.40) is the largest term |

Round 18's own caution saved it from being reported as a finding: I wrote that the automatic
H1 verdict was not trustworthy because the margin was 0.0086 over an arbitrary threshold.
**It was right to distrust it, for a different reason than I gave.**

## 5. What survives, and it is worth stating carefully

**Replay still works.** Round 17's comparison was against a no-replay baseline measured on
the same three seeds and the same single order, so the −24.22pp improvement
(31.90 → 7.68pp) is a paired comparison and is not affected by cross-order variance. What
this round changes is **what a 7.68pp means** — it is one draw from a distribution with
sd 5.36pp, not a point estimate of the mechanism's quality.

**The two accuracy gates do pass robustly.** Acquisition and final sit at 0.86–0.87 across
all 18 cells against a 0.65 gate, which is far outside the noise.

**The forgetting gates cannot be assessed with this protocol**, at any mechanism quality.

## 6. What this means for the mission

The gate is `mean forgetting ≤ 3pp, worst ≤ 5pp, every seed`. With this harness:

| requirement | measurable? |
|---|---|
| acquisition ≥ 0.65 | **yes** — observed 0.78–0.89, well clear |
| final ≥ 0.65 | **yes** |
| mean forgetting ≤ 3pp | **no** — noise floor 5.36pp exceeds the threshold |
| worst ≤ 5pp | **no** — worst ranges 7–21pp within a single seed across orders |

The honest recommendation, and I am not going to implement it unilaterally because it
changes a registered protocol:

**Either the gate needs more seeds and more orders so that its standard error drops below
3pp, or the gate needs to be compared against the measured noise floor rather than against
zero.** With 18 cells the standard error of the pooled mean is `5.36/√18 = 1.26pp`, which
*would* resolve a 3pp threshold — the problem is that I have 3 seeds, and the mission's
requirement is "on every seed".

The structural conflict is: **"every seed" plus a 3pp threshold plus within-seed order
variance of sd 3–6pp are mutually unsatisfiable.** The mission should see that arithmetic
before another mechanism is built against it.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **retracting two rounds of framing** | §4 lists both, including my own round-18 diagnosis |
| **using this to excuse a failing gate** | §6 does not ask to lower the gate; it asks the owner to reconcile three mutually inconsistent requirements |
| **order variance could be seed-specific** | 18 cells, all three seeds, all six orders, reported in full |
| **replay result could be read as weak** | §5 states it is a paired comparison and stands; only its interpretation as a point estimate changes |

## 8. Message for Gemini

A number I would ask for before accepting any retention claim, mine included:

**What is your within-seed variance across task orders?**

Mine, 18 cells, identical mechanism: **pooled sd 5.36pp**, against a 3pp gate. Within one
seed, across six orders: **range −0.59 to 7.62pp (sd 3.10)**, **1.95 to 18.75pp (sd 5.74)**,
**5.08 to 16.80pp (sd 3.95)**.

If yours is similar, then a single-order, three-seed retention result **cannot** distinguish
a 3pp mechanism from a 9pp one, and any PASS at that sample size is a draw. The specific
trap I fell into: I identified "seed 140023" as a stationary outlier across two rounds and
built a diagnostic on it. Under a different task order the same seed reports **5.08pp**.

## 9. Artifacts

`scripts/r7_r19_order_var.py` · `artifacts/deepseek-r7/order-variance.json` ·
`report-round18.md` (whose seed-outlier framing this retracts) · `report-round17.md`
