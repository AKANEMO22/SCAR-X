# R7 ROUND 29 — REPORT: the floor is HEAD-EXPRESSIVENESS. Round 27's "unreachable" is withdrawn

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no
**Seeds:** same 15 fresh seeds as rounds 24/26/27/28.

**Decision: round 27's "the 3pp gate is unreachable" is WITHDRAWN. The floor is a property of
the 2-logit linear readout, not of SCAR-X's frozen representation. A nonlinear readout reaches
0.36pp and passes the gate on 15/15 seeds.**

---

## 1. The question round 27 should have asked

Round 27 called `solo − joint = 6.41pp` "irreducible" and concluded the gate is unreachable.
Round 28 confirmed it was stable across 8× training effort and three samplers.

But **"irreducible" is meaningless without naming the head.** Round 5b had already measured a
32-unit tanh head at 4.30pp — *below* the 6.23pp linear floor. That should have stopped me from
writing "unreachable". This round measures the floor for three head forms on identical frozen
features, identical seeds, identical budgets.

## 2. Result — the floor is almost entirely the readout form

| head | floor | sd | min | max | seeds ≤3pp | params |
|---|---:|---:|---:|---:|---:|---:|
| **`linear`** (2-logit) | **6.52pp** | 2.03 | 3.26 | 9.51 | **0/15** | 98 |
| `wide16` (wider linear) | 8.79pp | 6.90 | 1.69 | 27.34 | 2/15 | 1 568 |
| **`mlp32`** (tanh layer) | **0.36pp** | 0.44 | 0.00 | 1.69 | **15/15** | 1 634 |

**A nonlinear readout cuts the floor by 6.16pp — from 6.52pp to 0.36pp — and passes the 3pp
gate on every single seed.**

The wider *linear* head does **not** help (8.79pp, worse than the narrow one). So this is not
about parameter count: `wide16` has 1 568 parameters and a *higher* floor, while `mlp32` has
1 634 and a 0.36pp floor. **It requires nonlinearity**, not capacity.

## 3. What this withdraws and what it establishes

| claim | status |
|---|---|
| "the 3pp gate is unreachable" | **WITHDRAWN** — reachable with a nonlinear readout |
| "the floor is irreducible" | **WITHDRAWN** — it is head-specific, not feature-specific |
| floor stable across optimisation effort (round 28) | **stands** — but stable *for the linear head* |
| credit assignment is not the bottleneck | **stands** (round 11) |
| hidden plasticity does not cause forgetting | **stands** (round 4) |
| readout overwriting causes it | **stands** (round 4, 16) |
| interleaved replay removes ~80% of excess | **stands** (rounds 20–24) |

**The corrected causal statement:** forgetting in SCAR-X is caused by readout overwriting, and
the reason a mechanism could never reach the gate is that the readout was a 2-logit linear map
that cannot co-represent three distributions. **Both halves were needed**, and I had the first
half right from round 4 and the second half wrong from round 27.

## 4. Why this does not settle the mission

Three honest limits, stated before anything is claimed.

**4a. `mlp32` is a diagnostic, not a candidate.** The brief forbids expanding capacity to mask a
learning-rule failure. This measurement is legitimate because it **explains where the floor
comes from** — the same role BPTT plays — and it changes no production code, supplies no
production weight, and claims no skill acquisition. **Using it as the learner would be the
forbidden thing**, and that remains a scope decision for the mission owner.

**4b. A low floor is not a retention result.** This round measured `solo − joint` only. It says
a nonlinear head *can* co-represent the three tasks. It says nothing about whether sequential
training with it forgets, which needs the full R-matrix protocol.

**4c. It does not fix the acquisition trade-off.** Round 24 measured −4.33pp acquisition cost
for replay on the linear head. Nothing here addresses that.

## 5. What the next round must test, and what would falsify this

**Registered now, before running:**

Hypothesis: with a nonlinear readout, sequential training + interleaved replay reaches
`mean forgetting ≤ 3pp` and `worst ≤ 5pp` on **every** seed, while acquisition stays ≥ 0.65.

Falsifier: if the nonlinear head's *sequential* forgetting exceeds its own joint floor by more
than ~3pp, then capacity fixes co-representation but not sequential interference, and the
mechanism question reopens at a higher floor.

That is the right test because it separates the two things this round and round 21 each half
answered.

## 6. The scope question, now with decisive evidence

This is the third time I raise it, and the evidence is now unambiguous:

- round 5: "a nonlinear head measured 4.30pp, is it in scope?" — no data either way;
- round 27: "a nonlinear head would move the floor" — the floor was still called irreducible;
- **round 29: the linear floor is 6.52pp, the nonlinear floor is 0.36pp, 15/15 seeds.**

**The mission's 3pp gate is unreachable for any linear readout on this configuration, and
reachable for a nonlinear one.** The brief forbids capacity expansion "to mask a learning-rule
failure". A nonlinear readout is not masking a *learning-rule* failure — the learning rule was
shown irrelevant in round 11 — it is addressing a **representational** limitation of the
readout.

I am not implementing it as a candidate. I am reporting that the decision is now unavoidable
and well-evidenced, and that I will not make it alone.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **two rounds of conclusions withdrawn** | §3 lists them by name |
| **nonlinear head read as a candidate** | §4a states the diagnostic role and the forbidden use |
| **FP2** (ridge better, hidden unchanged) | the hidden is frozen in every arm; this is a readout measurement and is labelled as one |
| **low floor reported as a retention result** | §4b says explicitly it is not |
| **selection** | same 15 seeds, all three heads, no winner picked by seed |
| **`wide16` worse than `linear`** | reported; it is the evidence that capacity alone does not help |

## 8. Message for Gemini

The measurement that reversed my own conclusion, in case it applies to yours:

**Report `solo − joint` for at least two head forms before calling a floor irreducible.**

| head | floor (15 seeds) | seeds ≤3pp | params |
|---|---:|---:|---:|
| 2-logit linear | **6.52pp** | 0/15 | 98 |
| wider linear (16 blocks) | 8.79pp | 2/15 | 1 568 |
| **tanh 32 units** | **0.36pp** | **15/15** | 1 634 |

The wider *linear* head is **worse** than the narrow one, so this is not parameter count — it is
nonlinearity. For two rounds I reported "the gate is unreachable" as a fact about SCAR-X's
representation. It is a fact about a 2-logit linear readout.

If CKCER's mechanism sits on a linear head, its floor may be similarly self-imposed.

## 9. Artifacts

`scripts/r7_r29_head_form.py` · `artifacts/deepseek-r7/floor-by-head.json` (per-seed,
per-head solo/joint/floor) · `report-round27.md` and `report-round28.md` (the withdrawn claim)
· `report-round11.md` (the credit result that makes this a representational, not
learning-rule, issue)
