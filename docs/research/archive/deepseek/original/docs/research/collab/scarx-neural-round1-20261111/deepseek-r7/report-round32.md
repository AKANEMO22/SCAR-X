# R7 ROUND 32 — REPORT: TIER 3 FAILS. The result is specific to T=3

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no
**Seeds:** 12 fresh seeds (320001 + 6k), never used before.

**Decision: REJECT the claim that round 31 generalises. At T=6 the gate fails completely:
27.57pp mean forgetting, 0/12 seeds passing.**

---

## 1. Tier 3 asks the question Tier 1 and 2 cannot

Round 31 was strong evidence at T=3: 90/90 cells inside every gate, 3 of 6 orders reaching
15/15 seeds. That is implementation validity (Tier 1) and causal validity (Tier 2).

The brief's third tier is *"does the effect exist outside the optimised conditions?"* — and the
most obvious optimised condition is **how many tasks the head must hold**. Round 29 measured
the co-representation floor and it was 0.36pp at T=3. Nothing established that it stays small at
T=6.

## 2. Result

I added three tasks (`and3`, `maj3`, `xnor_delay`) and ran the identical protocol.

| config | meanF | worstF | acq | seeds passing all gates |
|---|---:|---:|---:|---:|
| **T=3** (round 31, best order) | **0.42pp** | 0.65pp | 0.9037 | **15/15** |
| **T=6** (this round, best order) | **27.57pp** | 58.00pp | 0.8527 | **0/12** |

By order at T=6:

| order (first…last) | meanF | worstF | acq | seeds OK |
|---|---:|---:|---:|---:|
| `xnor_delay…sim_xor` | **27.57** | 58.00 | 0.8527 | **0/12** |
| `sim_xor…xnor_delay` | 28.92 | 59.17 | 0.8547 | 0/12 |
| `and3…parity3` | 29.94 | 60.79 | 0.8753 | 0/12 |

**All three orders fail identically.** This is not an order problem.

## 3. The confound check — and why round 32 survives it

I changed four things at once to fit six tasks (task count 3→6, samples 384→320, epochs
1600→1200, buffer 128→100). Any of the last three could have caused the failure, and reporting
it as a scale result without checking would have been exactly the error I made in rounds 17, 21
and 27.

Decisive test — run **T=3 with the reduced budget**:

| config | meanF | seeds OK |
|---|---:|---:|
| T=3, full budget (384 / 1600 / 128) | 1.38pp | 11/12 |
| **T=3, cut budget (320 / 1200 / 100)** | **0.71pp** | **11/12** |
| T=6, cut budget | **27.57pp** | 0/12 |

**The budget cut costs −0.67pp at T=3 — it makes things slightly better, not worse.**
**The scale effect is +26.86pp.**

**Round 32 stands as a genuine T=6 result.**

## 4. What this means

**Tier 3 fails.** The retention mechanism works when a nonlinear head must hold three tasks and
collapses when it must hold six. The 0.36pp floor at T=3 was not a stable property — it was a
property of T=3.

This is the fourth time in this session that a strong result failed to survive a change of
conditions, and the first time I predicted which change would break it and tested it.

## 5. Why it fails — the most likely cause, stated as a hypothesis

Per-task budget. At T=3 each task got 384 examples and 1600 epochs. At T=6 I gave 320 and 1200
while the *total* data stayed similar, so each task's share of the head's capacity halved twice
over: fewer examples AND more tasks competing for the same 1634 parameters.

The confound check rules out the budget cut *at T=3*, where the head has slack. At T=6 the head
is capacity-bound, and there **the reduced budget may well matter** — the two effects need not
be additive.

**That is not tested and I am not claiming it.** The honest statement: T=6 fails, budget alone
does not explain it at T=3, and whether budget matters at T=6 is open.

## 6. What survives from round 31

| claim | status |
|---|---|
| nonlinear readout removes the T=3 co-representation floor | **stands** (round 29/30/31) |
| interleaved replay removes sequential interference at T=3 | **stands** |
| replay is free in acquisition on a nonlinear head at T=3 | **stands** (+0.008pp) |
| seed failure was an order artefact at T=3 | **stands** |
| **the result generalises to 6 tasks** | **REFUTED** |

## 7. What I am not doing

- **Not** re-running T=6 with a bigger head. That would be the forbidden capacity escalation,
  and it would also be fitting to the failure.
- **Not** reporting the T=3 numbers as if they were the general result. The headline of this
  report is the failure.
- **Not** claiming the mission's second goal is met. It is met at T=3 and fails at T=6.

## 8. The honest state of the programme after thirty-two rounds

| goal | status |
|---|---|
| **1. hidden synapses learn a useful general representation** | **unproven** — 32 rounds, no evidence |
| **2. accumulate skills without severe forgetting** | **met at T=3, FAILS at T=6** |
| credit assignment identified as non-binding | established (round 11) |
| hidden plasticity as non-cause of forgetting | established (round 4) |
| readout overwriting as the cause | established (rounds 4, 16) |
| irreducible floor is readout expressiveness | established (round 29) |

## 9. Declared exposures

| risk | exposure |
|---|---|
| **reporting a failure as a success** | §2–4 lead with the failure; §6 lists only what survives |
| **confound left unchecked** | §3 tests the budget cut and shows it does not explain T=6 |
| **capacity escalation to fix it** | §7 refuses |
| **FP10** | §8 lists goal 1 as unproven and goal 2 as T=3-only |
| **cause stated as confirmed** | §5 labels the capacity explanation a hypothesis and says it is untested |

## 10. Message for Gemini

The test I would apply to any SCAR-X retention claim, mine included:

**Run the identical protocol at double the task count before believing a gate result.**

| | T=3 | T=6 |
|---|---:|---:|
| mean forgetting | **0.42pp** | **27.57pp** |
| worst forgetting | 0.65pp | 58.00pp |
| seeds passing every gate | **15/15** | **0/12** |

And the confound check that made this interpretable: I had also cut per-task budget, so I
re-ran T=3 at the reduced budget — **0.71pp versus 1.38pp, i.e. the budget cut is not the
cause**. Without that check I could not have told a scale failure from a budget failure, and I
have made exactly that mistake three times in this session.

Round 31's numbers are real at T=3. They are not a general result.

## 11. Artifacts

`scripts/r7_r32_tier3_six.py` · `artifacts/deepseek-r7/t6-scale.json` (36 cells, per-task
deltas) · `scripts/r7_r32b_confound.py` · `artifacts/deepseek-r7/budget-confound.json` ·
`report-round31.md` (whose generality this refutes)

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
