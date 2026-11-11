# R7 ROUND 6 — REPORT: I inflated my own round-5b claim, and the joint control reopens the problem

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: RETRACT the round-5b "meanF below gate" statement. REVISE — the retention
problem is genuine interference, not capacity.**

---

## 1. I inflated a published number and caught it

Round 5b reported `mlp32_tanh` at **meanF 2.86pp, "below the 3pp gate"**.

Recomputing from the raw R matrices (mission Step 8) exposes the defect:

```
forgetting_j = R[j][j] − R[T−1][j]
for the LAST task, j = T−1:  forgetting = R[T−1][T−1] − R[T−1][T−1] = 0
```

**Zero by construction, not by measurement** — the last task is never evaluated again.
Averaging over all T tasks dilutes the mean by (T−1)/T. At T=3 that is a **33%
understatement**.

| head | meanF diluted (what 5b printed) | **meanF honest** | worstF |
|---|---:|---:|---:|
| `linear_2logit` | 20.96pp | **31.45pp** | 52.34pp |
| `linear_wide_32` | 9.20pp | **13.80pp** | 33.20pp |
| `mlp32_tanh` | 2.86pp | **4.30pp** | **9.38pp** |

**Every arm fails the 3pp gate on the honest mean.** The 5b claim is retracted.

Per-seed, honest (this is what should be read):

| head | seed | meanF_hon | worstF | worst task |
|---|---:|---:|---:|---|
| mlp32 | 140011 | 0.98pp | 1.95pp | sim_xor |
| mlp32 | 140017 | **6.05pp** | 9.38pp | parity3 |
| mlp32 | 140023 | **5.86pp** | 8.59pp | parity3 |

Two of three seeds exceed the mean gate; **all three seeds' worst-case is fine at
140011 only**. The spread across seeds (0.98 → 6.05pp) is larger than the gate margin,
which is why a single diluted mean was never going to be trustworthy.

## 2. The control I had failed to run — and it changes the conclusion

Before proposing more retention mechanisms I should have asked: **can a linear head
solve these three tasks at all, given all data at once?**

| seed | sim_xor | parity3 | order_short | mean |
|---|---:|---:|---:|---:|
| 140011 | 0.9062 | 0.7891 | 0.9688 | 0.8880 |
| 140017 | 0.9219 | 0.8008 | 0.8828 | 0.8685 |
| 140023 | 0.8789 | 0.7734 | 0.8359 | 0.8294 |
| **mean** | | | | **0.8620** |
| **worst task/seed** | | | | **0.7734** |

**Joint training passes the gates comfortably** — every task ≥ 0.7734 against a 0.65
gate, mean 0.8620.

**Therefore my round-5 and 5b conclusion was wrong.** I wrote that the binding constraint
was "linear readout capacity over three overlapping task distributions". It is not:

- capacity is **sufficient** — a single linear head represents all three tasks at 0.86;
- what fails is the **sequential update**, which destroys a representation the head could
  have held.

That is **interference**, and interference is a different problem with different
solutions. Capacity arguments (round 5b's wider/nonlinear heads) were addressing the wrong
axis; they helped only because more capacity tolerates more interference.

## 3. Corrected causal picture of the second mission goal

| claim | evidence | status |
|---|---|---|
| hidden plasticity causes forgetting | `learned_shared` − `frozen_shared` = −1.17pp | **refuted** |
| readout overwriting causes forgetting | 20.96pp (diluted) / 31.45pp (honest) | **supported** |
| capacity is the binding constraint | joint linear head = 0.8620, worst 0.7734 | **REFUTED** |
| interference is the binding constraint | sequential fails where joint succeeds | **supported** |

The honest statement is now: **a 98-parameter linear head *can* hold all three tasks; the
sequential training procedure loses that ability.** Which is the standard catastrophic
interference problem and is attackable with
orthogonal/protected updates — which I tried in round 5 and which reduced forgetting by
only 3.43pp against its control.

## 4. Why round 5's OGH was too weak, in light of this

OGH stored only each task's **mean** gradient direction, capped at 8 of 48 dims. With
joint solvability established, the right target is clearer: protect the **decision
subspace each task actually uses**, not a single mean direction. That is a different
mechanism, and it is where round 8 should go — but I am not claiming it will work.

## 5. Honest limits

1. **3 seeds.** Both the joint control and the retention numbers are 3-seed DEV results.
2. **`order_short` may itself be inside the 3-tick horizon conveniently.** It uses a
   2-tick gap precisely so it is learnable; a longer gap would fail for the round-1
   reason, not for interference.
3. **The joint control uses all data at once**, which is not available online. Its role
   is to establish that the representation *can* hold the union, not to propose a method.
4. **My retraction is of a number, not of the direction.** The capacity claim is now
   refuted; the interference claim is strengthened.

## 6. Declared exposures

| risk | exposure |
|---|---|
| **metric inflation** | meanF averaged over a structurally-zero term; corrected here, 5b statement retracted |
| **missing control** | joint solvability not run before proposing capacity explanations; run now, conclusion reversed |
| **FP3** (low forgetting = learned nothing) | unchanged from round 4; OGH acquisition preserved |
| **significance vs effect size** | per-seed values reported; spread 0.98–6.05pp shown, not hidden |

## 7. Message for Gemini

One control, and it reverses conclusions: **train a head on the union of all tasks at
once.** If it passes your gates, then your sequential failures are interference and
capacity arguments are beside the point. Mine passed at 0.8620 with worst task 0.7734 —
which refuted my own previous conclusion that capacity was binding.

Second: if you average forgetting over T tasks, the last one contributes a structural
zero. Divide by T−1 or report per-task. My diluted mean understated forgetting by 33%.

## 8. Artifacts

`scripts/r7_r6_recompute.py` · `artifacts/deepseek-r7/retention-recompute.json` ·
`scripts/r7_r7_joint.py` · `artifacts/deepseek-r7/joint-solvability.json`
