# R7 ROUND 27 — REPORT: the 3pp gate is unreachable — the irreducible floor is 6.41pp

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no
**Seeds:** 15 fresh seeds, same set as rounds 24/26 so results compose.

**Decision: the mission's 3pp mean-forgetting gate cannot be met by any readout-side
mechanism on this suite. This is a measured bound on the protocol, and it is the direct answer
to the central question.**

---

## 1. The decomposition that decides it

`forgetting_j = solo_j − final_j` always factors into two parts:

```
forgetting_j = (solo_j − joint_j)  +  (joint_j − final_j)
             =   IRREDUCIBLE       +      EXCESS
```

- **irreducible**: the price of co-representing all three tasks in one head — paid even by a
  perfect continual learner.
- **excess**: what sequential updating actually destroys — the part a mechanism can remove.

Rounds 6 and 18 measured the joint ceiling but **never compared it to solo accuracy on the
same seeds.** This round does, at a budget-matched 4800 epochs for joint (3 × the per-task
1600).

**Identity verified in code:** `max|total − (irreducible + excess)| = 0.00e+00`.

## 2. Result

| component | mean | loss-based mean | max | min |
|---|---:|---:|---:|---:|
| **irreducible** (solo − joint) | **+6.41pp** | **6.41pp** | +13.28 | +0.00 |
| excess (joint − final) | +3.19pp | 4.33pp | +23.44 | −8.59 |
| total (solo − final) | +9.60pp | 9.60pp | +33.98 | +1.17 |

**The irreducible floor alone is 6.41pp against a 3pp gate.**

Even a mechanism that achieved **exactly** the joint solution — zero excess — would report
**6.41pp** and fail. The gate is not a mechanism problem.

Note the loss-based means do not add (6.41 + 4.33 ≠ 9.60) because `max(0,·)` is not additive.
The identity holds on the signed values, which is what the 0.00e+00 check verifies.

## 3. Why the floor exists

Per-task, the pattern is consistent: `solo` is always higher than `joint`, by 0.78pp to
13.28pp. Representative rows:

| seed | task | solo | joint | final | irred | excess | total |
|---|---|---:|---:|---:|---:|---:|---:|
| 240078 | order_short | 0.9219 | 0.8711 | 0.8203 | +5.08 | +5.08 | +10.16 |
| 240078 | parity3 | 0.7969 | 0.7188 | 0.6719 | +7.81 | +4.69 | +12.50 |
| 240078 | sim_xor | 0.9727 | 0.9258 | 0.9531 | +4.69 | **−2.73** | +1.95 |
| 240092 | parity3 | 0.8438 | 0.7227 | 0.6797 | **+12.11** | +4.30 | +16.41 |
| 240092 | sim_xor | 0.9727 | 0.9258 | 0.9453 | +4.69 | −1.95 | +2.73 |

Two facts stand out.

**3a. A head that fits everything fits each thing less well.** The union forces one linear
boundary set to serve three distributions; every task pays 3–12pp for the shared parameters.
That is a property of a 2-logit linear head over 48 features, not of the update rule.

**3b. Negative excess is common.** For `sim_xor` — the easiest task, frozen decodability 0.97
— the final accuracy is often *higher* than joint (`−2.73pp` excess). Replay keeps improving
it past what a joint head achieves. That is the same signed-average defect round 25 found,
appearing again per task: **`sim_xor` is not forgotten, it is improved.**

## 4. What this settles

| question | answer |
|---|---|
| Is the 3pp gate reachable by a better readout mechanism? | **No.** Floor is 6.41pp. |
| Is interleaved replay near a limit? | **Yes** — it removes the excess; the floor is what remains. |
| Is the bottleneck credit assignment? | No (round 11: +0.945 alignment, zero gain). |
| Is it hidden plasticity? | No (round 4: −1.17pp). |
| Is it readout capacity? | Not capacity — **co-representation cost**. |

**The mission's central question — "is SCAR-X learning poorly because of a missing learning
signal, bad feature geometry, gradient noise, plasticity limits, or the memory mechanism" — has
an answer: none of those is binding for retention. The binding constraint is that one linear
2-logit head cannot co-represent three overlapping distributions without paying 6.41pp.**

## 5. The protocol implication, stated as the brief requires

The brief says: *"Nếu ngưỡng bất hợp lý vì một lỗi định nghĩa benchmark, hãy trình chứng cứ và
đề xuất thay đổi trước khi có thí nghiệm mới. Không sửa âm thầm."*

Evidence:
- the 3pp gate is below the measured irreducible floor of 6.41pp;
- the floor is a property of **this configuration** (T=3 tasks, 48 features, a 2-logit linear
  head, 384 examples per task);
- it is not a learning-rule failure, since a perfect sequential learner would still pay it.

**I am not changing the threshold.** I am reporting that under this protocol no mechanism can
pass it, and that the meaningful question is whether the benchmark intends "forgetting" to
include the co-representation cost. Two defensible resolutions exist and both are decisions
above my authority:

1. **Measure excess, not total.** `joint_j − final_j` isolates what the learner controls. On
   this data that is 4.33pp loss-based — still above 3pp, but 2.1× closer and clearly a
   mechanism quantity.
2. **Compare against the measured floor** rather than zero, i.e. gate on
   `total − irreducible`.

## 6. What the mechanism work achieved, measured against the floor

| arm | total (loss-based) | excess over joint |
|---|---:|---:|
| `no_replay` | 23.41pp | ~17pp |
| `interleaved` | 6.84pp | 4.33pp |
| `interleaved_tail` | 5.86pp | ~3.4pp |

**Interleaved+tail has removed roughly 80% of the excess.** The remaining 3.4pp is what is
left of a mechanism-controllable quantity, and it is close to the noise floor round 19
measured (sd 5.36pp).

## 7. Honest limits

- **15 seeds, three synthetic tasks, H48, 12 ticks, spike-sum channel.** The floor is a
  property of this configuration; a different T, feature count, or head form would move it.
- **A nonlinear head would likely lower the floor** — round 5b measured 4.30pp mean / 9.38pp
  worst with one. That is a **capacity** change and the brief forbids it for candidates, which
  is exactly the scope question I raised in rounds 5 and 26 and still will not decide alone.
- **Solo uses the same 1600-epoch budget as each sequential task**, so the comparison is
  budget-fair, but a solo head could reach higher accuracy with more epochs.

## 8. Declared exposures

| risk | exposure |
|---|---|
| **using the floor to excuse a failing gate** | §5 does not ask to lower it; it presents the arithmetic and two resolutions that need the owner's decision |
| **FP3** | acquisition 0.83–0.89 throughout; the arms learn |
| **floor is configuration-specific** | §7 states it and names the one change (nonlinear head) that would move it |
| **best-of-15 selection** | nothing selected; aggregates over all 15 seeds |
| **negative excess** | §3b reports that one task is *improved*, not forgotten — the same defect as round 25, disclosed again |

## 9. Message for Gemini

The measurement I would ask for before any retention claim, because it reframes the whole
question:

**What is `solo − joint` on your task, and how does it compare to your gate?**

On my suite — 15 seeds, budget-matched joint training:

| component | loss-based mean |
|---|---:|
| irreducible (solo − joint) | **6.41pp** |
| excess (joint − final) | 4.33pp |
| total (solo − final) | 9.60pp |

My gate is 3pp. **The floor alone is 2.1× the gate**, so no mechanism can pass it, and my
interleaved+tail arm has already removed ~80% of the controllable part.

If CKCER's task has a similarly large irreducible component, its reported "forgetting" partly
measures co-representation cost rather than forgetting, and the comparison should be against
`joint`, not against the solo accuracy it can never regain.

## 10. Artifacts

`scripts/r7_r27_floor.py` · `artifacts/deepseek-r7/irreducible-floor.json` (full per-seed,
per-task solo/joint/final with the verified identity) · `report-round26.md` ·
`report-round19.md` (the noise floor that brackets the excess)

---

> ## WITHDRAWN by round 29 — the floor is a property of the READOUT, not of SCAR-X
>
> This report concludes the 3pp gate is unreachable because `solo − joint = 6.41pp` is
> "irreducible". **That word did the work of a claim I had not tested.**
>
> Round 29 measured the same decomposition for three head forms on identical frozen features,
> identical seeds, identical budgets:
>
> | head | floor | seeds at ≤3pp | params |
> |---|---:|---:|---:|
> | 2-logit linear | **6.52pp** | **0/15** | 98 |
> | wider linear (16 blocks) | 8.79pp | 2/15 | 1 568 |
> | **tanh 32 units** | **0.36pp** | **15/15** | 1 634 |
>
> **The gate is reachable.** The floor is expressiveness of the 2-logit linear map, not the
> frozen representation. The wider *linear* head has 16× the parameters and a *higher* floor,
> so this is nonlinearity, not capacity.
>
> What survives here: the arithmetic (a 2-logit linear head genuinely cannot co-represent the
> three tasks under 6.4pp), and round 28's finding that the **linear** floor is stable across
> 8× optimisation effort. What does not survive: "unreachable", and "irreducible".
>
> See `report-round29.md`.
