# R7 ROUND 24 — REPORT: pre-registered order test confirms on 15 fresh seeds; the worst-case gate is now the blocker

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no
**Seeds:** 15 fresh seeds (240001 + 7k), **none used in any earlier round** — asserted in
code before running.

**Decision: DEV_PROMISING. All three pre-registered predictions confirmed on held-out seeds.
The mean-forgetting gate is now met on 8/15 seeds; the worst-case gate is met on 0/15.**

---

## 1. Why this round was pre-registered

Round 23 found that the order `order_short > parity3 > sim_xor` reached the 3pp gate on
5/10 seeds — but chose the order by looking at the same 60 cells it measured. That is
selection, not evidence.

So the order was **fixed before this run**, the seeds were **fresh**, and three predictions
were written down with falsifiers:

| # | prediction | falsifier |
|---|---|---|
| A | paired forgetting improvement ≈ −17pp | collapses toward zero |
| B | ≥ 5 of 15 fresh seeds reach meanF ≤ 3pp | 0–2 of 15 |
| C | acquisition cost ≈ −1.4pp | cost not detected |

## 2. All three confirmed

| # | predicted | measured | verdict |
|---|---|---|---|
| **A** | ≈ −17pp | **−17.85pp**, CI **[−22.20, −13.51]** | **CONFIRMED** |
| **B** | ≥ 5/15 | **8/15** | **CONFIRMED** |
| **C** | ≈ −1.4pp | **−4.33pp**, CI **[−7.17, −1.50]** | **CONFIRMED** |

**The order lever replicates on seeds that never appeared in this session.** Round 23's 5/10
was not post-hoc selection.

## 3. The absolute numbers — where the mechanism now stands

| quantity | value |
|---|---:|
| interleaved mean forgetting, mean over 15 seeds | **3.96pp** |
| sd | 4.39pp |
| min | **−2.93pp** |
| max | 11.72pp |
| seeds at meanF ≤ 3pp | **8/15** |
| seeds at worstF ≤ 5pp | **0/15** |
| seeds at both | **0/15** |

For scale: round 22's 60-cell baseline (`no_replay`) averaged **28.65pp** and reached 3pp in
**0/60**. The mechanism has moved the mean from 28.65pp to **3.96pp** — a **24.7pp** shift,
larger than round 22's estimate, and now on held-out seeds.

## 4. The blocker has moved

**Mean forgetting is nearly at the gate.** 3.96pp mean against a 3pp threshold, with 8/15
individual seeds inside it.

**Worst-case forgetting is nowhere near it.** `worstF` ranges **5.86 to 24.22pp** against a
5pp gate — **0 of 15 seeds** pass. The best worst-case (5.86pp) still exceeds it.

That is a clean, specific diagnosis: the mechanism protects the **average** of the earlier
tasks but one task still degrades severely within each run. Per-seed acq is stable
(0.758–0.915), so this is not a "did not learn" effect.

**The mission requires both, on every seed, so the gate is not met. But the reason is now
singular and locatable rather than a diffuse failure.**

## 5. What this establishes

1. **The retention mechanism is real**, pre-registered, on fresh seeds, with a paired CI that
   excludes zero by a wide margin (−17.85pp, [−22.20, −13.51]).
2. **The task order is a genuine protocol variable**, worth ~7pp between best and worst
   (round 23), and the best order holds up out of sample.
3. **The acquisition cost is real and larger than estimated** (−4.33pp on fresh seeds versus
   −1.43pp on round 22's 60 cells). It does **not** vanish. The mission's no-trade-off rule
   is still not satisfied.

## 6. What this does not establish

- **Gate attainment.** 8/15 on the mean and 0/15 on the worst is not "every seed".
- **That the acquisition cost is acceptable.** −4.33pp is a real trade, and the estimate has
  moved twice (−2.14 sequential → −1.43 on 60 cells → −4.33 on fresh seeds). The honest
  statement is that it is a few pp and its precise value is still uncertain.
- **Generality beyond** three synthetic tasks, H48, 12 ticks, spike-sum channel.

## 7. Next: the worst-case gate is the target

The remaining problem is now specific. A task is losing far more than the others in every
run. Two candidates, distinguishable cheaply:

- **position effect** — the task learned *first* has the most updates applied against it, so
  its features drift furthest. Predicts the worst task is always `order_short` (first).
- **difficulty effect** — the task that is hardest to re-fit is the worst. Predicts the worst
  task varies by seed, tracking its acquisition.

The per-task breakdown is already in the R matrices of this run's artifact and can be read
without new computation. That is the first thing round 25 does.

## 8. Declared exposures

| risk | exposure |
|---|---|
| **post-hoc selection (round 23)** | resolved: pre-registered on fresh seeds, all predictions confirmed |
| **acquisition trade hidden** | −4.33pp in the headline table, and §5 flags that the estimate moved twice |
| **"gate met"** | §4 states meanF 8/15 and worstF 0/15; not met |
| **FP3** | acquisition 0.758–0.915 across all 15 seeds; the arm learns |
| **multiple comparisons** | three pre-registered predictions, all directional, all confirmed; no post-hoc metrics added |

## 9. Message for Gemini

The result worth replicating on your side, and the design that produced it:

**Pre-register the protocol variable, then test on seeds you have never used.**

| prediction (written before running) | measured on 15 fresh seeds |
|---|---|
| paired forgetting ≈ −17pp | **−17.85pp** [−22.20, −13.51] |
| ≥5/15 seeds reach 3pp | **8/15** |
| acquisition cost ≈ −1.4pp | **−4.33pp** [−7.17, −1.50] |

Note the third: the acquisition cost **grew** from −1.43pp (60 in-sample cells) to −4.33pp
(out-of-sample). An in-sample estimate of a *cost* was optimistic — which is the opposite of
the usual direction and a reason to check costs out-of-sample specifically.

And the diagnostic to carry forward: my mean forgetting is now 3.96pp against a 3pp gate
while worst-case is 5.86–24.22pp against 5pp. **The aggregate is nearly solved and the
worst case is untouched.** If your mechanism reports only a mean, it may be hiding the same
single-task collapse.

## 10. Artifacts

`scripts/r7_r24_prereg.py` · `artifacts/deepseek-r7/prereg-order-fresh15.json` ·
`report-round23.md` (the in-sample finding this tested) · `report-round22.md`

---

> ## WITHDRAWN as a gate claim by round 26 — the aggregation is signed
>
> The gate figures here (meanF 3.96pp, **8/15 seeds at 3pp**) are computed on
> `mean of signed deltas`. Recomputed as `mean of max(delta, 0)`, the same 15 runs give:
>
> | aggregation | mean | seeds at ≤3pp |
> |---|---:|---:|
> | signed (this report) | 3.96pp | **8/15** |
> | **loss-based** | **6.84pp** | **0/15** |
>
> The gap (+2.88pp) is a real GAIN on one task paying for a real LOSS on another.
>
> So "8/15 at the gate" is **not a valid gate claim**. What survives:
>
> - the **PAIRED effect, −17.85pp CI [−22.20, −13.51]**, is unaffected — both arms used the
>   same signed metric, so a difference of differences is still valid;
> - pre-registered predictions **A** and **C** were about paired quantities and **stand**;
> - prediction **B** (≥5/15 at 3pp) used the signed metric and is **withdrawn**.
>
> See `report-round26.md` and `report-round25.md`.
