# R7 ROUND 28 — REPORT: the irreducible floor survives its strongest confound check

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no
**Seeds:** same 15 fresh seeds as rounds 24/26/27.

**Decision: round 27's unreachability claim STANDS. The floor is stable, not an artefact of
under-trained joint training.**

---

## 1. The confound I had to rule out

Round 27 concluded the 3pp gate is unreachable because `solo − joint = 6.41pp`, and called it
irreducible. That conclusion was only valid if my **joint head was at its optimum**. Round 27's
joint run used 4800 epochs of randomly-sampled half-batches of the union — three obvious ways
that could be under-trained, each of which would inflate the "floor":

| hypothesis | prediction if true |
|---|---|
| **H1 undertrained** | more joint epochs → lower floor |
| **H2 sampling** | a full-batch or balanced sampler → lower floor |
| **H3 genuine** | floor stable across budgets and samplers |

## 2. Sweep: 4 budgets × 3 samplers = 12 joint configurations

| budget | half | full | balanced |
|---:|---:|---:|---:|
| 4 800 | 6.43 | 6.28 | 6.28 |
| 9 600 | 6.40 | 6.28 | 6.37 |
| 19 200 | 6.42 | 6.28 | 6.35 |
| 38 400 | 6.28 | 6.28 | **6.23** |

**An 8× increase in joint training effort moves the floor by 0.20pp.** Changing the sampler
from random-half to full-batch moves it 0.15pp. The floor sits at **6.23–6.43pp** across every
configuration.

**H1 and H2 are ruled out. H3 holds.**

## 3. What this means

The gap between a solo head and a joint head on these 15 seeds is **not** an optimisation
artefact. It is the price of forcing one 2-logit linear boundary set to serve three
distributions — and it does not shrink when you give the optimizer 8× more steps, a full
batch, or balanced task sampling.

So round 27's arithmetic stands:

```
mean-forgetting gate       3.00pp
irreducible floor          6.23pp  (best configuration)
```

**The gate is 2.1× below the floor.** A perfect sequential learner, achieving exactly the joint
solution with zero excess, would still report 6.23pp.

## 4. The per-seed spread, which matters for the mission's "every seed" rule

Per-seed floors span **3.39pp to 8.33pp**:

| seed | floor@4800 half | best over sweep |
|---|---:|---:|
| 240099 | 3.91 | **3.39** |
| 240064 | 4.56 | 3.65 |
| 240050 | 4.82 | 4.69 |
| 240078 | 5.34 | 4.56 |
| 240085 | 5.47 | 5.08 |
| 240092 | 7.42 | 7.16 |
| 240057 | 8.20 | 8.20 |
| 240071 | 8.33 | 7.68 |

**The best single seed (3.39pp) is barely above the 3pp gate, and it is the minimum of 15
draws.** Nine of fifteen seeds have their floor above 4pp regardless of optimisation effort.
Under the mission's "every seed" requirement the gate is not merely hard — it is below the
floor for two thirds of the seeds before any mechanism runs.

## 5. Where this leaves the whole programme

**Established, replicated, out-of-sample:**

| finding | evidence | round |
|---|---|---|
| credit assignment is not the bottleneck | exact trace, +0.945 alignment, zero gain | 10–11 |
| hidden plasticity does not cause forgetting | −1.17pp | 4 |
| readout overwriting causes it | +31.90pp baseline | 4, 16 |
| interleaved replay removes ~80% of the excess | −17.85pp paired, 15 fresh seeds | 20–24 |
| task order is a real protocol variable | ~7pp between best and worst | 23–24 |
| **irreducible floor is 6.23pp and stable** | 12 joint configurations, 15 seeds | 27–28 |

**Not achieved:**

| requirement | status |
|---|---|
| mean forgetting ≤ 3pp on every seed | **impossible** — below the floor |
| worst ≤ 5pp on every seed | impossible for the same reason |
| no acquisition trade-off | ✗ −4.33pp |

## 6. The proposal, as the brief requires

The brief: *"Nếu ngưỡng bất hợp lý vì một lỗi định nghĩa benchmark, hãy trình chứng cứ và đề
xuất thay đổi trước khi có thí nghiệm mới. Không sửa âm thầm."*

**Evidence now includes a confound check**: the floor is 6.23pp at the best of 12 joint
configurations, does not move with 8× budget, and 15/15 seeds fail it.

**Two resolutions, both needing the owner's decision, neither of which I have implemented:**

1. **Gate on excess, not total.** `joint_j − final_j` is the quantity a learner controls.
   Measured: **4.33pp** loss-based for interleaved, **~3.4pp** for interleaved+tail. Still
   above 3pp but a mechanism quantity, and no longer floored by co-representation.
2. **Gate on `total − floor_measured_per_seed`.** This makes the threshold relative to what
   the configuration can express.

I have not changed the threshold, not re-run on a friendlier benchmark, and not selected the
best seed. The gate as written cannot be met by anything operating through the readout.

## 7. What would legitimately change the floor

One measured option, already in the record: **a nonlinear head.** Round 5b measured a 32-unit
tanh head at **4.30pp mean / 9.38pp worst forgetting**, acquisition 0.8381 — better than the
6.23pp linear floor. That is a **capacity** change and the brief forbids it for candidates.

This is the same scope question I raised in rounds 5, 26 and 27. It is now backed by a
measured floor that is stable against optimisation effort, so the question is sharper:
**is a nonlinear readout in scope?** If yes, round 5b's 4.30pp is the starting point and is
still above the gate; if no, the gate is closed for this configuration.

## 8. Declared exposures

| risk | exposure |
|---|---|
| **claiming irrefutability** | §1–2 name the three confounds I tested and the exact sweep; the claim is conditional on them |
| **using the floor to excuse failure** | §6 does not ask to lower the gate; it presents the arithmetic and two owner decisions |
| **best-of-15 seed selection** | §4 reports 3.39pp as the minimum of 15 and states nine seeds exceed 4pp |
| **configuration-specific bound** | §7 names the one change that would move it and refuses it without authorisation |
| **FP3** | acquisition 0.83–0.89 throughout |

## 9. Message for Gemini

A confound check worth copying before anyone declares a bound:

**Does your "irreducible" component survive an 8× optimisation sweep and a sampler change?**

Mine: floor 6.23–6.43pp across budgets 4 800 → 38 400 and samplers {random-half, full-batch,
balanced}. **A 0.20pp move for an 8× budget increase.**
That is what makes it a property of the representation and not of my optimiser.

And the number that should reframe any SCAR-X retention claim: **solo − joint = 6.23pp on a
2-logit linear head over 48 features, three tasks.** If CKCER reports forgetting below ~6pp on
a similar configuration, either its head is not linear, or its "forgetting" is measured against
something other than solo accuracy.

## 10. Artifacts

`scripts/r7_r28_floor_check.py` · `artifacts/deepseek-r7/floor-stability.json` (12 joint
configurations × 15 seeds, per-seed breakdown) · `report-round27.md` (the claim this confirms)
· `report-round5b` within `report-round5.md` (the nonlinear-head measurement)
