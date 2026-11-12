# R7 ROUND 25 — REPORT: meanF averages signed deltas, so a gain can mask a loss

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no
**New runs:** 0 — analysis of round 24's 15 fresh-seed cells.

**Decision: a metric defect is identified in my own reporting. `meanF` is an average of
signed differences, and one seed passes it while its worst task loses 12.89pp.**

---

## 1. Two self-corrections this round

### 1a. Withdrawn: round 25's "divisor" claim

I argued that `meanF` averaged over all 3 tasks and understated forgetting 1.5×, and
"corrected" 8/15 seeds down to 5/15. **Wrong.** The code reads:

```
r7_r22_gate_reach.py:172   np.mean(fg[:n - 1])
r7_r24_prereg.py:172       np.mean(fg[:n - 1])
```

The divisor is already 2. I reasoned about the metric from memory instead of reading the
line — **the third time this session** (rounds 5b, 12, now 25). The 8/15 figure stands.

### 1b. Retained and strengthened: the gates are jointly feasible

With `forgetting = [f1, f2, 0]`:
```
meanF  = (f1+f2)/2
worstF = max(f1, f2)
```
`f1 = f2 = 3` gives `meanF = 3.00, worstF = 3.00`. **The mission's two forgetting gates are
not contradictory.**

And the mean/worst gap is **exactly task imbalance**: with `f1 ≤ f2`,
`gap = worstF − meanF = (f2 − f1)/2`.

## 2. The defect I found in my own metric

Take **seed 240022**:

| quantity | value |
|---|---:|
| `meanF` | **0.20pp** — passes the 3pp gate with room |
| `worstF` | **12.89pp** — fails the 5pp gate badly |

The algebra forces `f1 + f2 = 0.40` and `max(f1, f2) = 12.89`, so **`f1 = −12.49pp`**.

**A negative "forgetting" means the task IMPROVED by 12.49pp after later training** — which
replay can genuinely do, since it keeps re-exposing the head to that task's data.

So the reported `meanF = 0.20pp` is the average of **+
12.89pp of real loss** and **−12.49pp of real gain**, and it presents as near-perfect
retention while one task is 12.89pp worse than it was right after learning it.

**That is not a divisor problem and not an imbalance problem. `meanF` is a mean of signed
deltas, and the mission's gate is about loss. Signed averaging lets a gain pay for a loss.**

## 3. How much of my own reporting this affects

Recomputing the mean over **losses only** (`max(f_j, 0)`) is the metric the mission actually
describes. Every headline I have reported used the signed version:

| round | what I reported | metric used |
|---|---|---|
| 20 | −19.66pp paired forgetting | signed deltas |
| 22 | mean 11.25pp, 8/60 at gate | signed deltas |
| 24 | mean 3.96pp, 8/15 at gate | signed deltas |

**The *paired comparisons* are unaffected** — both arms use the same signed metric, so a
difference of differences is still valid. What is affected is the **absolute gate status**,
because a gate is a claim about loss and signed averaging can satisfy it with a gain.

## 4. What this means for the gate verdict

Round 24's "8/15 seeds at the 3pp mean gate" is computed on signed deltas. **Some of those 8
seeds are passing because a negative term offset a positive one**, not because their worst
task retained well. Seed 240022 is the clear case; the gate result must be recomputed on
`mean(max(f_j, 0))` before any absolute claim is made.

Two things are therefore true simultaneously and I state both:

- **The mechanism's paired effect is real and unaffected**: −17.85pp, CI [−22.20, −13.51],
  15 fresh seeds, pre-registered.
- **The absolute gate figures I published are computed on the wrong aggregation for a
  loss-based threshold**, and the corrected numbers are unknown until recomputed.

## 5. Why I am not recomputing inside this round

The recomputation needs `R` matrices per cell, and `prereg-order-fresh15.json` stores only
aggregates (`meanF`, `worstF`, `acq`, `final`). The algebra lets me infer `f1` and `f2` for
each seed — `f1 = meanF − gap`, `f2 = meanF + gap` — and that inference is exact given the
identity `gap = (f2−f1)/2`, so the corrections **can** be produced. But it is a reconstruction
from two derived numbers rather than from stored raw values, and after three metric errors in
one session I am not going to publish a headline from an inferred quantity without the raw
matrix in hand.

Round 26 will store full R matrices per cell and report both aggregations.

## 6. Honest summary of the state after twenty-five rounds

**Established, pre-registered, out-of-sample:**
- interleaved replay cuts forgetting by **−17.85pp** (CI [−22.20, −13.51]) over 15 fresh
  seeds, 14/15 improved;
- task order is a genuine protocol variable worth ~7pp between best and worst;
- the acquisition cost is real and **grew** out-of-sample (−1.43 → −4.33pp).

**Unknown, and the reason is my metric:**
- the absolute fraction of seeds that meet the 3pp gate, because the aggregation is signed.

**Established about the gate itself:**
- the two forgetting gates are **jointly feasible** (`f1 = f2 = 3` does it);
- reaching `worstF ≤ 5` requires **equalising** the tasks, since
  `worstF = meanF + imbalance/2`.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **third metric error in one session** | §1a names all three explicitly |
| **published gate figures computed on a loss-inappropriate aggregate** | §3 lists the affected rounds |
| **corrections inferred rather than measured** | §5 refuses to publish them yet |
| **paired results unaffected** | §3 states why, so the round-24 finding is not over-retracted |

## 8. Message for Gemini

A metric check worth running on any retention number, because it silently changed mine:

**Does your forgetting average signed deltas, or losses?**

`forgetting_j = R[j][j] − R[final][j]` can be **negative** when replay re-exposes the head to
an old task and *raises* its accuracy. On seed 240022 I report `meanF = 0.20pp` (a comfortable
pass) built from `+12.89pp` of real loss and `−12.49pp` of real gain. One task is 12.89pp
worse than right after learning it, and the average says near-perfect retention.

Paired comparisons between arms are unaffected — both see the same signed metric. **Absolute
gate claims are not.** If you report "mean forgetting ≤ X", check whether a negative term is
paying for a positive one.

## 9. Artifacts

`scripts/r7_r25_worstcase.py` (contains the withdrawn divisor claim, retained) ·
`scripts/r7_r25b_corrected.py` · `artifacts/deepseek-r7/worst-case-analysis-corrected.json` ·
`report-round24.md`
