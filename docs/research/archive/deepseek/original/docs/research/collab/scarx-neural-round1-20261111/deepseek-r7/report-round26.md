# R7 ROUND 26 — REPORT: round 25's metric defect confirmed; the loss-based gate is 0/15

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no
**Seeds:** same 15 fresh seeds as round 24, so round 24 comparisons are directly reusable.

**Decision: round 25's defect is confirmed and quantified. The tail re-consolidation
mechanism moves the right variable and is below its registered threshold.**

---

## 1. Round 25 was right: signed averaging inflated the gate result

| aggregation | mean | seeds at ≤3pp |
|---|---:|---:|
| `meanF_signed` (rounds 20/22/24 used this) | **3.96pp** | **8/15** |
| **`meanF_loss`** (`mean of max(Δ,0)`) | **6.84pp** | **0/15** |

**The gap is +2.88pp** — that is the magnitude by which a real *gain* on one task was paying
for a real *loss* on another, and the loss-based gate the mission describes is **met by zero
seeds**.

So every absolute gate figure I published — round 22's "8/60 at 3pp", round 24's "8/15 at
3pp" — is **withdrawn as a gate claim**. The mechanism's **paired effects remain valid**
because both arms used the same signed metric.

This is the fourth metric error in this session and the one that most cleanly explains why my
numbers looked like progress while the worst-case never moved.

## 2. The registered mechanism test

Round 25's algebra said `worstF = meanF + imbalance/2`, so closing `worstF` requires
**equalising** the tasks, and the earliest task absorbs the most subsequent updates. So: a
300-epoch tail pass mixing half task-1 replay with half last-task data.

**Registered prediction:** tail lowers `worstF` by ≥ 2pp.

Paired over 15 seeds:

| quantity | change | 95% CI | cells better |
|---|---:|---|---:|
| `worstF` | **−2.16pp** | **[−4.22, −0.10]** | **13/15** |
| `meanF_loss` | −0.98pp | [−2.34, +0.39] | 12/15 |
| final accuracy | +0.007 | [−0.01, +0.03] | 9/15 |

**−2.16pp clears the 2pp threshold, but the CI's upper bound (−0.10) does not exclude the
threshold's neighbourhood.** My script's verdict said "below the registered threshold" because
it tested `hi < −2.0` rather than `mean < −2.0`. **The direction is confirmed with a CI that
excludes zero; the magnitude is 2.16pp ± 2.06pp, so it is established as positive and not
established as ≥2pp.** I am reporting that distinction rather than picking the reading that
suits the mechanism.

**Final accuracy is unaffected** (+0.007, CI straddling zero) — the tail pass does not cost
accuracy.

## 3. Gate status across all arms, both aggregations

| arm | meanF_signed | n≤3 | **meanF_loss** | **n≤3** | worstF | **n≤5** |
|---|---:|---:|---:|---:|---:|---:|
| `no_replay` | 21.81 | 0 | 23.41 | 0 | 40.70 | 0 |
| `interleaved` | 3.96 | 8 | 6.84 | **0** | 12.40 | 0 |
| **`interleaved_tail`** | **3.52** | 7 | **5.86** | **2** | **10.23** | **2** |

The tail arm is the only one in twenty-six rounds that puts **any** seed inside a loss-based
gate: 2/15 on both `meanF_loss ≤ 3` and `worstF ≤ 5` simultaneously.

Still far from "every seed". The honest summary: **the mechanism is real, the metric is now
correct, and the gate is met by 2 of 15 seeds.**

## 4. What changed and what did not

| claim | status after round 26 |
|---|---|
| signed-vs-loss aggregation inflates results | **CONFIRMED**, +2.88pp |
| round 24's "8/15 at the 3pp gate" | **WITHDRAWN** as a gate claim (signed metric) |
| round 24's paired −17.85pp | **stands** (both arms same metric) |
| tail re-consolidation lowers worstF | **CONFIRMED, direction, 13/15 cells, CI excludes zero** |
| tail magnitude ≥ 2pp | **not established** (2.16pp ± 2.06pp) |
| tail costs final accuracy | **no** (+0.007, CI straddles zero) |
| any loss-based gate met | **2/15 seeds** |

## 5. Why the tail pass works, and its limit

The earliest task's decision region is overwritten by two subsequent training phases. The
tail pass re-exposes the head to it *after* the last task, so its boundary is refreshed with
knowledge of the later tasks present in the same batch. That raises the lowest per-task delta
directly, which is what `worstF` measures.

Its limit is visible in the numbers: `meanF_loss` improved only 0.98pp against `worstF`'s
2.16pp. **The tail pass compresses the spread rather than lifting the floor** — it is treating
the imbalance, exactly as the algebra predicted, and imbalance is only part of the residual.

## 6. What the mission requires versus what exists

| requirement | best (loss-based, the correct reading) |
|---|---|
| acquisition ≥ 0.65 | ✓ 0.83–0.89 |
| final ≥ 0.65 | ✓ 0.79–0.85 |
| mean forgetting ≤ 3pp | **2/15 seeds** |
| worst ≤ 5pp | **2/15 seeds** |
| every seed | **no** |
| no acquisition trade-off | ✗ cost −4.33pp (round 24) |

## 7. Declared exposures

| risk | exposure |
|---|---|
| **rounds 22/24 gate claims** | withdrawn here as gate claims; the paired effects are explicitly preserved |
| **threshold reading that flatters the mechanism** | §2 states 2.16pp vs 2.06pp uncertainty rather than calling it a pass |
| **FP3** | acquisition 0.83–0.89, unchanged by the tail pass |
| **best-of-15 selection** | 2/15 is reported as a count, not the best cell |
| **fourth metric error** | named as such in §1 |

## 8. Message for Gemini

The number to take from this round:

**signed average forgetting 3.96pp vs loss-based 6.84pp on identical runs** — and the gate
status flips from **8/15 seeds passing** to **0/15**.

If you report a mean-forgetting gate, check whether a negative delta on one task is paying for
a positive on another. My signed mean was 42% below the honest figure, and it converted a
complete gate failure into an apparent majority pass.

Second, the mechanism-side result: a 300-epoch **tail re-consolidation** of the earliest task
lowers worst-case forgetting by **2.16pp** (CI [−4.22, −0.10], 13/15 seeds better) at **no
final-accuracy cost** (+0.007, CI straddling zero), and it is the only intervention in my
26 rounds that puts any seed inside a simultaneous `meanF ≤ 3 / worstF ≤ 5` gate (2/15).

## 9. Artifacts

`scripts/r7_r26_rmatrix_full.py` · `artifacts/deepseek-r7/rmatrix-both-aggregations.json`
(full R matrices per cell, both aggregations, all three arms) · `report-round25.md` (the
defect this confirms) · `report-round24.md` (whose gate claim is withdrawn)
