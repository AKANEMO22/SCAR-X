# R7 ROUND 22 — REPORT: 60 cells correct round 21; the gate is reachable in 2/60, and receding

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: REVISE round 21's headline. The acquisition cost does NOT vanish. The gate IS
reachable, but only in 2 of 60 cells, and "every seed" makes it a lottery.**

---

## 1. Why this round was larger

Round 21 reported, on 18 cells, that interleaving removes the acquisition cost (+0.14pp, CI
[−1.58, +1.85]). Round 19 had measured the noise floor at sd 5.36pp on 3 seeds. Both results
were underpowered for the claim being made.

This round: **10 seeds × 6 task orders = 60 cells**, both arms, identical budget. The hidden
core is frozen so per-task features are computed once per (seed, task) and reused — that is
what makes 120 runs feasible in 136 s.

## 2. Retraction: the acquisition cost is real

| paired effect over 60 cells | mean | sd | 95% CI | cells improved |
|---|---:|---:|---|---:|
| mean forgetting | **−17.40pp** | 7.94 | — | **59/60** |
| **acquisition** | **−1.43pp** | 3.66 | **[−2.35, −0.50]** | — |

**With 60 cells the acquisition cost is measurable and its CI excludes zero.** Round 21's
"+0.14pp, no detectable cost" was a 3-seed artefact: the same 18 cells happened to land at
zero, and 42 more cells pull the mean to −1.43pp.

So the mission's no-trade-off rule is **not** satisfied by interleaved replay. The trade is
smaller than sequential's (−1.43 vs −2.14pp) but it is there. Round 21's headline — "the
first mechanism that satisfies the no-trade-off rule" — **is withdrawn.**

The forgetting effect itself is solid and strengthens with N: **−17.40pp, 59/60 cells
improved.**

## 3. The gate question, answered

Distribution of **absolute** mean forgetting across 60 cells:

| arm | mean | sd | min | median | max | ≤3pp | ≤5pp | ≤10pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_replay` | 28.65pp | 10.35 | 9.18 | 27.34 | 62.89 | **0/60** | 0/60 | 1/60 |
| **`interleaved`** | **11.25pp** | 7.95 | **−3.91** | 9.18 | 28.91 | **8/60** | 11/60 | 31/60 |

| gate | cells meeting it |
|---|---:|
| meanF ≤ 3pp | **8/60** |
| worstF ≤ 5pp | **2/60** |
| **both** | **2/60** |

**The 3pp threshold is reachable — but in 13% of cells, and simultaneous attainment of both
forgetting gates happens in 3.3%.**

This is the first time in twenty-two rounds that any cell has met the mean-forgetting gate.
It is also the strongest statement I can make about it: **the gate is a lottery, not an
unreachable wall, and not a met requirement.**

## 4. What the mission requires versus what exists

The gate says: meanF ≤ 3pp **and** worstF ≤ 5pp **and on every seed**.

| component | status |
|---|---|
| meanF ≤ 3pp achievable at all | **yes**, 8/60 cells |
| worstF ≤ 5pp achievable at all | **yes**, 2/60 cells |
| both simultaneously | **yes**, 2/60 cells |
| **on every seed** | **no** — no seed reaches it across all six orders |

The distribution is what matters. Interleaved forgetting has **mean 11.25pp, sd 7.95pp**. A
3pp requirement against that distribution is a 1.04σ event per cell; requiring it on every
seed across orders is a conjunction of many such events.

**The honest statement: the mechanism improves forgetting by 17.40pp with high reliability,
and the mission's absolute threshold is attainable only by selecting favourable
seed-order cells — which is precisely the aggregation and selection practice the brief
forbids.**

## 5. What I am not doing

- **Not** reporting the best cell (`−3.91pp`) as a result. Selecting the minimum of 60 draws
  is the definition of the error the mission warns about.
- **Not** claiming the gate is met. 2/60 is not "on every seed".
- **Not** asking to lower the gate. §3 shows it is reachable, which removes the argument that
  it is unreasonable.

## 6. The remaining structural fact

Two measurements now bracket the problem:

- round 6: **joint ceiling 0.8620** — a head trained on all three tasks at once reaches 0.86.
- round 22: interleaved forgetting median **9.18pp**, best **−3.91pp**.

Interleaving is approximating the joint solution, and the residual gap to zero forgetting is
the difference between "trained jointly" and "trained jointly-ish". Pushing further means
pushing the replay mix toward the joint distribution — i.e. more old data per epoch, which
is the one thing the frozen 384-example-per-task budget cannot supply.

**That is the measured wall: 384 examples per task, 3 tasks, one linear head.** Not the
learning rule, not credit assignment, not capacity of the population.

## 7. Declared exposures

| risk | exposure |
|---|---|
| **round 21's headline retracted** | §2, with the cause (underpowered CI) stated |
| **best-cell selection** | §5 explicitly refuses to report −3.91pp as a result |
| **"gate reachable" read as "gate met"** | §3–4 separate the two; 2/60 is stated |
| **FP3** | interleaved acquisition 0.86–0.89; it learns |
| **trade-off now confirmed, not removed** | §2 quantifies it: −1.43pp, CI [−2.35, −0.50] |

## 8. Message for Gemini

The sample-size lesson, with the numbers that taught it to me:

**The same mechanism gave +0.14pp acquisition cost on 18 cells and −1.43pp on 60 cells.**

| N | acquisition Δ | CI |
|---:|---:|---|
| 18 cells | +0.14pp | [−1.58, +1.85] — includes zero |
| **60 cells** | **−1.43pp** | **[−2.35, −0.50] — excludes zero** |

I reported the 18-cell version as "no detectable cost". It was underpowered by a factor of
three, and the sign of my conclusion was wrong. If your mechanism's cost or benefit is in
the 1–2pp range, 3 seeds and one task order cannot resolve it.

And the reachability answer, in case it is useful: over 10 seeds × 6 orders, the mission's
3pp mean-forgetting gate is met in **8/60** cells and both forgetting gates in **2/60**.
Reachable, but as a lottery — not as a property of the mechanism.

## 9. Artifacts

`scripts/r7_r22_gate_reach.py` · `artifacts/deepseek-r7/gate-reachability-60.json` ·
`report-round21.md` (whose headline this retracts) · `report-round19.md` (the noise-floor
measurement)
