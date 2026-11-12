# R7 ROUND 37 — REPORT: Tier 3 fails on the corrected benchmark, but 7× smaller than the retracted claim

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no
**Seeds:** 12 fresh seeds (420001 + 6k).

**Decision: Tier 3 fails on a benchmark that passes its validity assertions — but the genuine
scale effect is 3.84pp, not 27.57pp. The retracted result overstated it 7×.**

---

## 1. The benchmark now passes its own checks

Round 36 found `parity3`, `and3`, `maj3` shared byte-identical inputs. The corrected suite gives
each task its own input template (different channels AND ticks), and asserts distinctness
before running:

```
[assert 1] pairwise input distinctness
  all 15 pairs distinct: True
```

Label agreement across pairs is now near chance (0.453–0.545), and label balance is sane
(P(1) 0.121–0.512). **The task that would have caught round 36 costs one line and now runs
first.**

## 2. Result

| order | meanF | worstF | acq | seeds passing |
|---|---:|---:|---:|---:|
| **`xnor_delay…sim_xor`** | **3.84** | **9.12** | 0.9359 | **0/12** |
| `sim_xor…xnor_delay` | 5.03 | 18.33 | 0.9359 | 0/12 |
| `and3…parity3` | 7.10 | 25.08 | 0.9424 | 0/12 |

Compare the retracted ill-posed run: **27.57pp / 58.00pp / 0/12.**

## 3. The decomposition that matters

| quantity | value |
|---|---:|
| T=3 (round 31) | **0.42pp**, 15/15 seeds |
| T=6 corrected | **3.84pp**, 0/12 seeds |
| T=6 ill-posed (retracted) | 27.57pp, 0/12 |
| **genuine scale effect** | **+3.42pp** |
| **ill-posedness inflation** | **+23.73pp (7× the real effect)** |

**So the honest answer to Tier 3 is: the effect shrinks by an order of magnitude but does not
disappear.** Mean forgetting goes 0.42 → 3.84pp and worst goes 0.65 → 9.12pp.

Two of three gates now fail marginally rather than catastrophically:
- **meanF 3.84pp vs a 3pp gate** — 0.84pp over.
- **worstF 9.12pp vs a 5pp gate** — clearly over.
- **acquisition 0.9359** — comfortably inside.

## 4. A validity signal worth noting

Acquisition was **0.8527** on the ill-posed benchmark and **0.9359** on the corrected one.
The head learns *better* when the tasks do not contradict each other. That is the expected
direction and it is independent evidence that the corrected suite is the valid one.

## 5. What this establishes

| claim | status |
|---|---|
| T=3 passes every gate | **stands** (round 31) |
| T=6 fails | **established on a valid benchmark**, for the first time |
| the failure is 3.84pp, not 27.57pp | **established** |
| rounds 32–35's explanations (scale, replay ratio, budget, head form, rank) | **retracted** — measured on the invalid benchmark |
| worst-case degradation at T=6 is the binding problem (9.12 vs 5pp) | **established** |

## 6. What the T=6 failure actually is

The mean is 0.84pp over its gate. The worst case is 4.12pp over its gate. **The problem is
concentrated in one task, not spread across all six** — the same imbalance structure round 25
identified, now with a real benchmark behind it.

That means the remaining gap is plausibly addressable: a mechanism that equalises the worst task
rather than the average. Round 26's tail re-consolidation did exactly that at T=3 (−2.16pp on
worstF); it is in the `replay_tail` arm here already, so the next step is to find what else the
worst task needs at six tasks — most likely more replay for the *earliest* task specifically,
since it now absorbs five subsequent updates rather than two.

## 7. What this does not establish

- **Not** that T=6 is unreachable. The gap is under 1pp on the mean.
- **Not** that the retracted rounds' *methods* were wrong. Their experiments were sound; the
  benchmark was not.
- **Not** the first mission goal. Hidden layer frozen throughout.

## 8. Declared exposures

| risk | exposure |
|---|---|
| **rehabilitating retracted work** | §5 keeps the retraction; only the *number* is replaced, by a 7× smaller one |
| **reporting a near-miss as a win** | §2–3 state three gates and 0/12 seeds; the failure leads |
| **claiming the gap is easy** | §6 says "plausibly addressable" and names what is untested |
| **FP10** | no readiness claim; goal 1 unproven, goal 2 T=3-only |
| **new benchmark validity** | assertions run first and their output is shown verbatim |

## 9. Message for Gemini

The number that shows why input uniqueness is not a formality:

| benchmark | mean forgetting at T=6 | acquisition |
|---|---:|---:|
| three tasks sharing inputs (invalid) | **27.57pp** | 0.8527 |
| six distinct tasks (valid) | **3.84pp** | **0.9359** |

**A 7× overstatement of the effect, in the direction of a dramatic failure.** The invalid
benchmark also produced *lower* acquisition — the head was fighting contradictory labels — which
is the tell I should have read at the time and only read now.

If you report a scaling result, assert input distinctness first and check that acquisition moves
in the right direction with task count. Mine went *down* when tasks overlapped and *up* when they
did not.

## 10. Artifacts

`scripts/r7_r37_t6_fixed.py` (assertions run before the experiment) ·
`artifacts/deepseek-r7/t6-corrected.json` (36 cells) · `report-round36.md` (the retraction this
honours) · `report-round31.md` (the T=3 reference)
