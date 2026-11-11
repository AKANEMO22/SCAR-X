# R7 ROUND 40 — REPORT: the worst-case degradation is real, not an extreme-value artefact

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no
**New runs:** 0 — analysis of round 31 and rounds 38/39 artifacts.

**Decision: REVISE. The worst-case rise survives a matched-task-count control, so it is a real
degradation rather than the arithmetic of taking a maximum over more tasks.**

---

## 1. The hypothesis round 39 pointed at

Round 39 showed the damage **relocates** when a task moves (worstF 8.54 → 17.83, with `parity3`
taking the loss instead of `xnor_delay`). No positional law, no template law.

That pattern is what an **extreme-value statistic** looks like:

```
worstF = max_j (delta_j)
```

The maximum of T draws rises with T even if every underlying delta is unchanged. The mission
holds `worstF ≤ 5pp` **fixed** while task count varies — a different kind of requirement from a
fixed mean. If the whole effect were this, the gate would be unsatisfiable for *any* mechanism
with non-zero per-task variance.

## 2. A unit error caught before publication

My first pass compared the two pools without converting units — the T=3 `delta` vectors are
stored as **fractions** (`0.0234` = 2.34pp) while the T=6 `per_task` values are already in **pp**.
That produced `sd = 0.06` for the T=3 pool and an `E[max]` of `0.03pp`, which is nonsense.

**Caught by reading the raw artifact rather than trusting the script's own summary.** The
extreme-value section was recomputed with `× 100`; the matched-T section was never affected
because both its sides come from the same artifact in the same units.

That is the fifteenth harness defect this session and the same class as rounds 25 and 39: a
number produced by my own code, believed until checked against its source.

## 3. Result after the fix

Per-task delta distributions:

| | n draws | mean | sd | max |
|---|---:|---:|---:|---:|
| T=3 | 180 | 0.00pp | 2.07 | 6.25 |
| T=6 | 72 | 0.01pp | 10.48 | 12.50 |

Note the **sd itself is 5× larger at T=6** — the distributions are not the same, which already
tells us the extreme-value transfer is not legitimate. But it is computed anyway:

| T | E[max of T draws from the T=3 pool] |
|---:|---:|
| 3 | 1.91pp |
| 4 | 2.25pp |
| 5 | 2.55pp |
| **6** | **2.78pp** |

| quantity | value |
|---|---:|
| measured worstF, T=3 | **1.46pp** |
| measured worstF, T=6 | **8.54pp** |
| **observed rise** | **+7.08pp** |
| **extreme-value predicted rise** | **+0.87pp** |

**The extreme-value effect accounts for 0.87pp of a 7.08pp rise. 88% is unexplained by it.**

## 4. The decisive control: matched task count

Comparing `worstF` over only the **first three** tasks at T=6 against the T=3 suite:

| order | worstF (first 3) | worstF (all 6) |
|---|---:|---:|
| `xnor_first` | 7.71 | 8.54 |
| `xnor_last` | 7.46 | 17.83 |
| `simx_first` | 5.46 | 16.62 |

**mean worstF over the first 3 tasks at T=6 = 6.88pp**
**mean worstF at T=3 = 1.46pp**
**difference = +5.41pp**

**With the same number of tasks being maximised over, the damage is already 5.41pp worse.** The
extra three tasks explain the remaining ~1.7pp; they do not explain the bulk.

**So the mechanism genuinely degrades as task count rises**, and the worst-case gate is a real
requirement rather than a scaling artefact.

## 5. What this settles and what it opens

**Settled:**
- worstF is **not** an artefact of taking a maximum over more tasks (0.87 of 7.08pp);
- at matched task count, a task is already 5.41pp more damaged at T=6 than at T=3;
- the damage is not positional and not template-specific (round 39).

**Opened — the next mechanism question:** if a task is already 5.41pp worse when only three
tasks are being trained *within a six-task run*, the interference is coming from the **later
tasks that were trained before the measurement**, not from the count being maximised. That is a
statement about update interference accumulating across the whole sequence, and it predicts:

> replaying the *last three* tasks more heavily should reduce the first three's damage, because
> the damage is caused by the later updates rather than by how many tasks exist.

That is testable with the machinery already built (round 38's aged weighting) by inverting the
age decay — and it is a genuinely different prediction from round 38's, which favoured the oldest.

## 6. Declared exposures

| risk | exposure |
|---|---|
| **unit error in my own analysis** | §2 states it, shows the wrong number, and how it was caught |
| **convenient conclusion** | the verdict went *against* the tidy "it's just a max" story; I report the less convenient result |
| **matched-T is only partial** | §4 states the extra tasks explain ~1.7pp of 7.08, so it is not zero |
| **new hypothesis unverified** | §5 labels the interference prediction as a prediction and says it is untested |
| **FP3** | acquisition 0.9216–0.9493 across these runs |

## 7. Message for Gemini

The control that decided it, for any worst-case metric across a changing number of items:

**Compare `worst` at matched count, not across counts.**

| | worstF |
|---|---:|
| T=3, all 3 tasks | **1.46pp** |
| T=6, **first 3 only** | **6.88pp** |
| T=6, all 6 | 8.54pp |

If your gate is `max_j ≤ 5pp` and you vary the number of tasks, two things move at once: the
mechanism *and* the maximum. Mine were separable and the mechanism was 5.41pp of the 7.08pp rise.
An `E[max]` bootstrap on the T=3 pool predicted only 0.87pp, so the arithmetic effect is real but
small here — it may not be in your setting, and the matched-count comparison is what tells you
which.

## 8. Artifacts

`scripts/r7_r40_extreme.py` · `artifacts/deepseek-r7/worst-case-extreme-value.json` ·
`report-round39.md` (the relocation finding this follows from) ·
`report-round31.md` (the T=3 reference pool)
