# R7 ROUND 36 — RETRACTION: my T=6 benchmark was mathematically impossible

**Branch:** `research/deepseek/r7-causal-hidden`
**Production `src/scarx/neural/` modified:** no

**Decision: rounds 32, 33, 34 and 35 are RETRACTED. The T=6 tasks `parity3`, `and3` and `maj3`
have IDENTICAL INPUTS with DIFFERENT LABELS. No head can be correct on all three.**

---

## 1. The defect

All three generators place three bits at `x[i, t, 0:3]` for `t ∈ {1,2,3}` and the same gate
marker at `x[i, 10, 7]`. They differ only in the label function:

| task | label | P(1) |
|---|---|---:|
| `parity3` | a XOR b XOR c | 0.5400 |
| `and3` | a AND b AND c | 0.1500 |
| `maj3` | majority | 0.4700 |

**Verified identical:** `max|diff| = 0.000e+00` on the input tensors for all three pairs.

Label agreement between them: `parity3`↔`and3` 0.61, `parity3`↔`maj3` **0.29**, `and3`↔`maj3`
0.68.

**A single head with no task cue receives the same feature vector for the same underlying input
and is asked for contradictory labels.** It is mathematically impossible to satisfy all three.
The measured ceiling (joint 0.66) is not a representation limit — it is the best a head can do
when three of its six targets conflict on the same evidence.

## 2. How round 36 found it

I was looking for the mechanism behind the T=6 wall and computed pairwise **centroid distance**
between task feature distributions:

| pair | centroid distance | normalised separation |
|---|---:|---:|
| `parity3`↔`and3` | **0.0000** | **0.0000** |
| `parity3`↔`maj3` | **0.0000** | **0.0000** |
| `and3`↔`maj3` | **0.0000** | **0.0000** |
| `sim_xor`↔`xnor_delay` | 0.6866 | 1.13 |
| `sim_xor`↔`parity3` | 2.2504 | 3.54 |
| `order_short`↔`maj3` | 6.7933 | 12.10 |

**Exactly zero**, not "small". That is not something a population does by accident — it means
the inputs are byte-identical. I checked directly rather than assuming the number was noise.

## 3. What is retracted

| round | claim | status |
|---|---|---|
| **32** | "TIER 3 FAILS. The retention result is T=3-specific" | **RETRACTED** |
| **33** | "the T=6 failure is a genuine scale limit; replay-ratio and budget refuted" | **RETRACTED** |
| **34** | "at T=6 the limit is the FEATURES, not the readout" | **RETRACTED** |
| **35** | "the effective-rank explanation does not survive" | **RETRACTED** |

All four rest on the same ill-posed benchmark. **Tier 3 is not tested at all** — not failed, not
passed. I have no evidence either way about generality beyond T=3.

## 4. What round 35 got right by accident

Round 35's internal check was correct on its own terms — the pooled correlation was indeed a
cluster artefact and rank is indeed useless between configurations. **That methodological point
stands.** But its headline ("rank does not survive") was computed on the ill-posed benchmark and
tells us nothing about six *distinct* tasks.

## 5. Why this happened, stated plainly

The T=3 suite had three genuinely different tasks: `sim_xor` (same-tick 2-bit), `parity3`
(same-tick 3-bit), `order_short` (temporal). When I extended to six I wanted same-tick algebraic
variants and reused the same input template without checking that the **inputs** were distinct.

`R32.GENS[tk](seed, n)` uses the same RNG seed, draws the same bits in the same order, and
writes them to the same coordinates. **Only the label assignment differs.** I never wrote a
uniqueness check.

**This is the sixth harness defect of the session and the worst one**: the previous five produced
wrong numbers on valid experiments. This one produced four rounds of conclusions on an experiment
that cannot have a correct answer.

## 6. What survives

| finding | status |
|---|---|
| credit assignment is not the bottleneck | **stands** (round 11, exact trace) |
| hidden plasticity does not cause forgetting | **stands** (round 4) |
| readout overwriting causes it | **stands** (rounds 4, 16) |
| interleaved replay removes sequential interference at T=3 | **stands** (rounds 20–24, 31) |
| a nonlinear readout is decisive at T=3 (floor 6.52 → 0.36pp) | **stands** (round 29) |
| task order matters at T=3 (0.42–1.63pp) | **stands** (rounds 23–24, 31) |
| replay's acquisition cost is head-dependent | **stands** (rounds 22, 31) |
| **anything about T=6** | **NO EVIDENCE** |

## 7. What the next round must do first

**Build six tasks with pairwise-distinct input templates and a uniqueness assertion**, then
re-run the Tier-3 test. The check is trivial and I should have had it from the start:

```
for a, b in combinations(TASKS, 2):
    assert not np.array_equal(inputs[a], inputs[b]), f"{a} and {b} share inputs"
```

I am adding that assertion to the generator module before any further run.

## 8. Declared exposures

| risk | exposure |
|---|---|
| **four rounds of claims withdrawn at once** | §3 lists them by round number |
| **the defect was in my own generator** | §5 explains how it arose and what was missing |
| **"Tier 3 not tested" read as "Tier 3 passed"** | §6 states there is no evidence either way |
| **previous rounds' numbers** | §6 enumerates what still stands and why each is unaffected |
| **FP10** | no claim of readiness anywhere; one goal remains unproven and the other is T=3-only |

## 9. Message for Gemini

The check that would have saved me four rounds, and it costs one line:

**Assert that your tasks have distinct inputs before comparing anything across them.**

```python
for a, b in itertools.combinations(TASKS, 2):
    assert not np.array_equal(inputs[a], inputs[b])
```

My three "new" tasks placed identical bit patterns at identical coordinates and differed only in
the label function. Mean pairwise **centroid distance between their feature distributions was
exactly 0.0000** — I found it while hunting a representation limit, not while checking the
benchmark. The 0.66 joint ceiling was not a capacity, budget, or rank effect; it was three
contradictory targets on the same evidence.

I spent rounds 32–35 attributing it to scale, then replay ratio, then budget, then head form,
then effective rank. **Every one of those was a valid experiment on an invalid benchmark.**
Distinct inputs are a weaker requirement than a valid label function, and I did not check even
that.

## 10. Artifacts

`local/check_task_validity.py` (the direct check) ·
`artifacts/deepseek-r7/t6-feature-diagnosis.json` (the zero-distance finding) ·
`report-round32.md`, `report-round33.md`, `report-round34-35.md` (all annotated as retracted)
