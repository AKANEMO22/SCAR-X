# REPORT — Round 6D — harness audit and causal revalidation

**Branch:** `research/deepseek/r6d-audit`
**Audited commit:** R6A `ceb98e33b40e8b47c6788b7d483e8b52a97d3351`
**Protocol:** `protocol-v1.md`, committed before every run it governs.
**Environment:** DSH Web, Windows 11 · `deepseek-ai/deepseek-v4.1-flash-1017`

**Verdict: PASS on the data effect after correction; FAIL on R6A's stress conclusion,
which was a measurement artefact and is withdrawn.**

---

## 1. The final question, answered

> *After controlling normalization and matched updates fairly, does increasing the
> number of independent episodes still improve `both_A`? How far does that generalise,
> and what did hidden plasticity contribute?*

**Yes, it still improves, and the effect is not a normalisation artefact.**

| scaler | n=1 024 | n=16 384 | gain |
|---|---:|---:|---:|
| **R5B** (floor .05, clip ±5) | 0.6873 | **0.9178** | **+23.05pp [+17.16, +29.48]** |
| R6A (floor 1e-6, no clip) | 0.7138 | 0.8942 | +18.05pp [+12.98, +22.71] |

8 fresh seeds (`130001…130037`), never used in R5B, R6A or here.

**Matched updates, unique vs repetition, same 16 384 updates:**

| scaler | repeated-256 | unique-16384 | gap |
|---|---:|---:|---:|
| R5B | — | — | **+15.08pp [+9.68, +22.23]**, worst seed **+5.74pp** |
| R6A | — | — | +19.84pp |

Under the *corrected* R5B scaler the unique-minus-repeated gap is **positive on every
seed** (worst +5.74pp), so the diversity claim survives.

**Hidden plasticity contributed nothing: it was never run.** The hidden stayed frozen
in all 2×2 cells. This round measures readout learning on a fixed representation and
says so; it is **not** evidence that hidden neurons learned anything.

## 2. What the audit found

Six defects the brief named — **all six real**. Three brief-suggested defects — **none
reproduced**. Full evidence in `audit.md`.

| id | defect | severity | status |
|---|---|---|---|
| A1 | R6A normalizer differs from R5B (floor 1e-6 vs .05; no clip vs ±5) | CRITICAL | confirmed |
| A2 | stress features passed **unscaled** to a scaled head | CRITICAL | confirmed |
| A3 | "fixed calibration" was **not** fixed | CRITICAL | confirmed |
| A4 | G2 computed on `direct`, not `both_A` | MAJOR | confirmed |
| A5 | G4 and G5 hard-coded `True` | MAJOR | confirmed |
| A6 | G3 and G7 absent from the gate set | MAJOR | confirmed |
| N1 | query tick wrong in stress | — | **did not reproduce** |
| N2 | seed reuse dev/confirmation | — | **did not reproduce** |
| N3 | oracle or label error | — | **did not reproduce** |

### A3 is the one I want to be judged on

R6A's own script contained the string
`"calibration": "fixed 256-episode training prefix, identical for all budgets"`.
**That was false**, and R6A's own test had already proved budgets are not nested
(measured overlap 21/256). I documented the non-nesting in the R6A report and still did
not notice it invalidated my calibration claim. Measured drift of the calibration mean
between arms: **0.0891 / 0.1141 / 0.1699** at n = 1 024 / 4 096 / 16 384.

## 3. Does the correction change the conclusion?

**The direction survives; the confound does not reverse it.**

| effect | both_A | CI95 |
|---|---:|---|
| data, under R5B (clean) | **+23.05pp** | [+17.16, +29.48] |
| data, under R6A (as published) | +18.05pp | [+12.98, +22.71] |
| preprocessing at n=1 024 | +2.65pp | [−1.37, +7.11] |
| preprocessing at n=16 384 | −2.36pp | [−4.85, +0.25] |
| **interaction** | **−5.01pp** | — |

The preprocessing main effect is **within noise at both budgets** and the interaction is
−5.01pp. So **normalisation was not driving R6A's result.** The confound existed — a
scaler that drifted with data quantity — but it was not what produced the gain. Under
the scaler R5B would have used, the gain is in fact **larger** (+23.05 vs +18.05).

This is the honest version of what R6A claimed. R6A could not have known it, because it
never ran the R5B arm.

## 4. The stress conclusion is withdrawn

R6A reported temporal stress 0.8256 (worst seed 0.5557) and distractors 0.7726 (worst
0.5713), and concluded that "temporal robustness is seed-fragile". **That was an
artefact of defect A2.**

With the scaler applied correctly:

| condition | both_A (fresh, 8 seeds) | worst seed |
|---|---:|---:|
| temporal shift, no retrain | **0.9099** | 0.7167 |
| distractors, no retrain | **0.5502** | 0.3401 |

Temporal shift now holds at **0.9099** — the R6A "fragility" was the unscaled tensor, not
the model. But **distractors genuinely collapse to 0.5502**, which is a real limitation
that R6A's defective measurement had partly hidden behind a different failure.

Stress transforms were verified to preserve the task before scoring (protocol §4):
oracle answers identical under shift and under distractors, query marker located at
tick 11 after shift. `stress_valid: True` for all 8 seeds.

## 5. Gate status — all computed, none hard-coded

Fresh confirmation, 8 seeds:

| gate | value | verdict |
|---|---|---|
| G1 both_A at 16 384 ≥ 0.75 | 0.9178 | **PASS** |
| G2 gain on **both_A** ≥ +5pp, CI lower > 0 | +23.05pp [+17.16, +29.48] | **PASS** |
| G3 matched unique − repeated on both_A > 0 | +15.08pp, worst seed +5.74pp | **PASS** |
| G4 random-label control ≤ 0.60 | **0.4775** | **PASS** |
| G5 shortcuts: first-key ≈ .75, query-only ≈ .50 | **0.7501 / 0.4934** | **PASS** |
| G6 stress ≥ 0.75 | temporal 0.9099, **distractor 0.5502** | **FAIL** |
| G7 determinism | identical head digests | **PASS** |
| G8 forward parity | 0.00e+00 (test) | **PASS** |

`confirmation_pass = False`, failing only G6, and only on the distractor condition.

## 6. CORRECTED_REPLAY vs FRESH_CONFIRMATION

Protocol §6 required these to be labelled separately, and they are:

| label | seeds | data gain (R5B scaler) |
|---|---|---:|
| `CORRECTED_REPLAY` | 120011/120017/120023 (R6A dev, already seen) | +18.61pp [+13.97, +21.24] |
| `FRESH_CONFIRMATION` | 130001…130037 (never used) | **+23.05pp [+17.16, +29.48]** |

The replay reproduces the direction on the same populations R6A used; the fresh seeds
give the preregistered number. **R6A's own confirmation seeds `121011…121043` were
deliberately not reused** — they were scored once under a defective harness and reusing
them would not be independent.

## 7. Metric table, R6A vs R6D

| quantity | R6A published | R6D corrected | why it moved |
|---|---:|---:|---|
| both_A at 16 384, fresh seeds | 0.9237 | **0.9178** | R5B scaler instead of R6A scaler |
| both_A at 1 024, fresh seeds | 0.7176 | **0.6873** | same |
| gain (both_A) | +20.6pp | **+23.05pp** | same, and now CI'd properly |
| gain used in G2 | +5.98pp **on direct** | **+23.05pp on both_A** | defect A4 |
| temporal stress | 0.8256 (worst 0.5557) | **0.9099** (worst 0.7167) | defect A2 |
| distractor stress | 0.7726 (worst 0.5713) | **0.5502** (worst 0.3401) | defect A2, and now genuinely worse |
| calibration | "fixed", actually drifted to 0.1699 | pinned pool, digest identical | defect A3 |
| G4 / G5 | hard-coded True | 0.4775 / 0.7501 / 0.4934 | defect A5 |

## 8. How far the result generalises — and how far it does not

**Supported:**
- More distinct episodes raise `both_A` from ≈0.69 to ≈0.92 on this generator, under
  **both** scalers, on 8 unseen seeds, with the worst seed at 0.7167 under temporal
  shift.
- At equal update count, distinct episodes beat repetition by **+15.08pp**, positive on
  every seed.
- The effect is **not** a normalisation artefact (preprocessing main effect within
  noise; interaction −5.01pp).

**Not supported:**
- **Distractor robustness** — 0.5502 fails the gate outright. New failure mode.
- **Any hidden-plasticity claim.** The hidden was frozen everywhere. There is no
  evidence here that hidden neurons learned anything, and no arm tested whether they
  could.
- **Any non-relational task.** One generator, one architecture, H=48.
- **Generalisation past three keys.** Two keys, two values only.

## 9. Limitations

1. **Two scalers, not a continuum.** I tested the two conventions R5B and R6A used; I
   did not search for a better transform.
2. **Distractor collapse is unexplained.** It is now a genuine, correctly-measured
   weakness and the most interesting open question this round produced.
3. **The calibration pool** (offset `+700000`) is my choice, made in this protocol. A
   different pool could shift absolute numbers slightly, though not the contrast,
   because every cell shares it.
4. **Six defects in one round of my own work.** That rate is itself a finding about the
   harness discipline, not just about this experiment.

## 10. Reproduction

```
PYTHONPATH=src python scripts/r6d_matrix.py replay     # CORRECTED_REPLAY
PYTHONPATH=src python scripts/r6d_matrix.py confirm    # FRESH_CONFIRMATION
PYTHONPATH=src python scripts/r6d_controls.py replay
PYTHONPATH=src python scripts/r6d_controls.py confirm
python -m pytest tests/test_r6d_audit.py -q            # 15 pass
```
