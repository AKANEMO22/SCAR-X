# PROTOCOL v1 — Round 6D: harness integrity and causal revalidation

**Branch:** `research/deepseek/r6d-audit`
**Base main:** `495dab2f4bac0180e4d1cd28674391f055ccf09d`
**Audited commit:** R6A `ceb98e33b40e8b47c6788b7d483e8b52a97d3351`
**Environment:** DeepSeek Harness (DSH) Web, Windows 11 · `deepseek-ai/deepseek-v4.1-flash-1017`

## 0. Status of this document

Committed **before** any of the runs it governs. The audit itself
(`docs/.../deepseek-r6d/audit.md`) was written first and is descriptive only. No
`CORRECTED_REPLAY` or fresh-confirmation number existed when this was committed.

## 1. Defects being corrected

Full evidence in `audit.md`. Summary:

| id | defect | severity |
|---|---|---|
| A1 | R6A normalizer (floor 1e-6, no clip) differs from R5B (floor .05, clip ±5) | CRITICAL |
| A2 | stress features passed **unscaled** to a head trained on scaled features | CRITICAL |
| A3 | "fixed calibration" was **not** fixed; budgets are not nested | CRITICAL |
| A4 | G2 gain computed on `direct`, not `both_A` | MAJOR |
| A5 | G4 and G5 hard-coded `True` | MAJOR |
| A6 | G3 and G7 absent from the gate set | MAJOR |

Three brief-suggested defects **did not** reproduce: query-tick handling, seed reuse,
and oracle/label correctness. Recorded as `N1`–`N3` in the audit.

## 2. Design — the 2×2 the brief requires

Factors: **data quantity** {1 024, 16 384} × **preprocessing** {R5B, R6A}.

| cell | n_unique | scaler |
|---|---:|---|
| P1 | 1 024 | R5B (floor .05, clip ±5) |
| P2 | 16 384 | R5B |
| P3 | 1 024 | R6A (floor 1e-6, no clip) |
| P4 | 16 384 | R6A |

**Main effect of data** = `(P2 − P1)` under R5B, and separately `(P4 − P3)` under R6A.
**Main effect of preprocessing** = `(P3 − P1)`. **Interaction** = `(P4−P3) − (P2−P1)`.
The interaction term is what determines whether R6A's scaling conclusion depended on its
scaler.

A third arm uses **`fixed_calibration`** — statistics from a dedicated pinned pool
(offset `+700000`, 256 episodes) that is independent of the training budget, so the
scaler is bit-identical in every cell. This isolates A3.

**Matched-update contrast:** 256 unique × 64 epochs vs 16 384 unique × 1 epoch, both at
exactly 16 384 updates, run under both scalers.

## 3. What must be separated (brief §2)

| cause | isolated by | never conflated with |
|---|---|---|
| normalization | the R5B/R6A factor | data quantity |
| new data | the 1 024 → 16 384 factor | update count |
| update count | the matched-update arms | data quantity |

The fixed-calibration arm exists precisely so that a scaler cannot drift while data
quantity changes.

## 4. Stress, correctly instrumented (defect A2)

Every stress evaluation goes through `apply_features(seed, x, scaler, query_tick)`,
which **always** applies the scaler learned from training. There is no raw path.

Before scoring, the stress transforms are verified to preserve the task:

- temporal shift (+1 tick): the oracle answer must be **identical** on shifted and
  unshifted episodes, and the query marker must be located at tick 11.
- distractors: the oracle answer must be **identical**, and the distractor pulses must
  be on background channels only.

If either check fails, stress is reported as `INVALID_STRESS_EVIDENCE`, not as a result.

## 5. Gates — all computed, none hard-coded

| gate | threshold |
|---|---|
| **G1** case-A both-correct at 16 384 (R5B scaler) | ≥ 0.75 |
| **G2** gain 16 384 − 1 024 **on `both_A`** | ≥ +5pp, paired CI95% lower > 0 |
| **G3** matched-update unique − repeated **on `both_A`** | reported; must be > 0 for the diversity claim |
| **G4** random-label control | ≤ 0.60 (must be near chance) |
| **G5** shortcut controls | first-key ≈ 0.75 ± 0.05, query-only ≈ 0.50 ± 0.05 |
| **G6** stress retains ≥ 0.75 on `both_A` with the training scaler | ≥ 0.75 |
| **G7** determinism | identical head digest on a repeat run |
| **G8** forward parity | 0.00e+00 vs `SpikingCore.step` |

`confirmation_pass = all(eight gates)`.

## 6. Seeds, and how runs are labelled

| label | seeds | meaning |
|---|---|---|
| `CORRECTED_REPLAY` | `120011, 120017, 120023` (R6A dev) | re-running seeds already seen, to show what the corrected harness produces on the same populations. **Not** fresh confirmation. |
| `FRESH_CONFIRMATION` | `130001, 130007, 130013, 130019, 130023, 130029, 130031, 130037` | never used in R5B, R6A or R6D. This is the only preregistered evidence. |

R6A's confirmation seeds `121011…121043` are **not reused** here. They were already
scored once under a defective harness; reusing them would not be independent.

## 7. Statistics

Paired, seed-level. For each contrast report: per-seed values, mean, worst seed, and a
bootstrap CI95% (10 000 resamples, rng 2101) over the seed vector. No pooling of
correlated counterfactual siblings as independent units.

## 8. Stop and labelling rules

- If a gate fails, the round reports **FAIL** for that question. No threshold moves.
- If the harness is found defective during a run, that run is committed and labelled
  `INVALID_HARNESS`; the fix is disclosed and the corrected result is **not** called
  preregistration.
- If the full matrix exceeds the resource cap, the completed prefix is reported as
  `INCOMPLETE_RESOURCE`.

## 9. Budget

Process-tree committed memory ≤ 2 000 000 000 B, one thread, CPU-first. Estimated wall
clock for the full matrix: well under 300 s based on R6A's 12–40 s stages.

## 10. Claim scope

Only: *after controlling normalization and matched updates fairly, does increasing the
number of independent episodes still improve `both_A`, and how far does that
generalise.*

**Not claimed:** that hidden neurons learned anything (the hidden is frozen; this is
readout learning on a fixed representation); that any non-relational task improves; that
any production capability exists.
