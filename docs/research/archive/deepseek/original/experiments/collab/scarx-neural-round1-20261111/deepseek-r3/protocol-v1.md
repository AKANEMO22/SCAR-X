# PROTOCOL v1 — Round 3A: general-purpose temporal readout substrate

**Branch:** `research/deepseek/r3-temporal-substrate`
**Base:** `7d590f768b36eb9d60e5c9433f1966e4e620cff3` (R2A tip)
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model:** `deepseek-ai/deepseek-v4.1-flash-1017`
**Frozen core:** ALIF H=48, fan-in 8, leak .65, threshold .5 hard spikes,
`feature_decay_internal = .25`, `adaptation_strength .02`. Hidden weights frozen
for the entire round.

---

## 0. What Stage 0 established, and what it did not

Stage 0 audit: **10/10 pass** (`artifacts/deepseek-r3/stage0-audit.json`).

Four R2A defects were fixed, as required by the brief:

| defect | fix |
|---|---|
| `D1` `test_ridge_oracle_matches_repository_probe` compared `ridge_probe` to itself (`abs(x−x) < 1e-12`) | replaced by an **independent SVD pseudo-inverse** reference, different estimator and no z-scoring |
| `D2` `S5_no_leakage` was hardcoded `True` | replaced by a real assertion; documented as still weaker than ideal (see §9) |
| `D3` forward parity vs `SpikingCore.step()` was never run in R2A | added; voltage/spike/feature parity **0.00e+00** |
| `D4` no manual reference for spike count / decay | added; impulse-decay matches hand computation to 1e-6 |

**Pre-protocol theory audit** (linear diagnostic only, 2 seeds, not a capability
claim) measured all five substrates on all five tasks:

| substrate | dims | sim_xor | xor | order | pair_binding | strict_binding |
|---|---:|---:|---:|---:|---:|---:|
| `decay025` | 48 | 0.7733 | 0.5617 | 0.8767 | 0.8683 | 0.5233 |
| `spikesum` | 48 | 0.9783 | 0.6950 | 0.8200 | 1.0000 | 0.5483 |
| `bank5` | 240 | 0.9717 | 0.6783 | 0.9933 | 0.9867 | 0.5167 |
| `count_plus_fast` | 96 | 0.9783 | 0.7017 | 0.9750 | 0.9933 | 0.5433 |
| `temporal_bins_3` | 144 | 0.9750 | 0.7033 | 0.9983 | 1.0000 | 0.5300 |

**Two facts from this audit must be stated before the screen, not discovered after:**

1. **`strict_binding` is not carried by any substrate** (0.5167–0.5483, all near
   chance at the linear-diagnostic level). My round-1 branch already established
   that `strict_binding` needs an order-3 feature that an additive point neuron
   cannot form. Adding temporal substrates does not change that. **I predict
   `strict_binding` will fail for every substrate**, and the screen gate below is
   written so that this prediction does not need to be violated to pass.
2. **The candidates only beat the baselines on `order`** (+0.098 `count_plus_fast`,
   +0.122 `temporal_bins_3`). On the other four tasks they are within ±0.04 of the
   better baseline.

## 1. Hypothesis

**H3A (primary).** A fixed feature map that retains **both** total activity and
temporal structure lets the online delta-rule head learn **more task families** than
`decay025` or `spikesum` alone — specifically at least one task where the better of
those two baselines fails, with no regression elsewhere.

**H3A-0 (the honest null).** No candidate beats the baselines outside `order`, so
the answer is "temporal structure helps only where order is already the dominant
cue", and `spikesum` remains the best general-purpose substrate at 48 dims.

**The neutral prediction from §0:** `strict_binding` fails for all five substrates.
A report that showed it passing would be more surprising than one that shows it
failing, and I am saying so before the run.

## 2. Candidate formulas — fixed, no tuning after this commit

Bounded streaming state; the spike history is never retained.

```
A decay025           f   = .25 f + .75 z                                  (48)
B spikesum           c   = Σ_t z                                          (48)
C bank5              f_k = d_k f_k + (1-d_k) z,  d ∈ {.10,.25,.50,.75,.90} (240)
D count_plus_fast    [Σ_t z ; g],  g = .50 g + .50 z                      (96)
E temporal_bins_3    [Σ_{t<4} z ; Σ_{4≤t<8} z ; Σ_{t≥8} z]                (144)
```

`FAST_D = .50` and `BIN_EDGES = (4, 8)` are fixed in advance. **No dimension is a
neuron; hidden stays 48.**

## 3. Substrate state cost

| substrate | streaming state |
|---|---:|
| `decay025` | 192 B |
| `spikesum` | 192 B |
| `bank5` | 960 B |
| `count_plus_fast` | 384 B |
| `temporal_bins_3` | 576 B |

All bounded, all independent of sequence length.

## 4. Tasks — five families, shared generators

| task | what it requires | order-dependent? |
|---|---|---|
| `sim_xor` | two bits on one tick | no |
| `xor` | two bits on different ticks | **yes** |
| `order` | which event came first | **yes** |
| `pair_binding` | key→value pairing across the window | partly |
| `strict_binding` | query at t=10 about data at t∈{1,2},{4,6} | **yes** |

All five are deterministic, balanced, repository-authored. No label appears in any
feature. Train/dev/confirmation use disjoint seed offsets.

## 5. Readers

| reader | role |
|---|---|
| `ridge` | **linear diagnostic only** — explicitly *not* an information ceiling |
| `head` | R2A delta rule: `lr=.08`, `clip=.025`, z-scored basis, `epochs=1` |
| `raw` | serving-style head with no conditioning, for the compatibility gap |

**Calibration disclosure:** the head z-scores using statistics built from the
**training split only**. This is *training-only calibration*, **not** strictly
online. No test feature ever enters the statistics (§9 test).

**One variable at a time:** every substrate is read by the **same** head with the
**same** hyperparameters. Changing substrate never changes the reader.

## 6. Splits, seeds, budget

| role | offset | count |
|---|---|---|
| calibration | `+100 000` | 128 |
| training | `+200 000` | 1024 |
| dev / test | `+300 000` | 512 |

- **Screen seeds:** `91001, 91007, 91013` — fresh, unused. **Not** R2A's 81001-set,
  **not** R2B's 92011-set.
- **Confirmation seeds:** `92001, 92007, 92013, 92019, 92023, 92029, 92031, 92037`
  — 8 fresh, not opened unless the screen passes.
- One setup, one epoch, no sweep. **600 s wall-clock cap for the whole screen.**

## 7. Screen gate — fixed now, not adjusted after results

A candidate passes only if **all** of:

| gate | threshold |
|---|---|
| **S1 reach** | candidate head ≥ 0.65 on **≥ 3 of 5** tasks |
| **S2 beats baseline** | ≥ **+5pp** over the better of `decay025`/`spikesum` on **≥ 2 tasks**, at least one of which is in `TIME_DEPENDENT` |
| **S3 no regression** | ≤ **3pp** drop versus that baseline on **every** task, including controls |
| **S4 no leakage** | calibration statistics provably free of dev/test features |
| **S5 deterministic** | rerun reproduces head digests bit-identically |
| **S6 budget** | wall ≤ 600 s; peak process-tree commit < 2 000 000 000 B |

`strict_binding` is scored and reported but **excluded from S2** (no substrate
carries it; §0 fact 1). It **is** included in S3, so a candidate may not damage it.

## 8. Stop rules

- One definition of "the" candidate per measurement; no averaging across candidates
  to hide a regression.
- If **both** candidates fail S1–S3 → **STOP**, report `spikesum` as the best
  general-purpose substrate at 48 dims and do not open confirmation.
- If a candidate passes → 8 fresh confirmation seeds, paired bootstrap CI95%
  (10 000 resamples, rng 2101), **lower bound > 0** on the primary contrasts, S3
  re-checked per task, and matched-capacity controls for any claim that depends on
  dimension count.
- Two adjustment rounds maximum; no adjustment after confirmation.

## 9. Honest weakness in my own audit

The `D2` no-leakage check is **stronger than R2A's hardcoded `True`** but still not
a full proof: it shows that training-only statistics differ measurably from
train+test statistics (so leakage would be visible) and that the statistics are
bit-identical whether or not dev data is in scope. The unit test
`test_calibration_uses_training_features_only` in `tests/test_r3_substrate.py`
does assert that calling `norm()` on dev features does not mutate the running
statistics, which is the operationally relevant property. I state this limitation
rather than overclaiming.

Similarly, `S5 deterministic` proves reproducibility of the computation, not the
absence of a subtle ordering bug. The manual-reference check in Stage 0 covers the
decay and count formulas directly.

## 10. Claim scope

Only: *on five synthetic 12-tick tasks at H=48 with a frozen hard-threshold ALIF
population, whether a fixed temporal feature map lets the online head exceed what
`decay025` or `spikesum` achieves, without regression.*

**Not claimed:** chat, vision, reasoning, retention, multitask transfer (each task
has its own readout — this measures *substrate learnability*, not a shared skill),
streaming deployment readiness, or that any substrate should replace the product
readout. A ridge diagnostic is a linear reference, never an information ceiling.
