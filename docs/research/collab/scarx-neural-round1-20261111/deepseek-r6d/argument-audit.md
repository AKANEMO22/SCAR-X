# ARGUMENT AUDIT — Round 6D

Claim ledger for the audit and the corrected experiment. Format:
**claim → prediction → measured → strongest alternative → counter-test → verdict.**

---

## C1 — R6A's normalization differed from R5B's

- **Prediction:** if the two conventions were equivalent, applying each to the same
  features would give identical tensors.
- **Measured:** R5B floor **0.05** + clip **±5**; R6A floor **1e-6**, no clip. The floors
  differ by 50 000×. `Scaler('r5b')` reproduces R5B's `basis()` to `2.38e-07`
  (float32 rounding); `Scaler('r6a')` produces `max|f|` unbounded above 5.
- **Alternative:** the difference does not matter because the features have `sd ≫ 0.05`.
- **Counter-test:** false. R4B's own geometry audit measured median feature sd
  **0.0042**, which is *below* R5B's 0.05 floor and far below R6A's 1e-6. The two
  conventions scale that coordinate by different amounts.
- **Verdict: CONFIRMED. Severity CRITICAL for cross-round comparison of magnitudes.**

## C2 — R6A's stress features were never scaled

- **Prediction:** the head is fit on scaled features; feeding it raw features should show
  a magnitude mismatch and a spurious accuracy drop.
- **Measured** (seed 121017): training `mean|f| = 0.780`, correctly-scaled test
  `0.783`, **as-Stage-5-did test `0.563`** (28% smaller).
- **Alternative:** the drop was real robustness loss.
- **Counter-test:** rerun with `apply_features(..., scaler)` — temporal both_A goes to
  **0.9099** (worst seed 0.7167) versus R6A's reported 0.8256 (worst 0.5557). The
  "fragility" disappears.
- **Verdict: CONFIRMED, and R6A's stress conclusion is WITHDRAWN.**

## C3 — R6A's calibration was not fixed

- **Prediction:** if fixed, the scaler digest must be identical across budgets.
- **Measured:** drift of the calibration mean vs the n=256 arm:
  `0.0891 / 0.1141 / 0.1699` at n = 1 024 / 4 096 / 16 384.
- **Alternative:** the drift is negligible relative to feature scale.
- **Counter-test:** feature `mean|f|` is order **0.78**, so a drift of 0.17 is a **22%**
  change in the centring — not negligible.
- **Root cause:** R6A fitted on `trx[:256]`, and budgets are **not nested** (overlap
  21/256). R6A's own test proved the non-nesting; the calibration claim contradicted it.
- **Verdict: CONFIRMED. Severity CRITICAL — data quantity was confounded with
  preprocessing in every R6A budget comparison.**

## C4 — The data effect is not a normalisation artefact

- **Prediction:** if the gain were driven by the drifting scaler, giving every budget the
  *same* pinned scaler should remove or reverse it.
- **Measured with the fixed pinned pool:**

| scaler | n=1 024 | n=16 384 | gain | CI95 |
|---|---:|---:|---:|---|
| R5B | 0.6873 | 0.9178 | **+23.05pp** | [+17.16, +29.48] |
| R6A | 0.7138 | 0.8942 | +18.05pp | [+12.98, +22.71] |

Preprocessing main effect: **+2.65pp [−1.37, +7.11]** at n=1 024 and
**−2.36pp [−4.85, +0.25]** at n=16 384 — both within noise. Interaction **−5.01pp**.

- **Alternative:** the R5B scaler is "better" and that is the real effect.
- **Counter-test:** the preprocessing effect is not significant at either budget and has
  *opposite signs* at the two budgets. A genuine scaler advantage would be consistent.
- **Verdict: the data effect SURVIVES correction.** Under R5B's scaler it is larger,
  not smaller.

## C5 — Unique data beats repetition at equal updates

- **Prediction (protocol G3):** `both_A(unique-16384) − both_A(repeated-256) > 0` at
  exactly 16 384 updates.
- **Measured:** **+15.08pp [+9.68, +22.23]**, **worst seed +5.74pp** — positive on all 8
  seeds. Under the R6A scaler: +19.84pp.
- **Alternative:** the repeated arm is harmed by clip drift, not by lack of information.
- **Counter-test:** the repeated arm reaches high *training* accuracy while losing
  held-out accuracy, i.e. it converges on its sample and fails to transfer. A
  clip-blocked learner would underfit train too.
- **Verdict: CONFIRMED under the corrected scaler, per-seed.**

## C6 — Distractor robustness genuinely fails

- **Prediction (protocol G6):** every stress condition ≥ 0.75 `both_A`.
- **Measured:** temporal **0.9099** (worst 0.7167) — passes on mean, one seed below;
  distractors **0.5502** (worst 0.3401) — **fails outright**.
- **Alternative:** another scaling artefact.
- **Counter-test:** the distractors path now goes through the identical
  `apply_features` pipeline that produces the passing temporal number, so the two
  conditions are measured with the same instrument. The distractor failure is real.
- **Verdict: G6 FAILS. A new, correctly-measured weakness.**

## C7 — No shortcut explains the result

- **Measured (fresh seeds):** random labels **0.4775**; first-key **0.7501**;
  query-only **0.4934**.
- **Alternative:** the head answers key A always (ceiling 0.75).
- **Counter-test:** reference `both_A` is **0.9504** against a 0.7501 first-key ceiling,
  and `needB` (query=B with `vA ≠ vB`) is the subset where first-key is structurally 0.50.
- **Verdict: PASS.**

## C8 — Hidden plasticity contributed nothing

- **Measured:** the hidden is **frozen in every cell**; sha256 verified unchanged.
- **Verdict: NO CONTRIBUTION, because there was no plasticity.** This round provides
  **zero** evidence about hidden learning. Any reading of these numbers as "the network
  learned" is wrong — this is readout learning on a fixed representation.

---

## Three ways my favoured explanation could be wrong

### Objection 1 — "Your pinned calibration pool is your own choice, so the corrected numbers are also arbitrary."

**Test run.** The pool (offset `+700000`, 256 episodes) is shared by **every** cell, so
it cannot favour one budget. And the 2×2 shows the conclusion is robust to the scaler:
+23.05pp under R5B, +18.05pp under R6A, both CI-positive.

**Remaining uncertainty:** a different pool shifts absolute values. It cannot
systematically favour n=16 384 because that arm never sees the pool's contents
differently from n=1 024.

### Objection 2 — "The unique-vs-repeated gap is a clip artefact."

**Test run.** The repeated-256 arm reaches high training accuracy before losing held-out
accuracy. A clip-blocked learner would be stuck on training too.

**Remaining uncertainty:** I measured endpoints, not within-arm convergence curves. I did
not localise where the repeated arm stops transferring.

### Objection 3 — "Six defects in one round means your corrected round is probably also defective."

**This is the objection I take most seriously, and I cannot refute it by argument.**
What I can report:

- Every defect was fixed **structurally**, not by patching a number: `apply_features`
  makes a raw evaluation path impossible; `fixed_calibration` takes only a seed, so no
  budget can influence it; `Scaler` requires an explicit convention string; no gate in
  the script is a literal boolean.
- 15 regression tests exist, one per defect, and each fails if the defect returns
  (`test_stress_uses_training_scaler`, `test_calibration_is_actually_fixed`,
  `test_no_gate_is_hardcoded`, `test_normalization_parity_with_r5b`, …).
- Three defects the brief suggested were checked and **did not reproduce**; I did not
  report them as real to appear thorough.

**Remaining uncertainty:** the same class of error — a harness that measures something
subtly different from what it reports — could exist in a form I have not thought to
test. The specific guard against it is that every number in `report.md` comes from the
same `apply_features` / `build_cell` path, and that path is the object of the tests.

---

## Limitations: confirmed vs speculative

**Confirmed by execution:**
- six R6A defects, each with a source-line citation and a measured effect;
- three brief-suggested defects did not reproduce;
- data gain on `both_A` = **+23.05pp [+17.16, +29.48]** over 8 unseen seeds under the
  R5B scaler, and +18.05pp under the R6A scaler, with the preprocessing main effect
  within noise;
- matched-update unique-vs-repeated = **+15.08pp**, positive on every seed;
- temporal stress holds at **0.9099** once scaled correctly;
- **distractor stress fails at 0.5502**;
- random-label / shortcut controls clean.

**Speculative / not established:**
- that the effect generalises past one generator, two keys and H=48;
- any hidden-plasticity conclusion (not tested);
- why distractors break the readout;
- whether a better scaler exists beyond the two conventions compared here.

## Final argument

The audit confirms six real harness defects in my own R6A, including one — a "fixed"
calibration that was not fixed — that R6A's own test had already given me the evidence
to catch and I missed. Correcting them changes the numbers but **not the direction**: the
data effect holds at +23.05pp under the scaler R5B would have used, and the
unique-vs-repeated contrast is positive on all eight unseen seeds. One published
conclusion is withdrawn outright — R6A's claim that temporal robustness was seed-fragile
was a normalisation artefact — and one new genuine weakness is exposed in its place:
distractor robustness fails at 0.5502.
