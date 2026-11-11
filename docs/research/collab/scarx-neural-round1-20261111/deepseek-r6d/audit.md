# HARNESS AUDIT — Round 6D

Independent re-audit of R6A (`ceb98e33`). The brief listed candidate defects; each is
verified below against source lines and, where possible, measured. **Four of the
claimed defects are real. Two are not.** One additional defect is found that the brief
did not name, and it is the most serious.

Severity: **CRITICAL** = invalidates a published conclusion; **MAJOR** = changes a
number; **MINOR** = cosmetic.

---

## A1 — R5B and R6A normalize differently — **CRITICAL, CONFIRMED**

**R5B** (`r5_probe.py`):
```
s = np.maximum(np.sqrt(self.m2 / max(1, self.n - 1)), .05)
return np.clip((f - self.mean) / s, -5, 5)
```
variance floor **0.05**, then **clip ±5**, using **running** statistics updated by
`observe()`.

**R6A** (`r6_scaling.py` and every stage script):
```
mu, sd = cal.mean(0), np.maximum(cal.std(0), 1e-6)
```
variance floor **1e-6**, **no clip**, **fixed prefix** statistics.

**Impact.** The floor differs by a factor of 50 000. In R4B's geometry audit the
decay-0.25 features had median sd **0.0042** — below R5B's 0.05 floor but far above
R6A's 1e-6. So R5B *compresses* those coordinates toward zero while R6A *divides by
their true scale*, amplifying them. The unclipped form additionally lets outlier
coordinates reach `max|f| = 21.76` in R6A where R5B would cap at 5.

**Consequence for R6A's claims.** R6A's absolute numbers are on a different feature
scale from R5B's. The brief's own comparison (`0.9237` here vs `0.6676` there) is
therefore **not a controlled comparison**. Direction may survive; the magnitudes do not.

**Regression test that would catch it:**
`test_normalization_parity_with_r5b` (in `tests/test_r6d_audit.py`) asserts that the
R5B transform is reproduced exactly, including the floor and the clip.

---

## A2 — Stress features were never scaled — **CRITICAL, CONFIRMED**

`r6_stage5_confirmation.py`, stress block:

```
xp = G.shifted(d["tex"], 1, seed=seed)
fp = S.capture_features(seed, xp, QT + 1)      # <-- RAW features
...
h.acc(fp, d["y_te"])                            # head trained on SCALED features
```

`capture_features` returns the **unscaled** bilinear map. The head `h` was trained on
`d["f_te"]`, which in `build()` is passed through `(a - mu) / sd`. **The stress
evaluation therefore feeds a differently-scaled tensor to a head that has never seen
that scale.**

**Measured** (seed 121017, 512 held-out episodes):

| tensor | mean \|f\| | max \|f\| |
|---|---:|---:|
| training (scaled) | 0.780 | 15.97 |
| test, scaled correctly | 0.783 | 21.76 |
| **test as Stage 5 did it** | **0.563** | **5.89** |

The stress arm saw features ~28% smaller in magnitude than anything the head was fit on.

**Consequence.** The G6 failure and the reported "stress drops" of +25.59pp / +27.02pp
are **measurement artefacts**. R6A's conclusion that "temporal robustness is
seed-fragile" is **withdrawn pending re-measurement**.

**Regression test:** `test_stress_uses_training_scaler`.

---

## A3 — Calibration was NOT fixed across budgets — **CRITICAL, CONFIRMED**
### This is the defect the brief did not list, and it is mine.

`r6_stage1_scaling.py` claims in its own docstring:

> `"calibration": "fixed 256-episode training prefix, identical for all budgets"`

**That claim is false.** Budgets are not nested: `independent_query_set_v1(seed, n)`
draws `n` episodes then reshuffles, so `trx[:256]` at `n=256` and at `n=16384` are
**different episodes**. R6A's own test caught the non-nesting and I documented it in
the report — but I did **not** notice that it invalidates the calibration claim.

**Measured** (seed 121011, calibration mean drift relative to the n=256 arm):

| n | max \|calib_mean(n) − calib_mean(256)\| |
|---:|---:|
| 1 024 | 0.0891 |
| 4 096 | 0.1141 |
| 16 384 | 0.1699 |

So the scaler changed by up to **0.17** in feature units between arms while the head's
input distribution is order-1. Data quantity and preprocessing were therefore
**confounded in every R6A budget comparison**.

**Regression test:** `test_calibration_is_actually_fixed` recomputes each arm's scaler
from the *same pinned prefix* and asserts bit-equality across budgets.

---

## A4 — G2 used `direct`, not `both_A` — **MAJOR, CONFIRMED**

`r6_stage5_confirmation.py`:
```
def ladder(n, k="direct"):          # <-- default is direct
gain = np.array(ladder(16384)) - np.array(ladder(1024))
```
The primary relational metric is `both_A`. The gain gate was computed on `direct`,
which is inflated by the 25% of episodes where `vA == vB` and a first-key reader is
already correct.

**Impact.** G2's `+5.98pp` describes `direct`, not `both_A`. It overstates the
relational improvement. Re-derivation is in §3 of `report.md`.

---

## A5 — G4 and G5 hard-coded TRUE — **MAJOR, CONFIRMED**

```
"G4_random_labels_chance": True,      # verified on dev in stage 4
"G5_matched_update_reported": True,
```
Neither was computed inside the confirmation run. G4's controls *were* genuinely run on
the dev seeds (Stage 4, `runs/…r6-controls-v1-77a238`), so the value is not invented —
but a gate that cannot fail is not a gate. `confirmation_pass = all(gates.values())`
includes these two as free passes.

## A6 — G3 and G7 absent from the gate list — **MAJOR, CONFIRMED**

`G3` (protected-task regression) and `G7` (parity/determinism) do not appear in the
confirmation script at all. Grep: `'G3' in src → False`, `'G7' in src → False`.
`confirmation_pass` therefore cannot reflect them.

---

## Claims from the brief that did NOT reproduce

### N1 — "Query tick may be wrong in stress" — **NOT A DEFECT**

Verified: `temporal` uses `QT + 1` (correct, the shift moves the query one tick);
`distractor` uses `QT` (correct, distractors do not move the query). This is the R4A
failure mode and it was **not** repeated here.

### N2 — "Seed reuse between dev and confirmation" — **NOT A DEFECT**

`DEV = {120011, 120017, 120023}`, `CONF = {121011…121043}`, intersection **empty**.
No non-confirmation script references `CONF_SEEDS`. Confirmation seeds were untouched
before Stage 5.

### N3 — Oracle and labels — **NOT A DEFECT**

`oracle_all` locates its markers rather than assuming ticks, and R5B's tests (reused)
assert `oracle == y` on 4096 episodes and that shift/reversal/distractors preserve the
answer. No label leakage found: features are label-independent and hidden weights are
frozen (sha256 verified).

---

## Defect-to-test map

| id | severity | defect | test that would catch it |
|---|---|---|---|
| A1 | CRITICAL | normalizer differs from R5B | `test_normalization_parity_with_r5b` |
| A2 | CRITICAL | stress features unscaled | `test_stress_uses_training_scaler` |
| A3 | CRITICAL | calibration not fixed | `test_calibration_is_actually_fixed` |
| A4 | MAJOR | G2 on `direct` not `both_A` | `test_gain_gate_uses_both_a` |
| A5 | MAJOR | G4/G5 hard-coded | `test_no_gate_is_hardcoded` |
| A6 | MAJOR | G3/G7 missing | `test_all_registered_gates_present` |

## What must be re-run

Everything that used the R6A scaler for both training and evaluation, and every
stress number. That is: the whole budget ladder, the matched-update contrast, and the
stress rows. The 2×2 design in `protocol-v1.md` does exactly that.

## What survives

- The **direction** of the data effect is not explained away by A1–A3: A3 shows data
  quantity was *confounded* with scaling drift, which is a confound, not a reversal.
  §3 of the report tests whether it reverses.
- The **non-nesting** of budgets and the **no-transition** finding (per-curve second
  differences at chance) are unaffected: they concern the curve shape, not the scaler.
- The **R6A stress conclusion is withdrawn** outright (defect A2).
