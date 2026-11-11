# PROTOCOL v2 — head conditioning, then learning signal

**Campaign:** `scarx-neural-round1-20261111`
**Agent:** deepseek (same session as `antigravity`; independence weight zero)
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model:** `deepseek-ai/deepseek-v4.1-flash-1017`
**BASE_SHA:** `1735ad66ea9e7b416f5df15df99d9c13cea46b4e`
**Supersedes nothing.** v1's screen stands (PIVOT). This is a new experiment
motivated by the finding in `correction-2-head-vs-ridge.md`.

Committed **before** any formal run.

---

## 1. The problem this protocol addresses

Measured in `correction-2`, frozen hidden, `sim_xor`, hard-threshold neuron:

| reader | accuracy |
|---|---:|
| z-scored **ridge** (offline) | 0.6867 – 0.7233 |
| **delta-rule head** (what the learning rule updates) | 0.4967 – 0.5267 |

Same features. The head never leaves chance across 1500 examples × 3 passes, while
an offline ridge on the same features reads 0.70.

`tasks21.ridge_probe` z-scores every dimension (`(f − mean)/std`) before fitting;
the delta rule multiplies **raw amplitude**. On a 48-neuron spike feature vector,
a few high-variance dimensions dominate the update and the discriminative
low-variance dimensions are ignored.

**Consequence for the campaign:** every hidden-learning comparison so far — all
three branches' negatives included — used a head of the second kind. If the head
cannot read the features, hidden learning has nothing to improve and the nulls are
uninformative about the learning rule.

## 2. Hypotheses

**H1 (head conditioning).** A delta-rule head with running feature normalisation
reaches the offline ridge's accuracy on a **frozen** hidden. Falsified if the
conditioned head stays ≥ 5pp below ridge on `sim_xor`.

**H2 (signal matters only after H1).** On a head that satisfies H1, a per-neuron
`W_out.T @ error` factor produces a larger hidden-representation gain than the
scalar score rule. Falsified if `prop30 − scalar` CI95% lower ≤ 0 on
`sim_xor`.

**H3 (no free lunch).** Head conditioning alone does **not** improve hidden
representation. Falsified if the conditioned-frozen arm's refit probe exceeds the
raw-frozen arm's by a CI95% lower bound > 0, which would mean my attribution to
scale was wrong.

H2 is only tested if H1 passes. If H1 fails, the report says so and stops — the
prerequisite was not met and running H2 would repeat correction-2's error.

## 3. Identity and signal boundary

One ALIF population, H=48, `fan_in=8`, `plastic_rows=12`, unchanged neuron
equations. No serving-path edits. No autograd, BPTT, RTRL, or hidden→hidden error
transport.

The `prop30` arm uses `W_out.T @ error` per neuron, which is the boundary approved
in proposal 30, disclosed as **not fully biologically local**. It is used here for
the first time under a head that can actually read the features.

`bptt` is **not** used in this protocol.

## 4. Conditions

### Stage A — head conditioning (H1, H3)

| arm | head update |
|---|---|
| `raw` | `ΔW = lr · err ⊗ f` (current behaviour) |
| `zscore` | running mean/var per feature; `ΔW = lr · err ⊗ (f − μ)/σ` |
| `rms` | running RMS per feature; `ΔW = lr · err ⊗ f/rms` |
| `zscore_frozen` | `zscore` head, hidden frozen |
| `raw_frozen` | `raw` head, hidden frozen |

`zscore` and `rms` normalise **inside the head**, before the readout. This is a
readout change, not a neuron change. Feature statistics are updated online from
training features only (no test data), and are part of the checkpoint.

### Stage B — learning signal on the winning head (H2)

Only if H1 passes. On the conditioned head:

| arm | hidden signal |
|---|---|
| `scalar` | `clip(−loss − baseline, −2, 2) · elig` (current rule) |
| `prop30` | `(W_out.T @ err)_j · elig_j` per neuron |
| `shuffle` | `prop30` with the per-neuron factor permuted across neurons |
| `zero` | no hidden update (must equal frozen bit-for-bit) |
| `frozen` | hidden never updated |

## 5. Data, splits, seeds

Generator: `sim_xor` (from `point-neuron-nonlinearity.md`) — two bits on one tick,
label = XOR, raw-tape order-1 = 0.49. Hard-threshold neuron (what `unified.py`
ships).

| role | seed offset | dev | confirmation |
|---|---|---|---|
| train | `+200 000` | 1500 | 1500 |
| probe fit | `+100 000` | 128 | 128 |
| dev/test | `+300 000` | 300 | 512 |

- **Stage A dev seeds:** `71001, 71007, 71013` (fresh, unused).
- **Stage B dev seeds:** `72001, 72007, 72013`.
- **Confirmation:** `73001, 73007, 73013, 73019, 73023, 73029, 73031, 73037`
  (8 fresh) — opened only if the dev screen passes.

Labels enter only the readout target. A P0 test asserts the input hash is unchanged
under label permutation.

## 6. Budget

H=48, 1500 examples × 3 update passes = 4500 updates per arm per seed. 12 ticks.
Hard cap 3 hours. Peak process-tree commit recorded.

Stage A: 5 arms × 3 seeds = 15 trials. Stage B: 5 arms × 3 seeds = 15 trials.

## 7. Metrics and gates

**Primary (Stage A):** `head_acc` = online head accuracy on the dev split.
**Primary (Stage B):** `probe_gain` = refit-head ridge on learned hidden minus on
initial hidden, same fit/test split, same algorithm.

| gate | threshold |
|---|---|
| **G1 (H1)** | `zscore` or `rms` head within 5pp of the ridge on the same frozen hidden, on every dev seed |
| **G2 (H3)** | `zscore_frozen` probe_gain ≤ raw_frozen probe_gain + 0 (conditioning must not itself help the hidden) |
| **G3 (H2)** | `prop30 − scalar` probe_gain CI95% lower > 0 |
| **G4 (signal validity)** | `prop30` > `shuffle` and > `zero` with CI95% lower > 0 |
| **G5 (acquisition)** | winning head ≥ 65% dev, not chance |
| **G6 (control)** | `zero` and `frozen` weights bit-identical |

Statistics: paired by seed; 95% seed bootstrap, 10 000 resamples, rng 2101
(`phase21.interval` — the repository convention, named explicitly).
`readout_lr ∈ {0.02, 0.05, 0.10}` screened on dev, locked before confirmation.

## 8. P0 (required before any capability number)

1. `zero` == `frozen` bit-for-bit (weights, RNG).
2. Label permutation leaves input hash and hidden updates unchanged at equal reward.
3. Normalisation statistics computed from training features only; never from test.
4. Checkpoint round-trip bit-exact; cold-process resume reproduces the next step.
5. Rollback restores every array.
6. `prop30` uses the per-neuron factor, verified by asserting the signal shape is
   `(plastic_rows, ...)` and not a broadcast scalar.
7. Peak RSS + commit, in-process and across the process tree, < 2 GB.
8. `pytest -q` passes (pre-existing failures recorded with proof they pre-date this).

## 9. Stop rules

- H1 fails → **STOP**. Report that head conditioning did not close the gap; do not
  proceed to Stage B. Do not tune the normaliser to rescue it.
- H1 passes, H2 fails → report **NO-GO for the per-neuron signal under a capable
  head**. This is the informative negative the campaign has been missing.
- H3 fails (conditioning itself helps the hidden) → report it; my scale attribution
  was incomplete.
- Two adjustment rounds maximum. No retune after seeing confirmation.

## 10. Claim scope

Only: *on `sim_xor` at H=48 with a hard-threshold ALIF population, whether an
online delta-rule head can be conditioned to read what an offline ridge reads, and
whether a per-neuron `W_out.T @ error` factor then beats the scalar rule.*

Not claimed: chat, vision, reasoning, retention, general binding, or that SCAR-X
now learns. `sim_xor` is a 12-tick synthetic task. An oracle ridge is not a skill.
