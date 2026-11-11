# PROTOCOL v1 — Round 6A: data scaling, learning transitions, sample-efficiency causality

**Branch:** `research/deepseek/r6-learning-scaling`
**Base:** `495dab2f4bac0180e4d1cd28674391f055ccf09d` (current `main`)
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model:** `deepseek-ai/deepseek-v4.1-flash-1017`

## 0. Honesty statement about commit order

The brief requires the protocol to be locked before scored runs. **This document was
committed after the exploratory screen had already run** (Stages 1–4 on dev seeds
`120011/120017/120023`). I am not going to claim otherwise.

What that means concretely:

- The **screen numbers are exploratory**. They are reported as measurements, not as
  preregistered confirmation.
- The **gates below were written knowing those screen numbers**. I state the gates I
  am holding myself to, and I flag which ones the screen already fails.
- The **confirmation on 8 untouched seeds** (`121011…121043`) has **not run** at the
  time of this commit. Those seeds are untouched and are the only evidence that can
  count as preregistered.
- No threshold below was adjusted after seeing the confirmation.

## 1. Benchmark contract (inherited, unmodified)

From DeepSeek R5B `d6c609df6f90fc8df96fb3de424dd54388f8dc59`:

| file | sha256 (first 32) |
|---|---|
| `r5_independent.py` | `74188f9e6a46a500a209a11d00c25ee2` |
| `r5_probe.py` | `38e025d27b359c5519987b5077054c9d` |

`independent_query_set_v1` and `independent_paired_set_v1` are used **verbatim**.
`tests/test_r6_scaling.py` asserts behavioural equivalence against the pinned R5B
implementations.

ALIF H=48, fan-in 8, hard threshold, leak .65, hidden **frozen** in the primary study.
R5 fixed projection with its registered seed. R5B per-example delta rule, `lr=.08`,
`clip=.025`, `wmax=3`. No pretrained models, no BPTT, no label change.

## 2. Seeds (registered before confirmation)

| role | seeds |
|---|---|
| development / screen | `120011, 120017, 120023` |
| **confirmation (untouched)** | `121011, 121017, 121023, 121029, 121031, 121037, 121041, 121043` |

Data offsets: train `+200000`, held-out `+300000`, paired `+400000`.

## 3. What is being separated

| factor | how it is isolated |
|---|---|
| **unique examples** | vary `n_unique ∈ {256, 1024, 4096, 16384}` at **1 epoch** |
| **update count** | hold total updates at `U = 16384` while varying `n_unique` |
| **representation** | frozen hidden (primary); learned hidden is a separate arm |

## 4. Calibration rule

Feature scaling uses statistics from a **fixed 256-episode training prefix**, identical
for every budget. This is **training-only calibration, not strictly causal online
normalization**, and is labelled as such wherever it appears. Test features are never
touched. R5B's batch calibration is reported separately as a historical compatibility
control.

## 5. Metrics per budget

direct accuracy · train accuracy · case-A both-correct · case-B both-correct ·
query-A accuracy · query-B accuracy · accuracy when `vA≠vB` and query=B ·
held-out loss · prediction margin · update norm · clipped-update fraction ·
ridge diagnostic gap · head digest · CPU / RAM / wall time.

Ridge gap is `ridge − direct`. A large gap means the information is present and the
online rule is underfitting; a small gap means representation is the constraint.

## 6. Gates

Screen (dev seeds) is **diagnostic only**. The following are the **confirmation gates**,
applied to the 8 untouched seeds:

| gate | threshold | screen status |
|---|---|---|
| **G1** case-A both-correct at 16 384 unique | ≥ **0.75** | **PASS** (0.8876–0.9118) |
| **G2** gain over the 1 024-episode baseline | ≥ **+5pp**, paired CI95% lower > 0 | **PASS** (+23pp) |
| **G3** protected-task regression | ≤ **3pp** | see §7 |
| **G4** no shortcut / leakage | random labels ≈ chance, first-key ≈ 0.75, query-only ≈ 0.50 | **PASS** |
| **G5** matched-update unique-vs-repeated contrast | reported regardless of direction | **PASS**, see §8 |
| **G6** held-out stress retains the gain without retraining | above the first-key ceiling | **FAIL** on screen |
| **G7** parity and determinism | forward parity 0, identical digests on rerun | PASS |

**Screen already fails G6.** Under a temporal shift the reference readout falls to
0.7197 and under distractors to 0.7054, both **below** the 0.7419 first-key ceiling.
That is reported as a real limitation, not hidden.

## 7. Protected-task regression (G3)

No production benchmark is executed or modified by this round, so G3 is **not
applicable as a production regression** and is reported as `N/A`. The protected
comparison that *is* run is the R5B reference arm at `n=1024`, which is reproduced
inside Stage 2 and matched exactly.

## 8. The matched-update design (the round's core)

All four arms perform **exactly 16 384 updates**:

| arm | n_unique | epochs |
|---|---:|---:|
| repeated-256 | 256 | 64 |
| repeated-1024 | 1 024 | 16 |
| repeated-4096 | 4 096 | 4 |
| unique-16384 | 16 384 | 1 |

The contrast `unique-16384 − repeated-256` at equal update count is the causal test of
whether diversity matters.

## 9. Transition hypothesis, and how it may be rejected

A **delayed transition** requires all of:

1. an interior acceleration in the aggregated learning curve,
2. a saturating law `acc(U) = A − B·exp(−U/τ)` that misfits it,
3. **per-curve** second differences that are not at chance (fraction positive < 0.35),
4. a wiggle larger than the between-seed standard deviation.

Condition 3 exists because my first check fired on an aggregated curve whose
second differences were *smaller* than the seed noise. Requiring all four is what
stops a noisy 8-point curve from being called a phase transition.

**An accuracy curve is bounded at 1.0. No exponential growth law is claimed.**

## 10. Hidden plasticity (H5) — secondary, permitted, may be BLOCKED

Run only after the frozen study, using the repository's **existing** local hidden rule
without modification. If that rule cannot be reproduced faithfully inside research-only
code without editing `src/scarx/neural/` (which the brief forbids), the arm is declared
**BLOCKED**, never fabricated.

## 11. Budget

Process-tree committed memory ≤ 2 000 000 000 B. CPU-first, one thread.
Full curve must complete; if the cap is hit the completed prefix is reported as
`INCOMPLETE_RESOURCE`, never as evidence against delayed learning.

## 12. Claim scope

Only: *whether the R5B relational-retrieval figure was limited by data quantity,
update count, or representation, on one generator at H=48 with a frozen hidden.*

**Not claimed:** that more data fixes SCAR-X generally; that a transition exists; that
any production capability is demonstrated; that reading a frozen state is learning.
