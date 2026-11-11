# PROTOCOL v1 — Round 5B: independent key/value binding and query retrieval

**Branch:** `research/deepseek/r5-independent-relational-binding`
**Base:** `495dab2f4bac0180e4d1cd28674391f055ccf09d` (current `main`)
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model:** `deepseek-ai/deepseek-v4.1-flash-1017`
**Core:** ALIF H=48, fan-in 8, hard threshold, leak .65, hidden **frozen** throughout.

---

## 0. Why R4A cannot answer this question

R4A used `values = (stored, 1 − stored)`. The two values were **complementary**, so
`answer = stored XOR query` and the whole task carried **one bit**. R4A's 0.9987
pre-query decode and 0.9316 head are real, but they say nothing about whether ALIF
holds two **independent** facts or whether a readout can select between them.

R4A also had a confirmed harness defect: `query_cue()` read tick 10 while the shift
generator moved the query to tick 11. **No temporal-shift number from R4A is treated
as established here.**

## 1. What is different in this round

```
vA, vB ~ Uniform{0,1}   drawn INDEPENDENTLY
q      ~ Uniform{0,1}
answer = vA if q==A else vB
```

All four memory states (00, 01, 10, 11) are legal and uniform.

## 2. Stage 0 result — the ceilings are 0.75, not 0.50

This is the single most important design fact of the round, and it is **measured**,
not assumed. Stage 0 audit: **28/28 pass**.

| baseline | measured | what it means |
|---|---:|---|
| **always answer key A** | **0.7522** | a 75% score may be no query use at all |
| knows both values, no query | 0.7473 | same ceiling without the query |
| majority class | 0.4900 | |
| query-only | 0.4988 | |

Also verified: `corr(vA,vB) = +0.0207`; all four states uniform
(0.2529 / 0.2373 / 0.2524 / 0.2573); oracle 4096/4096; counterfactual case A
(538 pairs) all flip and case B (486 pairs) none flip; siblings differ **only** at
the query tick; forward parity `0.00e+00`; hidden weights unchanged.

**Every capability claim in this round is measured against 0.75, never 0.50.**

## 3. Hypotheses with explicit predictions

| id | hypothesis | prediction if true | falsifier |
|---|---|---|---|
| **H1** storage | both bits are held | pre-query decode of `vA` and `vB` each above chance, and the **joint 4-state** decode recovers all four | joint decode ≤ a 1-bit probe |
| **H2** binding | the state knows *which* value belongs to *which* key | joint 4-state decode succeeds while a bag-of-values probe fails | bag probe matches joint probe |
| **H3** query integration | the query creates an answer-specific code | post-query state decodes the answer above the pre-query state, and zeroing the query destroys it | no change from zeroing |
| **H4** readout learnability | a local online head can learn it | head beats 0.75 on held-out, on both pair cases | head ≤ 0.75 |
| **H5** shortcut | success is positional/label-driven | first-key, query-only, majority match the learner | controls fall far below |
| **H6** temporal robustness | the mechanism survives moving events | accuracy holds under train-shift and under OOD test-shift | collapse under shift |

**H1 vs H2 are separated by construction**: a probe that reads only the *multiset* of
values `{vA, vB}` can distinguish 00 / {01,10} / 11 but **cannot** separate the
swapped pair {01} vs {10}. A joint 4-state probe that succeeds where the bag probe
fails is evidence of binding, not merely of storage.

## 4. Representations

| id | content | role |
|---|---|---|
| **R0** | input oracle | task-well-specified check; never SCAR-X's answer |
| **R1** | pre-query state at the tick before the query | diagnostic |
| **R2** | full spike trajectory | accessibility only, not a proposed history store |
| **R3** | post-query state | diagnostic |
| **R4** | `decay025`, `spikesum`, `bank5` | production comparison |
| **R5** | fixed bilinear interaction + query cue | historical mechanism under test |

`capture()` takes `query_tick` as a parameter and the oracle **locates** markers
rather than assuming ticks 2/6/10. R4A's harness defect cannot recur.

## 5. Counterfactual scoring — two cases, never merged

| case | condition | required behaviour | metric |
|---|---|---|---|
| **A** | `vA ≠ vB` | swap must **flip** | flip consistency |
| **B** | `vA = vB` | swap must **not** flip | invariance |

Case B is **never** scored as a flip failure. Both cases are reported with their
weights (`caseA_fraction` measured at 0.525 in Stage 0). Additionally report
**both-answers-correct**: the fraction of pairs where *both* siblings are answered
correctly. That is the metric a 0.75 shortcut cannot fake.

## 6. Baselines — all six required

1. majority class (~0.50)
2. query-only (~0.50)
3. first-key (~0.75)
4. query-free but knows both values (~0.75)
5. **correct pair-specific null** — the score of always answering key A evaluated
   *on the paired counterfactual corpus*
6. counterfactual both-answers-correct

## 7. Splits, seeds, budget

| role | offset | count |
|---|---|---|
| train | `+200 000` | 1024 |
| dev / test | `+300 000` | 512 |
| paired test | `+400 000` | 256 pairs (512 rows) |

- **Screen seeds:** `97001, 97007, 97013` (fresh; not R2A 81001, not R3A 91001, not
  R4A 95001, not R4B 94011).
- **Confirmation seeds:** `98001, 98007, 98013, 98019, 98023, 98029, 98031, 98037`.
- Screen ≤ **600 s**, one thread, process-tree commit ≤ 2 000 000 000 B.

## 8. GO gate — locked now, not adjusted after results

All of:

| gate | threshold |
|---|---|
| **G1** direct held-out accuracy | ≥ **0.85** on the full independent set |
| **G2** gain over strongest fair baseline | ≥ **+10pp** over max(first-key 0.75, knows-both 0.75, query-only 0.50) |
| **G3** both-answers-correct, case A | ≥ **0.70** |
| **G4** both-answers-correct, case B | ≥ **0.70** |
| **G5** paired CI95% lower | > **0**, bootstrapped over **independent units** `(seed, pair)` |
| **G6** label leakage | no label in any feature or state update |
| **G7** protected-benchmark regression | ≤ **3pp** drop |
| **G8** numerical / forward parity | parity `0.00e+00`, no NaN |
| **G9** resources | within budget |

**A score near 0.75 is explicitly treated as failing G1 and G2.** The protocol states
this before the run so that a 75% result cannot later be reinterpreted as success.

## 9. Stage structure

- **Stage 1** information-path audit R0–R5, frozen, no candidate.
- **Stage 2** functional dissection of R5 against eight controls, including
  true-vs-shuffled cue split into shuffle-in-train and shuffle-at-test, and
  query-zeroed state **separated** from query-zeroed-but-cue-supplied.
- **Stage 3** at most **one** candidate, only if Stage 1 supports it. A
  relative-query-conditioned readout is the permitted direction; anything else needs
  a stated reason it would transfer.
- **Stage 4** confirmation on 8 fresh seeds, only on GO.

## 10. Temporal stress, correctly separated

| question | train | test | what it measures |
|---|---|---|---|
| **in-distribution shifted learning** | shift 1 | shift 1 | can it learn when everything moved |
| **OOD temporal transfer** | clean | shift 1 | does it transfer without retraining |

Plus: query-tick variation, reversed write order, distractors, noise change,
unseen association configurations.

**No single shift experiment may conflate timing with background noise.** The noise
change is applied as its own separate condition.

If a harness defect is found during confirmation, the failing run is committed and
labelled `INVALID_HARNESS`; the fix is disclosed and is **not** claimed as pure
preregistration.

## 11. Stop rules

- Stage 0 fail → STOP, no capability experiment.
- Stage 1 shows no supported path → STOP, no candidate invented.
- Screen fails G1–G9 → STOP. **No threshold relaxation after seeing scores.**
- Confirmation fails any hard gate → STOP.

## 12. Claim scope

Only: *whether frozen ALIF H=48 holds two independent key/value facts and supports a
local online readout that answers unseen queries above the 0.75 shortcut ceiling, or
whether R4A's success was specific to a one-bit XOR.*

**Not claimed:** chat, vision, reasoning, retention, lifelong learning, P2, or
production readiness. Reading a frozen state is not learning. A ridge is a linear
diagnostic, never an information ceiling, and never evidence that a head can learn.
