# ARGUMENT AUDIT — Round 5B

Claim ledger. Every row cites an executed artifact.

Artifacts:
- `artifacts/deepseek-r5/stage0-audit.json`
- `runs/20261111T064558Z-r5-information-path-v1-cddd2b`
- `runs/20261111T064645Z-r5-dissection-v1-550f00`
- `runs/20261111T064716Z-r5-screen-v1-8b1a78`
- `runs/20261111T064932Z-r5-confirmation-v1-bef298`
- `runs/20261111T064835Z-r5-confirmation-v1-ad77ef` — **INVALID_HARNESS**, not cited

---

## Prior beliefs and how they could have been falsified

| belief | why | falsifier |
|---|---|---|
| The state holds both bits | the markers are strong pulses in 8 channels | a probe at chance on vA or vB |
| Binding is the hard part | a bag of values cannot say *which* key holds what | bag probe matches the mixed-only probe |
| 0.50 is the wrong null | independent values make first-key worth 0.75 | measuring the baselines |
| R4A generalises here | R4A's storage was perfect | heads stuck at the 0.75 ceiling |

## Claim ledger

### C1 — The 0.75 ceiling is real and not 0.50
- **Predicted:** always-answer-key-A ≈ 0.75.
- **Observed:** **0.7522**; knows-both-no-query 0.7473; majority 0.4900; query-only 0.4988.
- **Alternative:** the generator secretly correlates vA and vB.
- **Counter-test:** `corr(vA,vB) = +0.0207`; four states uniform
  (0.2529/0.2373/0.2524/0.2573).
- **Conclusion:** supported. Corroborated by Stage 1: all un-conditioned heads land
  0.7480–0.7773, exactly on the ceiling.

### C2 — Both independent values are stored
- **Predicted:** vA and vB each decodable pre-query.
- **Observed:** vA **0.9967**, vB **0.9974**, shuffled controls ~0.5.
- **Alternative:** the probe reads the input markers rather than neural state.
- **Counter-test:** `R4_decay025` reads only 0.7070/0.7168 on the same data with 48 dims,
  so the probe is not trivially reading pulses; and `spikesum` at 48 dims reaches
  1.0000/0.9993, so the information survives a lossy marginal summary.
- **Conclusion:** supported.

### C3 — Values are bound to the correct key
- **Predicted:** the mixed-only probe > 0.5 while a bag probe cannot do it.
- **Observed:** mixed-only **0.9947**; joint4 0.9863; bag3 0.9518.
- **Alternative:** bag3 succeeding means only the multiset is stored.
- **Counter-test:** bag3 collapse `{01,10}` into one class, so it cannot answer the
  mixed-only question at all; the 0.9947 is measured *within* that collapse.
- **Conclusion:** supported. **I first mis-flagged bag3 as a shortcut** (§Stage-1
  correction in the report) and retracted it.

### C4 — The query conditions the answer
- **Predicted:** post-query answer decode > pre-query.
- **Observed:** 0.7454 pre → 0.8984 post; R5 0.9948.
- **Alternative:** elapsed ticks, not the query.
- **Counter-test:** `query_zeroed_plus_cue` = **0.9876**, i.e. with the query channels
  zeroed at the population and the cue supplied to the readout, retrieval is *better*.
  So the query's effect on the core is not required — it is a readout-side selector.
- **Conclusion:** supported, and the mechanism is more specific than "query integration
  in the state": **the facts persist; the query selects.**

### C5 — The local head beats the shortcut ceiling
- **Predicted:** > 0.75 on held-out.
- **Observed:** screen 0.9635 (+21.13pp); confirmation clean **0.8989** (+14.67pp,
  CI [+11.45, +17.63] over 8 seeds).
- **Alternative:** it answers key A every time (0.7522).
- **Counter-test:** `needB` = 0.9254 on episodes where a first-key reader is 0.50 by
  construction; case-A flip consistency 0.8471 vs first-key's **0.0000**.
- **Conclusion:** supported.

### C6 — Timing robustness
- **Predicted:** holds under shift.
- **Observed:** OOD clean→shift **0.8977** (no retrain); shifted learning 0.8999;
  query-tick variation 0.9119.
- **Alternative:** the shift transform did nothing.
- **Counter-test:** the zero-pad shift was a **no-op** (`max|diff| = 0.0`, first run
  identical to four decimals across three conditions). Replaced with a noise-padded
  shift (`max|diff| = 1.0`). Only the corrected run is cited.
- **Conclusion:** supported — **conditional on the transform being a real perturbation**,
  which was not true of my first attempt.

### C7 — JOINT correctness on differing-value pairs is NOT achieved
- **Predicted by the screen:** 0.8419.
- **Observed on 8 fresh seeds:** **0.6676** (gate 0.70), while `both_B` = 0.9809.
- **Alternative:** a scoring bug.
- **Counter-test:** `both_B` uses the same code path and returns 0.9809, and the
  first-key null returns exactly 0.0000 on case A, so the metric separates readers.
- **Conclusion:** a **real limitation**. The screen did not replicate.

## Competing explanations for the main gain, each with a rejecting test

| apparent gain | alternative | rejecting test | outcome |
|---|---|---|---|
| R5 0.9635 | answers key A (0.75) | `needB` = 0.9254; case-A flips 0.8471 vs 0.0000 | rejected |
| R5 0.9635 | wider feature space | poststate-linear-258 = 0.7637; R4-matched-258 = 0.7637 | rejected — interaction adds +20pp |
| R5 0.9635 | memorised dataset statistics | train 0.9775 vs test 0.9635, and it survives shifted OOD 0.8977 | rejected |
| R5 0.9635 | reads the cue only | query-only arm = 0.5091 | rejected |
| query-zeroed 0.9876 | label leakage | features are label-independent (Stage 0 E) and case-A flips need the cue | rejected |

## Counterexamples where my strongest claim fails

1. **`both_A` 0.6676** on confirmation, below the gate, while `both_B` is 0.9809. My
   claim "the local head retrieves correctly" is true per-episode (0.8989) but **false**
   at the joint-pair level for differing values.
2. **Distractors and noise** push `both_A` to 0.4536 and 0.4512 — near the shortcut
   floor. The readout is close to capacity.
3. **R4A's 0.9316 does not transfer as 0.93 here**; the harder generator costs ~4pp on
   direct accuracy and ~17pp on the joint case-A metric.

## Causal vs correlational

- **C4 (query zeroing)** is causal: removing the query from the core leaves retrieval at
  0.9876, so it is the *cue*, not core integration, that selects.
- **C5 needB / case-A flips** are causal in the sense that they isolate episodes where a
  shortcut is arithmetically forced to 0.50.
- **C2/C3 storage claims** are correlational (probe accuracy) but capacity-controlled by
  the 48-dim R4 arms.

## Three reasons my conclusion may be wrong

1. **The generator may be solvable by position alone.** Writes sit at ticks 2 and 6 and
   the query at tick 10, so a reader could learn "channel 4 means query A" from position
   rather than content. *Test:* randomise write ticks per episode within a window and
   rerun; if accuracy holds, position is not the code. **Not run.**
2. **The cue is supplied from the input, so R5's success may be my construction rather
   than a property of the network.** `query_zeroed_plus_cue` = 0.9876 makes this
   sharper: the readout can retrieve with the core completely blind to the query.
   *Test:* force the readout to infer the query from the neural state alone (no input
   cue), and see whether accuracy survives. **Not run.**
3. **One generator, one architecture, 3 screen seeds and 8 confirmation seeds.** The
   screen→confirmation gap on `both_A` (0.8419 → 0.6676) shows 3 seeds are not enough
   for this metric. *Test:* a third, structurally different relational task with a
   pre-registered case-A gate. **Not run.**

## Limitations: confirmed vs speculative

**Confirmed by execution:** the 0.75 ceiling; both bits stored; binding to the correct
key; the head beats the ceiling by +14.67pp CI [+11.45,+17.63]; case-A flip
consistency 0.8471 vs a structural null of 0.0000; OOD temporal transfer 0.8977;
retrieval survives zeroing the query at the core.

**Speculative:** that R5 is a candidate worth deploying (the projection is fixed random
and the cue is external); that the mechanism generalises past three keys; that the
0.6676 is a capacity limit rather than a suboptimal readout; that any of this implies
capability.

## Final argument

The screen gate passed 9/9 and the confirmation failed C3. Protocol §11 says a
confirmation failure ends the round, and no threshold was relaxed. **STOP.**

The round nonetheless answers the scientific question it was given: R4A was **not**
merely a one-bit XOR artefact. Two independent facts are stored (0.9967 / 0.9974), bound
to their keys (0.9947 within the ambiguous subset), and a local online head retrieves
them above the 0.75 shortcut ceiling (0.8989, CI lower +11.45pp) including under OOD
temporal shift. What it does **not** do is get both siblings right on differing-value
pairs (0.6676 < 0.70) — the exact case where the first-key shortcut is most attractive,
and the case my screen over-estimated at 0.8419 on three seeds.
