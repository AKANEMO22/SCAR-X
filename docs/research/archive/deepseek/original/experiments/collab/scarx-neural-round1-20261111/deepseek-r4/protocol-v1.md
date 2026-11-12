# PROTOCOL v1 — Round 4A: relational representation, query conditioning, local learnability

**Branch:** `research/deepseek/r4-relational-representation`
**Base:** `495dab2f4bac0180e4d1cd28674391f055ccf09d` (current `main`)
**Environment:** DeepSeek Harness (DSH) Web, Windows 11
**Model:** `deepseek-ai/deepseek-v4.1-flash-1017`
**Core:** ALIF H=48, fan-in 8, hard threshold, leak .65, hidden frozen throughout.

---

## 0. Stage 0 result, recorded before hypothesis testing

Stage 0 audit **12/12 pass**. `query_set` is a verbatim copy of Grok R3B
(`blob 874fb835…`, body sha256 `01b50b4d…`) and an independent re-implementation
from the written specification reproduces the task statistics. Oracle matches
128/128 labels. Counterfactual siblings differ **only** at the query tick.

**Reproduction of Grok's near-chance query result (3 seeds, 1024/512):**

| seed | decay head | spike-sum head | **spike-sum ridge** | bank5 head |
|---|---:|---:|---:|---:|
| 95001 | 0.5430 | 0.4980 | **0.5469** | 0.5566 |
| 95007 | 0.5605 | 0.5156 | **0.4941** | 0.5625 |
| 95013 | 0.5762 | 0.5781 | **0.5664** | 0.5508 |
| mean | 0.5599 | 0.5306 | **0.5358** | 0.5566 |

**The offline ridge is also at chance (0.5358).** This is the most important
number in the round and it is recorded here, before any hypothesis test, because
it already constrains which hypotheses remain live.

## 1. Why this generator is hard — stated before measuring, not after

```
pair = (stored, 1 - stored)          # the two values are COMPLEMENTARY
label = pair[query] = stored XOR query
```

Two consequences:

1. **The stored information is exactly ONE bit.** "A=0,B=1" and "A=1,B=0" are the
   only two fact patterns. There is no independent pair of values to store.
2. **The label is `stored XOR query`, an order-3 interaction** of (key marker,
   value marker, query). My R2A branch already established that an additive point
   neuron plus a linear readout cannot express order-3, and that this is exactly
   why `strict_binding` failed (order-1 0.47–0.52, order-2 0.48–0.52, order-3
   1.00).

So a near-chance score here is **predicted by the generator's algebra**, not
evidence about storage or learning. **H1 as literally stated ("the network fails to
preserve the association") cannot be distinguished from "the task is order-3 and a
first-order readout cannot express it" without a decisive test**, which is what
R5 and the oracle comparisons are for.

## 2. Hypothesis predictions — fixed now

| id | hypothesis | pre-query value decode (Q A) | post-query decode (Q B) | online head | distinguishing test |
|---|---|---|---|---|---|
| **H1** | storage failure | **chance** | chance | chance | R1/R2 probes on pre-query state both at chance |
| **H2** | query integration failure | **above chance** | chance | chance | pre-query probe succeeds, post-query probe fails on the same episodes |
| **H3** | readout failure | any | **above chance** | chance | ridge succeeds, head fails on identical samples |
| **H4** | shortcut | any | any | any | query-only / first-key / position controls match the claimed learner |

**Explicit falsifiers:**

- H1 is refuted if **any** pre-query representation decodes the stored value above
  chance on held-out data with a shuffled-label control near chance.
- H2 is refuted if the post-query representation decodes the requested value above
  chance while the pre-query one does not distinguish queries (i.e. post-query
  gains come from the query, not from pre-existing variance).
- H3 is refuted if the ridge on the **same** samples as the head also fails.
- H4 is refuted if the controls fall to chance while the learner is above it.

**A crucial asymmetry I commit to now:** because the stored information is one bit
and the label is its XOR with the query, **Q A and Q B are the same quantity up to
the query**. Decoding "the value of key A" and "the requested value" both reduce to
decoding `stored`, XOR-ed with a known cue. Therefore H1 vs H2 can be separated
cleanly (pre-query `stored` decodable or not), but H3 must be tested with the
**query-conditioned** representation, not with the marginal features.

## 3. Representations

| id | content | role |
|---|---|---|
| **R0** | raw input oracle | task-well-specified check only, never an answer |
| **R1** | pre-query state at t=9: spikes, voltage, adaptation, features | diagnostic |
| **R2** | full spike trajectory (12×48) | information accessibility only |
| **R3** | post-query state at t=10: spikes, voltage, adaptation, features | diagnostic |
| **R4** | final features: decay025, spikesum, bank5 | production comparison |
| **R5** | bounded query-conditioned bilinear map (fixed random projection × query cue) | interaction diagnostic |

R5 is a **diagnostic**, not a candidate learner: the projection is fixed at a
registered seed, never trained. It exists to answer "would an interaction make the
stored value readable?" without claiming that SCAR-X learned anything.

## 4. Three decoding questions

- **Q A:** decode the value bound to each key from R1/R2 (pre-query). Report key A
  and key B separately.
- **Q B:** decode the query-selected value from R3/R4/R5 on paired counterfactuals.
- **Q C:** train the registered R2A delta-rule head on the **same** frozen
  representation, same samples, same budget. Report train and held-out accuracy
  separately, plus **paired consistency** = fraction of counterfactual pairs where
  the two predictions differ in the required direction.

## 5. Controls (H4)

1. query-only features → must be ≈ chance.
2. first-key-only policy → must not exceed the balanced baseline.
3. key-only (write positions, no values) → must be ≈ chance.
4. position-only (tick indices only) → must be ≈ chance.
5. label-shuffle control on every probe.
6. majority-class baseline.

## 6. Data, seeds, splits

| role | offset | count |
|---|---|---|
| train | `+200 000` | 1024 |
| dev/test | `+300 000` | 512 |

- **Dev seeds:** `95001, 95007, 95013` — fresh, independent of R2 (81001) and R3
  (91001) sets.
- **Confirmation seeds:** `96001, 96007, 96013, 96019, 96023, 96029, 96031, 96037`
  — 8 fresh, only if Stage 3 GO.
- Every dataset carries a sha256 recorded in the run manifest.
- Counterfactual siblings are kept in the **same** split (protocol §4 of the brief).

## 7. Gates

**Stage 1 has no GO gate** — it is a diagnostic whose output is the
location-of-failure table.

**Stage 2** registers at most **one** learned candidate, and only if Stage 1 shows
a supported information path (some representation above chance on Q A or Q B with
controls at chance).

**Stage 3 screen GO** (fixed now, not adjusted later):

| gate | threshold |
|---|---|
| G1 | direct query accuracy ≥ **0.70** on balanced held-out pairs |
| G2 | paired counterfactual consistency ≥ **0.60**, with its null measured |
| G3 | ≥ **+10pp** over the strongest matched online baseline |
| G4 | paired bootstrap CI95% **lower > 0** on the primary improvement |
| G5 | no regression > **3pp** on pre-registered controls |
| G6 | all leakage / oracle / parity checks PASS |

**Stage 4 confirmation** only if all of Stage 3 passes: 8 fresh seeds, bootstrapping
over **independent units** (the episode pair, not the two siblings), plus temporal
shifts, reversed write order, query swaps, distractors and held-out noise.

## 8. Stop rules

- If **no** representation in R0–R5 decodes above chance with controls at chance →
  **STOP**, documented as a representation bottleneck, and **no candidate is
  invented**.
- If a representation decodes but the single registered candidate fails G1–G6 →
  **STOP**, with the readout bottleneck documented.
- At most one candidate; no second candidate to rescue the first; no changing the
  generator after scored results; no uncontrolled parameter sweep.

## 9. Budget

Screen ≤ **600 s** wall clock, one thread. Peak process-tree RSS and commit
reported separately. Memory target ≤ 2 000 000 000 B committed.

## 10. Claim scope

Only: *whether the frozen ALIF state at H=48 contains the key/value information
needed for a later query-conditioned answer, and where it becomes unusable.*

**Not claimed:** chat, vision, reasoning, retention, P2, or that any substrate
should enter production. A ridge is a linear diagnostic, **not** an information
ceiling. `strict_binding`-style order-3 tasks remain subject to the R2A
function-class finding, which this protocol tests rather than assumes.
