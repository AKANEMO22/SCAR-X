# ARGUMENT AUDIT — Round 4A

Compact claim ledger. Every row cites an executed artifact. `claim -> predicted
evidence -> observed evidence -> alternative explanation -> counter-test ->
conclusion`.

Artifacts:
- `artifacts/deepseek-r4/stage0-audit.json`
- `runs/20261111T043311Z-r4-information-path-v1-051593/result.json`
- `artifacts/deepseek-r4/stage1-controls.json`

---

## A. Prior hypotheses and their falsifiers (registered before measuring)

| id | prediction if true | registered falsifier |
|---|---|---|
| H1 storage failure | pre-query decode at chance | any pre-query representation above chance on held-out |
| H2 query integration failure | pre-query ok, post-query fails | post-query above chance while pre-query fails |
| H3 readout failure | ridge ok, head fails on same samples | ridge also fails |
| H4 shortcut | controls match the learner | controls at chance while learner above |

## B. Claim ledger

### C1 — Pre-query state does not carry the value linearly

- **Predicted:** R1 ridge ≈ chance.
- **Observed:** R1 ridge **0.4720**, shuffled 0.4837 (3 seeds).
- **Alternative:** the value *is* present but distributed in time, so an
  instantaneous snapshot misses it.
- **Counter-test:** R2, the full 576-dim spike trajectory, **0.5521** (shuffled
  0.4857) — above chance but weak.
- **Conclusion:** supported for R1 specifically. It is **not** evidence of storage
  failure, because R2 shows marginal information and C6 below shows the information
  is fully recoverable.

### C2 — The query transforms the state into an answer-specific representation

- **Predicted (H2):** post-query decoding fails.
- **Observed:** R3 **0.7910** vs R1 **0.4720**. **+31.9pp**, same episodes, same
  probe, shuffled 0.4850.
- **Alternative 1:** one more tick of integration, not the query, produced the gain.
- **Counter-test:** C3 below zeroes the query tick. Post-state then falls to
  **0.4668** — i.e. with the query removed there is no gain at all.
- **Conclusion:** the query, not elapsed time, produces the representation change.
  **H2 is refuted; the opposite of its prediction holds.**

### C3 — The established un-conditioned features discard the query's work

- **Predicted (H3):** ridge succeeds, online head fails.
- **Observed:** ridge on `decay025` 0.5801 / `spikesum` 0.5358 / `bank5` 0.5651;
  online heads 0.5599 / 0.5306 / 0.5566; head train 0.5859 / 0.5618 / 0.6022.
- **Alternative:** the head is undertrained rather than incapable.
- **Counter-test:** head **training** accuracy is also near chance
  (0.5618–0.6022), so the head never fits even the training split. That is an
  expressivity limit, not a budget limit.
- **Conclusion:** **H3 confirmed** for these three substrates. They fail because
  each is a marginal summary and cannot represent a query×value interaction.

### C4 — Width is not the explanation

- **Predicted:** if the gain were capacity, a wide random projection of the same
  features would match R5.
- **Observed:** C1, projecting `decay025`/`spikesum`/`bank5`/post-state to the
  **same 258 dims** with **no interaction**:
  ridge 0.5807 / 0.5358 / 0.5586 / **0.7637**; heads 0.5169 / 0.5293 / 0.5501 /
  **0.5508**.
- **R5 with interaction at the same width:** ridge **0.9928**, head **0.9316**.
- **Alternative:** the random projection happened to be unlucky.
- **Counter-test:** the *post-state* projection (the strongest no-interaction
  control) reaches 0.7637 ridge but its **head is 0.5508** — so even the best
  no-interaction representation leaves the online learner at chance.
- **Conclusion:** **the interaction, not the width, carries the result.** +22.9pp
  ridge and **+38.1pp head** over the best matched-width control.

### C5 — The cue is necessary and load-bearing

- **Predicted:** if R5 ignores the cue, shuffling it should not matter.
- **Observed:** R5 with shuffled cue (train and test) **0.7708**; R5 trained on the
  true cue but evaluated with a shuffled cue **0.6992**; R5 true cue **0.9928**.
- **Reading:** the 0.7708 floor equals the cue-free post-state information
  (0.7637), so shuffling the cue destroys **exactly** the interaction benefit and
  leaves the marginal state. The 0.9928 − 0.7708 = **+22.2pp** is the cue's
  contribution.
- **Conclusion:** the cue is necessary. This also shows my own R5 number is not an
  artifact of width: with the cue broken it falls to the no-interaction ceiling.

### C6 — The information is genuinely STORED, and a supplied cue recovers it

This is the decisive control and it refutes H1 outright.

- **Construction:** zero the query tick (t=10) entirely, rerun the population,
  capture the t=10 state, then apply R5 using the **true cue from the input**.
- **Observed:** query-zeroed post-state alone (ridge) **0.4668** — chance.
  Query-zeroed state **+ true cue**, through the same bilinear map: **0.9961**.
- **Alternative:** the 0.9961 comes from the cue alone.
- **Counter-test:** the query-only control is **0.4948**, and C3 shows the
  query-zeroed state alone is 0.4668. Neither component alone is informative; the
  combination is 0.9961.
- **Conclusion:** **the association IS stored in the ALIF state**, and it is fully
  recoverable when an explicit cue is combined with it. **H1 is refuted.** The
  network's own query tick is a second, weaker route to the same information
  (R3 0.7910).

### C7 — No shortcut explains the result

- **Observed controls:** query-only 0.4948, key-only 0.5104, position-only 0.4987,
  first-key policy 0.5000, majority 0.5104.
- **Strongest anti-shortcut evidence:** on **counterfactual pairs that differ only
  at the query tick**, R5 paired consistency is **0.9935** against a measured null
  of **0.5000**. A shortcut cannot distinguish two episodes that are byte-identical
  everywhere else.
- **Conclusion:** **H4 refuted.**

## C. Competing explanations, each with a rejecting test

| apparent gain | alternative 1 | alternative 2 | test that rejects it |
|---|---|---|---|
| R3 0.7910 > R1 0.4720 | more ticks integrated | query tick adds variance | C3 query-zeroed → 0.4668 rejects both |
| R5 0.9928 | broader feature space | fixed projection is lucky | C1 matched 258-dim → 0.7637 rejects width; C5 shuffled cue → 0.7708 rejects luck |
| paired consistency 0.9935 | pairs leak label | pairs share noise | Stage 0 verified siblings differ **only** at tick 10 |
| head 0.9316 | head memorised | task is trivial | head train 0.9528 with test 0.9316, and all un-conditioned heads ~0.53 |

## D. Counterexamples where my strongest claim fails

1. **R2 at 0.5521** is above chance pre-query. My C1 statement "pre-query state
   does not carry the value" is therefore **true only for the instantaneous
   snapshot R1**, not for the trajectory. I state it as such rather than widening it.
2. **C5's 0.7708** is a legitimate counterexample to "you need R5 to decode": a
   cue-free post-state projection already reaches 0.7637 ridge. R5's advantage is
   real but it is **+22pp**, not "+from chance".
3. **The matched-width head 0.5508** shows that width alone does not help the
   *head* even though it helps the *ridge* (0.7637). Ridge and head diverge here,
   which is exactly the R2A lesson that a capable reader does not imply a
   learnable one.

## E. Causal versus correlational

- **C3 (query zeroing) is causal**: it removes the factor and the effect vanishes
  (0.7910 → 0.4668). It does not merely correlate.
- **C5 (cue shuffling at test) is causal for the cue**: breaking the cue at
  evaluation drops 0.9928 → 0.6992 on a fixed trained readout.
- **C1 and C4 are correlational in construction** (different feature sets) but they
  are *capacity-controlled*, which is what makes them interpretable.
- **C6 is causal in the strong direction**: it shows the information survives
  removal of the network's own query, so the storage claim does not depend on the
  query being present.

## F. Limitations — confirmed versus speculative

**Confirmed by execution:**
- The task is well specified (oracle 128/128), the generator is unmodified (Grok
  blob `874fb835…`), and the forward pass matches `SpikingCore.step` exactly.
- R3 > R1 and the effect is causal (C3).
- R5 ≫ every matched-width control, for the head as well as the ridge (C4).
- The information is stored and recoverable with an explicit cue (C6, 0.9961).
- No listed shortcut explains any of it (C7).

**Speculative / not established:**
- **That SCAR-X learned anything.** R5's projection `P` is **fixed random**, never
  trained. Only the final 2×258 readout learns. R5 is a diagnostic of
  representational sufficiency.
- **That the cue must be supplied externally in production.** The brief permits an
  explicit legal cue for R5; the network's own query path gives 0.7910, not 0.9928.
- **Generalisation beyond this generator.** One task, three dev seeds, no
  confirmation.
- **A "local" learner in the biological sense.** R5's bilinear map is a fixed
  nonlinearity followed by a delta rule; it is local in its weight update, but the
  interaction is hand-specified, not learned.

## G. Final argument — why STOP, and what is GO

The registered Stage 1 gate asked for a supported information path before any
candidate could be registered. The path **is** supported, and more strongly than
predicted: the information is stored (C6), the query acts causally (C3), and a
bounded interaction makes it readable to the online head (C4, 0.9316).

I nonetheless report **STOP for the un-conditioned readouts and no candidate**,
because:

1. R5 already answers the scientific question. Registering a trainer on top would
   blur "the representation suffices" with "a learner works", which is precisely the
   confusion the brief forbids.
2. One generator and three dev seeds cannot support a GO. The confirmation tiers
   (8 seeds, temporal shift, reversed order, query swap, distractors) were not run
   because no candidate was registered.
3. The largest remaining ambiguity — whether a *learned* interaction can be
   discovered rather than supplied — is a new question, not a gate I can relax here.

**The scientifically justified conclusion:** the frozen ALIF state genuinely
contains the key/value information after the query, and even **before** the query
in a form recoverable with an explicit cue (0.9961). The established readouts fail
because they are marginal summaries that cannot represent a query×value
interaction. A bounded fixed interaction makes it learnable by the registered local
delta rule at **0.9316**. **The bottleneck is the feature map's function class, not
storage, not query integration, and not credit assignment.**
