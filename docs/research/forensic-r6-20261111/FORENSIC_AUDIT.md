# SCAR-X independent forensic audit — P0 to P3 + Gemini supplement

**Verdict: PIVOT / no learner promotion.** R6D readout data effects and R6B's
negative candidate screen reproduce. Gemini CKCER has strong fixed-hidden
retrieval performance, but its10/10 preregistered-GO interpretation is not
supported by execution history/protocol fidelity. No new architecture is justified
by those labels alone. Main and live model remain unchanged.

## What has actually been audited

- Full user handoff read and archived byte-exact with custody hashes; JSON values
  independently compared to Git/source/raw rows, not treated as an oracle.
-19branch snapshots with full SHA, protocol/run/manifests/source.zip comparison;
  new Gemini8f24a5a separately pinned after it appeared.
- Independent event/pair evaluator, typed complete-gate aggregation and mutation
  tests against wrong tick, labels, sibling IDs/noise, raw/scaled features,
  no-op shift, direct-for-both_A, unsupported gateTrue/missingG8, changed order,
  episode leakage and duplicate/shadowed source.
- R6D80row matrix +8row controls exact replay; original R5B/R6B metrics/screen
  exact replay; source/scaler/order/head digests retained. Prediction arrays can
  be regraded without original scorer.
- Actual NoClip/CKCER engineering checks; Gemini344pair-score groups independently
  recomputed. Numerical discrepancies and failed resource runs preserved.

## Priority findings

| ID | Severity | Claim / issue | Evidence | Verdict |
|---|---|---|---|---|
| F01 | P1 | R6D all gates computed | matrix:153–189 + original/replay nulls | Contradicted; incomplete aggregation |
| F02 | P1 | Controls are the same trained head | different order/head hashes, same source/features/init | NON_EQUIVALENT_TRAINED_HEAD |
| F03 | P1 | R6D stress is query-only counterfactual | actual paired noise differs in all temporal/nearly all distractor pairs | INVALID_CAUSAL_PAIRING |
| F04 | P2 | R5B-style equals original R5B pipeline | source of stats/ddof differs; max fixture delta.72048 | False equivalence |
| F05 | P2 | G7/G8 fully established by tests | one-cell repeat; no original step-parity test; limited new parity fixture | Scope missing/partial |
| F06 | P2 | Larger sample count is only changed causal factor | non-nested sampling, order varies,8logical states | Readout effect real; broad attribution limited |
| G01 | P2 | Grok toy tests check real NoClip | imported class shadowed by local class | Coverage gap, corrected in audit |
| G02 | P2 | Grok runner has unique definitions |3top-level functions shadowed | Effective source recorded/normalized audit copy |
| G03 | P2 | Grok screened every registered stress | shift/reverse/distractor/noise absent | NOT_TESTED; STOP unchanged |
| C01 | P1 | Gemini confirmation is untouched | first STOP, protected code/baselines changed, same seeds rescored | Corrected exploratory replay, not pristine GO |
| C02 | P1 | Gemini executed its registered stress design | train augmentation/noise/ticks differ | Protocol deviation |
| C03 | P1 | Gemini ablations isolate a single factor | hard perceptron vs softmax-delta mismatch | Causal attribution confounded |
| C04 | P1 | R6C demonstrates hidden synapse learning | frozen recurrent weights; Oja in head | Not tested for recurrent hidden learning |
| C05 | P2 | Branch isolation eliminates all interference | inactive weights0; shared-bias score drift.025 | Overclaim |
| C06 | P1 | R5 shared trunk cannot affect query margin | actual post-query state differs; unit-readout delta1.80248 | Missing theorem condition |

Detailed source anchors, raw counterexamples and limits are in
R6D_SOURCE_AUDIT.md, R5B_R6B_REPRODUCTION.md and REVIEW_GEMINI_8f24a5a.md.

## What did hidden synapses learn?

For **R6D/R6B/R6C**, recurrent hidden synapses did not learn: they were frozen.
Their state can encode a small set of event facts; trained readouts learn how to
decode them. R6C adds a learned query gate outside the recurrent weights. This is
useful associative readout research, not evidence of newly acquired hidden skill.

Do not erase narrower positive evidence elsewhere: main paired-score20 reports
held-out refit gains on order/XOR, and DC-ALIF round1 has task-scoped learned-
hidden gains. Those are different learners/tasks and have not established robust
transfer or retention. They do not justify claiming either “all local learning
works” or “hidden synapses never learn”.

## Main/product recommendation

Candidate main learner changes: **none approved**. Recommend review of reusable
research infrastructure (independent scoring, source/data/scaler/order identity,
null-safe gates, negative regression tests). Keep CKCER's pre-gating normalization,
temporal summary and neural query gate as research hypotheses, with shared-bias
and matched-optimizer controls. Do not import them into G1 without fresh approved
research and checkpoint/product acceptance.

Existing product is a local stateful prototype; strong synthetic two-bit retrieval
does not establish normal dialogue, natural vision, autonomous reasoning or
lifelong learning. The finite-domain production task list is in
CHATGPT_WEB_REVIEW_BRIEF_VI.md; no percent-AGI or delivery-date estimate is inferred.
