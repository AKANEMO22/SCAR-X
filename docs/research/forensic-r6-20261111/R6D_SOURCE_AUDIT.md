# R6D source audit — pinned 4bacd7b

Source: `4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893`, protocol
`experiments/collab/scarx-neural-round1-20261111/deepseek-r6d/protocol-v1.md`.
This report preserves all original results and interprets them separately.

## Source custody

R6D matrix/controls source.zip files match every hash in their manifests. Their
text differs from Git blobs primarily by CRLF; normalized text matches pinned
HEAD. The earlier invalid R5B run has substantive generator/runner differences;
the corrected R5B run `...064932...bef298` matches current pinned source. Full
checks are in BASELINE_INVENTORY.json; source.zip is evidence, not an empty badge.

## Confirmed code findings

| Finding | Source anchor | Independent evidence / verdict |
|---|---|---|
| G4/G5/G8 null, no final confirmation_pass | `scripts/r6d_matrix.py:153–189` | Exact replay preserves nulls; FAIL G6 plus INCOMPLETE_AGGREGATION |
| Grid/controls/stress use different orders | matrix:73,88,101; controls:37–45 | Same train features/labels/head init, different order and final head digests on8seeds; NON_EQUIVALENT_TRAINED_HEAD |
| R5B-style is not original R5B calibration | `r6d_audit.py:61–70,97–105`; `r5_probe.py:115–125` | Separate256calib+ddof0 vs fulltrain-runningstats+ddof1; fixture max scaled-feature difference.7204816 |
| Mean gates, not all-seed gates | matrix:154–159; protocol:82–89 | Historical mean clean/temporal pass; worst clean.70068 and temporal.71667. Do not retroactively change historical rule |
| stress_valid checks another128rows | matrix:37–44 vs105–115 | New tap validates exact1024held/512paired rows each stress/seed; row oracles remain correct |
| Same-noise sibling contract is broken | `r5_independent.py:246–248,278–280` | Temporal256/256pairs differ; distractor252–255/256differ eachseed. Invalid query-only causal interpretation |
| G7 repeats only first cell | matrix:163–169 | A repeat of one seed/condition; cannot certify all combinations/checkpoint restart |
| G8 not actually in dedicated test suite | `tests/test_r6d_audit.py` | No SpikingCore step parity test exists there. Independent8sequence fixture: spike/feature0error, voltage2.384e−7; record tolerance/scope, never fill full G8 from grep |
| Typo phase opens confirmation | matrix:51–53, controls:26–27 | Any non-replay string uses confirmation bank. Audit CLI instead has explicit choices |
| Non-nested train ladder | `r5_independent.py:60–72` | Actual8seed1Kset is not subset of16Kset; DATA_LINEAGE.json records overlap and8logical states |
| raw feature API still callable | `r6d_audit.py:84–94` | Documentation convention, not an enforced data type. Audit has tagged feature/scaler contract and negative tests |

Original tests pass15/15. They remain useful unit tests, but their calibration
loop never exercises a budget-changing caller, their stress check uses clean
features, and their gate-presence loop omitsG8. Green tests did not rule out the
observed defects. Audit mutation tests exercise actual contrary behavior.

## Reproduction and statistics

- Matrix replay80source trial rows + summary exact; controls replay8rows+summary exact.
- Independently saved predictions/labels/facts/query/pair IDs match the source
  pair metrics. No agent summary was used as a scoring oracle.
-1K→16K R5B-style gain+.23053784, paired seed bootstrap95% [.17161145,.29482062].
-Matched-update unique−repeated+.15075840, CI [.09684684,.22234585].
-Preprocessing×data interaction−.05007063, CI [−.11551755,.00801609]: not robust
  evidence for an interaction. Source rounds seed effects before its interaction
  bootstrap; audit keeps full precision.
-Recorded distractor both_A.55023045. Even supplementing controls does not turn
  the failed recorded G6 into a global pass; counterfactual stress validity is
  separately limited by changed sibling noise.
-No raw episode overlap in train/calib/held/paired at the inspected sizes/seeds.
  This does not prove absence of unpublished probes, nor novelty of logical facts.

## Attribution

`P.capture()` runs deterministic frozen ALIF weights. `A.train_head()` creates
only `A.Head` and updates its readoutW/b. The large gain concerns an engineered
post-state+query bilinear representation and online readout fitting. It does not
show hidden synapses learned representation, robust general reasoning, language,
knowledge accumulation or continual learning. The positive readout data effect
is real in this finite generator; its broad causal interpretation remains limited
by sampling/order/calibration conventions and missing stress isolation.

Machine-readable results: R6D_CLAIM_RECOMPUTE.json and R6D_GATE_INTEGRATION.json.
No source or historical output in the author branch was edited.
