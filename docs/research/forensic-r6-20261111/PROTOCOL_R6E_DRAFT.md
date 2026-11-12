# R6E DRAFT — readout geometry, data support and interference

**PENDING USER APPROVAL. NOT RUN.** This is a proposal, not a retroactive fix to
R6D/R6C. No candidate implementation or new scored seed is authorized here.

## Competing explanations

- H-data: broader sampled support improves a fixed local head; normalization or
  hidden learning is unnecessary for the observed16K gain.
- H-routing: standardize-before-gating and separate query branches reduce a
  specific interference path, beyond a matched change in feature geometry.
- H-optimizer: reported routing benefit is materially explained by unmatched
  perceptron/softmax, bias, update norm, calibration, or presentation order.

No R6E result can establish learned recurrent hidden representation: hidden is
frozen in every R6E arm. A mixture of these effects is allowed; no forced winner.

## Registration and data

Pin final source/protocol/generator and independent evaluator hashes before runs.
Provisional development seeds171301/171307/171313; provisional confirmation
171401/171407/171413/171419/171423/171429/171431/171437. Before approval/locking,
scan all newly published branches, protocols and raw result seed/dataset lineage.
These values have not been generated/scored by Codex. Replace only if documented
collision occurs **before** results; never claim freshness from constants alone.

One fixed parent pool16,384perseed; small subsets256/1024 are registered nested
indices. Independent pinned256calibration pool. Held1024IID +256counterfactual
groups use independent pool IDs and sample hashes. Record logical state coverage
separately from noisy sample count. Do not add more neurons, facts or data budget.

Transform the same source siblings with common padding/distractor/noise draws.
Check oracle, query location, vA/vB, group identity, non-query equality and actual
feature change on **every scored row/group**. Broken stress becomes
INVALID_HARNESS, not poor intelligence. Clean→stress zero-shot is primary;
train-augmented→stress, if approved, is separately labeled and never substituted.

## Bounded comparisons

1. Reproduce original R5B fulltrain stats/sample-ddof convention as its own arm.
2. Pinned calibration with R5B-style floor/clip.
3. Same pinned calibration with R6A floor/unclipped convention.
4.1K/16Kone-pass ladder and256×64vs16K×1matched-update contrasts under each.
5. CKCER temporal summary, pre/post-gating normalization and shared-trunk/bias
   ablations use **the same softmax delta rule**, identical initialization where
   shapes permit, actual update counts and matched applied-norm diagnostics.
6. A fixed second order checks path sensitivity on the primary scaler; no search
   over additional orders, projections, floors or learning rates after scoring.
7. Baselines: first-key, query-only, shuffled training labels, no learning, true-
   cue and neural-query-gate controls. Ridge uses exactly the same feature tensor
   and is diagnostic only. Report all parameters, calibration state and gate cost.

The full executable cell list must be reduced/locked to at most12candidate
configurations perseed before approval; no exhaustive factorial sweep is implied.
Include required negative controls in the resource estimate. Initial code review
must confirm that the proposed bounded design separates the factors it claims.

## Primary decisions (to be locked unchanged before any new output)

- Primary metric both_A; direct/both_B/needB separate. Report perseed and paired
  seed bootstrap10,000/RNG2101. Correct multiple primary contrasts with Holm.05.
- Useful readout finding: clean both_A≥.75 and primary matched-control improvement
  ≥5pp with lower95%>0, positive on at least7/8confirmation seeds.
- Robustness: **same clean-trained head** both_A≥.75 on each registered stress
  condition mean, every seed≥.65; both_B≥.90. These are new proposed thresholds,
  not changed historical gates. No condition may be dropped after results.
- Protected-task regression≤1pp on the exact registered same-run baseline and
  generator. These independently fit controls do not count as continual retention.
- Missing gate/identity/scaler mismatch/label leak blocks globalPASS. A valid
  failed gate yieldsSTOP; source/metric corruption yieldsINVALID_HARNESS.
- Screen3seeds then one untouched8seed confirmation only if screen passes. One
  revision after a registered failure requires its own approval; no unlimited tuning.

## Resources and outputs

H48, fan-in8, fixed CPU one numerical thread. Process-tree cap2GB; checkpoint/
snapshot buffers counted. Based on the audited R6D matrix157.5s and CKCER replay
43.4s on this machine, budget≤30minutes per stage, subject to a pre-run timing
estimate on unscored fixture data. Stop on1200s per bounded job/2GB/error.
No API/paid service, language pretraining or serving integration.

Save source.zip, actual thread-pool metadata, all per-sample predictions/oracle
metadata, train/calib/test/group/scaler/order/head digests, seed-level metrics,
resource logs, all failed/superseded runs and final gate mapping. Use a new sealed
run for any corrected replay; it is never renamed untouched confirmation.
