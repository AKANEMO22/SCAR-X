# RC-D3: learning-signal forensic, data-only

**VERIFIED arithmetic consistency; causal failure attribution NOT MEASURED.**
Recomputed 4,608 saved updates from all24 P1b cells, with no neural forward,
training, probe fit, or TEST access. Original **G2 FAIL remains unchanged**.
Ponytail tools were unavailable in the session. This is an independent
arithmetic implementation authored by the same Codex system, not external
peer review or reproduced training.

Protocol and analyzer were committed before execution at
`bee9b68c530d174f1bbc28d42cee5e11aebc81a9`. Immutable source checkout:
`1ea065be194d44000ba795138c5917f3d13adda0`; historical P1b registration source:
`a0f5ac4c1a146bd07a554c6ed32919dc8a0ed8fb`. The source checkout and historical
registration commits describe different stages and are not interchangeable.

Run: [`20261116T-d3-saved-signal-v1-bee9b68c-01`](../../runs/20261116T-d3-saved-signal-v1-bee9b68c-01/).
It finished at2026-11-15T19:17:03Z (16November Asia/Saigon). There are3 model
seeds,24 arm-runs and192 updates/cell, not24 independent seed replications.
The original run and its source snapshots were read without modification.

## What the saved signals show

The table pools3 seeds ×192 sequential updates per sign/arm. Cosine excludes
undefined cases; `n` reports its actual denominator. Sign conflict is the
mean per-update fraction among jointly active coordinates, with absolute
activity threshold1e-12. These are descriptive summaries, not confidence
intervals or independent observations.

| Sign/arm | Mean cos(label,hint), n | Mean joint-active sign conflict | Mean label L2 | Mean hint L2 | Mean physical delta L2 |
|---|---:|---:|---:|---:|---:|
| + teacher B | +0.4322,181 |12.58%|0.1648|0.3060|0.02689|
| − teacher B | −0.1255,218 |62.39%|0.09552|0.4127|0.02689|
| + shuffled C | +0.1166,576 |47.45%|0.1630|0.4567|0.02933|
| − shuffled C | −0.07541,576 |54.29%|0.08396|0.5552|0.02961|

The sign association is strongest on positive training examples. For B, mean
label/hint cosine is+0.7098 at+ initialization (70 defined updates) versus
−0.2366 at− initialization (85 defined). Positive-example jointly active
sign conflict is0.868% versus66.81%. This measures conflicting saved local
signals; it does not establish that conflict causes final failure.

B's hint-bank norm is <=1e-12 on395/576 updates at+ initialization and358/576
at− initialization. C has a nonzero bank on all576 updates for each sign.
Therefore conditional cosine and its valid count must be read together;
unconditional median B hint/label norm ratio is0. Bank sparsity is observed
after the eligibility/modulation sum; missing time-series diagnostics do not
allow attribution to one tick or neuron-state mechanism.

| Seed | Sign | B DEV accuracy / positive recall | B mean cos(label,hint), n | C DEV accuracy / positive recall |
|---|---|---:|---:|---:|
|261161|+|87.50% /50.00%|+0.4164,56|90.625% /62.50%|
|261167|+|91.40625% /65.625%|+0.4222,68|84.375% /37.50%|
|261173|+|92.18750% /68.75%|+0.4596,57|87.50% /50.00%|
|261161|−|75.00% /0%|−0.1214,81|75.00% /0%|
|261167|−|75.00% /0%|+0.01917,69|75.00% /0%|
|261173|−|75.00% /0%|−0.2771,68|75.00% /0%|

The slightly positive mean cosine in one failed−seed prevents treating the
pooled sign association as a sufficient explanation. Recomputed final DEV
metrics match original completed records. Label-only and readout-only retain
75% accuracy at both signs. B's final learned detector readout contrast
`mean(Wout[1]-Wout[0], units8:36)` is positive even for all failed−seeds:
0.09178/0.14192/0.12898. “The head stayed negative” is contradicted for this
detector summary, while other decoder/representation issues remain possible.

## Eligibility and the actual optimizer pipeline

Reconstructed label credit uses the saved **pre-update** head, saved
probability/error, clipped endpoint q and endpoint eligibility. It is added to
the saved hint bank only for B/C; stable rows0:8 remain masked. The analyzer
then evaluates the registered momentum, squared-direction energy normalizer,
scalar step clip and weight-bound equations against saved states and physical
weight differences. No learner was imported, instantiated or rolled forward.

All4,608 update checks passed with tolerance1e-5, including learning CSV
norms/loss, head/bias updates, final optimizer states and frozen coordinates.

| Maximum absolute discrepancy | Value |
|---|---:|
|Reconstructed direction|3.3022e-8|
|Momentum state|1.9372e-8|
|Energy state|1.0563e-9|
|Physical hidden delta|3.0101e-8|
|Next saved hidden weights|2.3284e-10|
|Head update|7.9758e-9|
|Bias update|3.0995e-8|
|CSV norms/loss|2.1104e-7|
|Saved softmax probability|1.6336e-7|

Endpoint eligibility is nonzero at every B/C update; mean L2 is approximately
0.3786 for B and0.3807–0.3809 for C at both signs. This does not show an
endpoint eligibility collapse. It does not prove correct temporal credit or
exclude selective loss of useful information.

Physical update direction often differs from current combined direction:
B mean scalar sign disagreement is32.71% at+ and31.58% at−; whole-vector
cosine is negative on172/576 and171/576 updates respectively. Momentum and
normalization history are therefore materially different from raw credit,
but this statistic does not separate successful+ from failed−training.
Readout-only has exactly zero physical hidden update and retains optimizer
state; its hypothetical proposal metrics are not applied updates.

Scalar clipping of normalized hidden proposals occurs on88/576 B updates and
93/576 C updates at each sign. All such updates lie in the preregistered
first48-update window; later windows have no scalar clipping.

| Sign/arm | Mean clipped fraction among active scalar proposals, all updates | Same metric, updates1:48 |
|---|---:|---:|
|+ B|5.97%|23.87%|
|− B|7.82%|31.27%|
|+ C|4.17%|16.70%|
|− C|6.03%|24.13%|

Hidden weight-bound interventions and occupancy are0 throughout. Across all
cells max trained-coordinate absolute weight is0.56285, far below3. Endpoint
feature clipping fraction is0 throughout, as are head scalar clipping and
head weight-bound interventions. These saved endpoint/weight boundaries
therefore do not account for the observed sign failure. Full-episode student
feature saturation remains **NOT MEASURED** because intermediate q was not
saved. Endpoint q alone cannot rule it out.

In the registered one-proposal omission calculation, removing H while holding
the same saved pre-state fixed changes B physical delta by mean symmetric
relative L2 of11.70% at+ and21.62% at− (median0 for both). For C this is30.25%
and36.57% (median23.13%/30.96%). This demonstrates a nonzero instantaneous
arithmetic influence of the hint on some actual proposals after optimizer
history. It does **not** estimate the counterfactual no-hint training
trajectory: no weights were written and no subsequent state was evaluated.

## Correct versus shuffled teacher after the whole pipeline

At update1, matched B/C cells have exactly identical saved weights,
eligibility and label components. Their physical deltas already differ:
mean symmetric relative L2 is26.41% across3 seeds, at either sign; cosine is
0.9546. The relative difference denominator is the mean of the two norms;
this is not the historical preflight metric or a replacement for it.

Across all updates B/C median physical relative difference is87.62% at+ and
84.49% at−, with median physical cosine0.6559/0.6767. Median pre-update weight
relative differences are67.65%/69.50%; median label-component relative
differences20.43%/63.84%. Eligibility vectors remain close in angle (mean
cosine0.99969 at either sign), but are not identical. Once trajectories
diverge, these comparisons combine hint choice, hidden state, readout and
optimizer history. They cannot isolate sample alignment causally.

The saved controlled study already shows B−C DEV gains of−3.125/+7.03125/
+4.68750pp at+ initialization and0pp for every−seed. D3 establishes that
correct/shuffled hints remain physically distinguishable after the pipeline;
it does not show a robust useful advantage. It preserves the original
minimum-recall and per-cell-gain failures.

## Artifacts, resources and next discriminating evidence

The single analyzer execution took1.9174s internally and2.1733s shell wall.
Sampled process peak RSS61,177,856bytes; Windows OS peak working set
70,696,960bytes; one BLAS/OpenMP thread, no GPU. RSS was sampled, not protected
by a hard RSS cap. Both measured peaks are below the registered300MB budget;
execution is far below60s. Registration time precedes metrics. Resource
receipt elapsed time is sampled before its own final serialization/hash/status
bookkeeping; the shell wall includes that bookkeeping and interpreter startup.

- [Locked protocol](../../experiments/rc-d3-signal-forensic-v1.md) and
  [analyzer](../../scripts/research/rc_d3_signal_forensic.py).
- [Registration/config/source hashes](../../runs/20261116T-d3-saved-signal-v1-bee9b68c-01/registration.json),
  [input hashes](../../runs/20261116T-d3-saved-signal-v1-bee9b68c-01/input-hashes.json).
- [All4,608 update metrics](../../runs/20261116T-d3-saved-signal-v1-bee9b68c-01/updates.csv),
  [1,152 B/C paired updates](../../runs/20261116T-d3-saved-signal-v1-bee9b68c-01/teacher-shuffled-paired-updates.csv).
- [Full per-cell/class/window metrics and audit](../../runs/20261116T-d3-saved-signal-v1-bee9b68c-01/metrics.json),
  [resource receipt](../../runs/20261116T-d3-saved-signal-v1-bee9b68c-01/resource.json),
  [atomic terminal status](../../runs/20261116T-d3-saved-signal-v1-bee9b68c-01/status.json),
  [output hashes](../../runs/20261116T-d3-saved-signal-v1-bee9b68c-01/output-hashes.json).

Config SHA256:`5e8317fa7a1ede32e36ea3b464f3d66808984a7817e9a8b1f3c2ad146ecc4609`.
Input manifest SHA256:`5df732a4626d073d0e0bf336865b5e8e5e48535dc826b43c2ea30155644b5fed`.
Script SHA256:`cc33cd01f6939431f4669ce19bb044048a39622d4bc52914cae42d22e3b41fe3`.
Protocol SHA256:`784cd0e5ddeec92e5970c2aca44fcee82dd8cfddfd28d2209dc6961ff17661ee`.

These arrays reject a missing-physical-update explanation and do not support
hidden bound or endpoint saturation as the cause. They associate−initialization
with greater label/hint conflict, but do not causally separate decoder quality,
hidden information and optimizer coordination. D1/D2 checkpoint diagnostics
are the lower-cost next discriminators. A new learner or optimizer intervention
would need a separate registered experiment; D3 did not run one. There is no
evidence here of reusable composition, continual learning, or an advantage
over a small LLM plus RAG and skill library.
