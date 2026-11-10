# P2 readout conditioning diagnostic v1

Registered2026-11-09 before measurements. Master P2 prioritizes acquisition and
readout stability alongside P1; this diagnostic isolates why a refitted probe
can decode useful fixed features while online output-local learning stays near
chance. No hidden representation improvement or continual-retention claim.

Same deterministic ALIF48/K8/R12, fresh seeds3109,3119,3137, D8/O2. Tasks
order/XOR/strict-binding from existing generators.2048train, first512used only
for unlabeled per-neuron Welford mean/variance calibration; remaining1536receive
one output-local categorical update each.256development,128refit-probe fit.
Generator offsets+1,500,000train/+1,600,000fit/+1,700,000eval. No test labels enter
calibration. All conditions have identical initial fixed hidden synapses,
features, labels/update count and initial readout weights/bias.

Conditions:
- raw: features as produced by ALIF.
- centered: presynaptic neuron subtracts its calibrated mean.
- normalized: subtract mean and divide by max(std,0.05); clip to[-5,5].

Statistics are frozen before learning/evaluation. No inverse covariance,
ridge inference, pretrained feature model, global/temporal backprop or new
neuron. Each mean/variance belongs to one presynaptic neuron; readout synapse
uses that local feature and its own postsynaptic categorical error.
Update lr0.08,per-update clipping0.025,weight clip3, no anchors/replay.
Refit ridge on raw features is a diagnostic control only, not runtime.

Report raw/condition direct accuracy and train loss; paired differences vs raw.
Screen is informative if centered/normalized gains>=10pp with direct>=65% on
at least2tasks; this is NOT the historical representation gate (hidden frozen).
No selection or deployment based on development. If informative, separately
register8fresh-seed confirmation and a P2 continual interference/retention test
before any deployment. Otherwise stop this conditioning trial. No parameter
search after results. CPU1thread,<=120s,2GBcap, no live state changes.

Preflight tests: evaluate never changes statistics/weights; stats cannot be
updated after freezing; constant-neuron features remain finite; explicit local
readout update matches math; all arrays/counters can serialize if later adopted.
