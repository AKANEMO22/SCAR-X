# P1 second/final local-target-family adjustment: random error projection

Registered2026-11-09 AFTER v1 failed its development gate. V1 raw results stay.
Same masterP1 constraints, same48/K8/R12 and ALIF/synaptic eligibility equations.
No serving deployment. This is the final registered adjustment in this family;
if this screen fails, stop this family rather than tune a third round.

## Single change and locality audit

V1 used neuron rate matching `m_j=C[j,y]-f_j`. V2 instead uses a fixed random
projection of the current output categorical error:

`m_j = C[j,y] - sum_k C[j,k]*p_k`.

C remains fixed balanced0.02/0.35 code, NEVER copied/trained from W_out. The
existing local forward eligibility and weight-update equation stay identical.
Readout uses its own normal output-local delta. Error-vector information is
now broadcast nonlocally through a fixed projection; disclose this explicitly.
No output-weight transpose, hidden-to-hidden error propagation, reverse-time
graph, autograd or whole-network gradient. This is a random-feedback three-factor
candidate inspired by e-prop, NOT fully local in the sense of no teacher/error
broadcast, not exact e-prop/BPTT or a faithful paper reproduction.

Target y is available only during learning; inference receives input alone.
Shuffled control replaces y in the HIDDEN modulator with independent random
label, while readout keeps correct y. Test exact modulator math, fixed-code
invariance and checkpoint/RNG continuation before the screen.

## Locked comparisons / gate

New independent DEVELOPMENT seeds2203,2213,2237; confirmation seeds2707,2711,
2729,2731,2741,2749,2753,2767 if and only if screen passes. Generator seed offsets,
2048train/128probe-fit/256development/128OOD, tasks order/XOR/strict-binding,
config and two LR0.002/0.02 match v1. Readout-only/scalar-STDP baselines0.002,
random-label feedback control per LR;2workers, CPU1thread each, aggregate2GB,
180s/trial,1200s/round. Fresh initializations, not old release adaptation.

Select LR only by mean development probe gain; freeze before confirmation.
Require >=5ppgain over initial AND direct>=65% on at least2tasks. Order baseline
was93.75% in v1, a near-ceiling diagnostic: report ceiling/headroom honestly;
failure there alone is not evidence of a weak representation. No easier task,
new threshold or test selection substituted after looking at v2.

If qualified, identical8seed confirmation,512eval, seed-bootstrap95%/signflip/
Holm and original P1 gates (lowerCIgain>=5pp, direct>=65%, advantage over readout
positive and adjusted significance on>=2tasks). Existing sources at
https://www.nature.com/articles/s41467-020-17236-y and
https://arxiv.org/abs/1909.01311 are context, not performance evidence for this
custom candidate. No retention or normal-chat/vision claim from P1 passing.
