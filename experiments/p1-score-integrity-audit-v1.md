# P1 score eligibility integrity audit v1

Registered2026-11-09 before numerical measurements. Mathematical diagnostic,
not a training candidate, hidden-learning quality test or reopening the stopped
local-target-rate/random-feedback/local-classifier families. Original Phase2.1
score family remains a small-scale PIVOT with no second implementation round
yet; any future adjustment needs its own protocol, at most one final round.

Question: does a local Bernoulli score trace match the derivative of expected
reward under the specified stochastic ALIF path distribution? How do clipping
and antithetic uniforms alter bias/variance? No BPTT/autograd or model deployment.

For a fixed entire previous spike history, adaptation/reset terms are fixed;
conditional voltage derivative for incoming weight is
`q_t=leak*q_(t-1)+pre_t` (input or previous source spike).
`score_ij = sum_t (z_jt-p_jt)/temperature * q_ij,t`.
With clipped logits, multiply each contribution by indicator of unsaturated
logit. Unclipped eligibility and independent-past reward baseline give the
likelihood-ratio estimate `(R-b)*score`. This statement is about a smooth
real-arithmetic probability model, not a full floating-point optimizer/update.
Clipping trace/advantage or using Adam changes the full learning algorithm;
never call a bounded optimizer an exact unbiased network gradient.

Oracle:2ALIFneurons,D2,K1,self/cross indices fixed,6steps=4096spikepaths percase.
Manual forward float64 enumeration of all path probabilities/rewards; exact
weighted score expectation vs centered finite differences at epsilon1e-5.
Readout fixed2classes, reward=-token/classNLL of final filtered spikes; no
target in inputs. Test all6incomingweights (4input,2recurrent). Baseline-0.7.
Fixtures fixed before data: moderate alternatinginput; stronginput10x with
inverse inputweight scaling (same operating voltages but larger sensitivities);
delay configuration with initial cue and final marker. No success gate based
on easy recall or model accuracy. Oracle math tolerance maxabs1e-7 and
relative1e-5 for nonzero gradients; totalpathprobability within1e-10 of1.

Compute bias for actual traceclip10 and advantageclip2, and unsaturated-logit
derivative omission separately. Do not assume clipping bias occurs in ordinary
H48tasks merely because it can occur in a constructed fixture.

Variance audit:fixed seeds211,223,227;8192independent pairs perfixture. Compare
single rollout estimator, mean of two independent rollouts and mean of two
antithetic-uniform rollouts U/1-U with SAME prior baseline. Recurrence uses
each trajectory's own presynaptic history. No pair-dependent reward baseline
or difference-of-rewards shortcut. Compare estimator mean/error/trace-covariance
sum and per-parameter variances; report antithetic improvement or worsening,
not a guarantee. Mean of pair has marginally correct paths, so unbiased raw
expectation follows mathematically; Monte Carlo is finite-sample evidence only.

No parameter training, live-state mutation, model growth or new language corpus.
All math/MC uses finite float64 fixtures, different from productionfloat32.
If oracle fails, fix implementation before proposing training. Publish actual
progress/gradient diagnostics to localhost with diagnostic scope, no mock chat.
CPU1thread,2GBprocess-treeRAM,60s/fixture,300sround; stop on NaN/probability,
derivative consistency or resource failure. Preserve result/protocol historical.

Reference context: Williams1992REINFORCE and reward/eligibility treatment in
https://neuronaldynamics.epfl.ch/online/Ch19.S4.html . This diagnostic uses a
Bernoulli ALIF discrete-time likelihood model, not a biological proof.
