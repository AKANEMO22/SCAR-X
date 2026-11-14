# Hidden representation common-probe diagnostic — before outputs,14/11/2026

Parent032f30 failed. Bindingc1f23f also shows weak counterfactual response. Do
not train a new candidate or retune the failed one. This is posthoc diagnosis
on the same owned train/finalDEV (not a fresh confirmation TEST).

For3parentseeds260113/137/171 and all4arms, load final scored checkpoints and
freeze all neural weights. Extract unstandardized core.features128 after each
original16tick sequence on320train+400finalDEV, on CUDA. Use no learned readout
logits or online f_mean/f_var in the representation probe. Fit train-only
mean/std (floor1e-3) and one identical linear ridge readout per representation:
128normalizedfeatures+intercept, targets-1/+1, lambda1 on every coefficient
including bias, decision score>0. No hyperparameter search or label access to
finalDEV during fit. This closed-form diagnostic head is not a production
learner or local-learning success; never import its weights into the SNN.

Solve129×129 SPD system on GPU in float64 using explicit conjugate gradients,
max512iterations/relative residual<=1e-9, check independently against NumPy
linear solve afterward. Original neural weights remainFP32, not compressed.
Backward/autograd/BPTT are absent. Fitting one readout cannot establish an
advantage of the original online head or end-to-end LLM+SNN.

For currenthidden and protectedhidden, additionally reset only w_in/w_rec to
that seed's initial values in private clones, preserving trained readout/state.
Re-extract features and fit the same probe. Reset-feature matrices must equal
the originalfrozen matrices exactly; readoutonly must also equal frozen. This
checks that learned readout/normalization are excluded from attribution.
Topology/indices/config remain fixed. Ephemeral forward state is allowed to
change; compare all weights/stable/statistics/importance/RNG/replay/counters
before/after each extraction. Original parent checkpoint bytes unchanged.

Archive feature matrices, probe weights/statistics, scores, source/data hashes,
solver residual and every prediction. Report train/finalDEV per-family accuracy
and learnedhidden-minus-frozen pairedseed/family95%bootstrap (seed26011499,
10000resamples). Disclose original2/400encoded aliases; a descriptive probe
gain never overrides failed65%/3-5pp/5pp original gates or implies lifelong skill.

At most12960neuralforwards (18representations×720cases),7minutes,512MiBphysical
GPUreserve checked periodically. Stop on solver/source/immutability/resource
failure and retain partial evidence. No VLM calls, consolidation, live-state
updates, active8h source changes, main/architecture modifications or promotion.
Publish measured scores on8775 as a diagnostic, not new neural learning.
