# Recorded action policy reset diagnostic — locked before inference

Following targetattribution70f370: do100updates make hiddennecessary for the
20actual verifydecisionstates, beyond changing readout/statistics? This is
retrospective inference on previouslyseenstates, no new train/dev/testdataset,
confirmation, learner/rule tuning, optimization or deployment. One existing
seed260113, ALIFH128/D64/FP32 and existing localforward equations unchanged.
All neural calculations CUDA0, native VLM/ASR untouched. No new live weights.

Exactly20new-experience verifydecisions from completed554c19, joined by
decisionIDs to finalcheckpointcontroller.decisions (43entries fit64bound).
Use saved sequence and actualallowedmask, not recomputed labels/goalflags.
Lock caseIDs/sequencearrays/allowed before modelinference. Target is verify
action from independent arithmetic proof, not the calculated number. No label
or finalanswer inserted into sequence. Rawsource/replayproofmetadata pin.

Four arms, exactly80forwardattempts total maximum, no otherforward/tests:
1.initial seed260113 untrainedparameters, frozenmode;
2.final100updateparameters/statistics;
3.final with w_in/w_rec reset to initial, retain finalreadout/statistics;
4.final with w_out/bias reset to initial, retain finalhidden/statistics.
Reset only ownedclones; no learn/consolidate/gradient call. Parameterfingerprint
must match before/after inference; ephemeral membrane/trace/ticks may change.
All arms same recordedinputs/masks, reset neuronstate before eachforward as in
actual LocalALIF._forward_native. Preserve same connectivity andFP32precision.

Report actualactionprobabilities/allowedargmax, verifycounts and vectoragreement,
percase standardizedfeatures, readoutweights/bias for independentprobability
rederivation. No qualityPASS/CI significance from20relatedseenstates/one seed.
Finalweights on earlierstates include futuretrainingrelative to original
decisiontime: posthocfit only. Initialvsfinal changes hidden/readout/statistics
together; hiddenreset retainingstats is not a complete factorial causalclaim.
Equalchoices after reset refute necessity only on these inputs, not allSNNtasks.

Budget<=180s,<=80attemptedGPUforwards; count before each invocation, stop on
frozen-source/profile/state/device errors. Need>=2048MiB physicalfree before
CUDAallocation and>=512MiB throughoutsampledexecution, foregrounduserpriority.
No VLM/ASR/modelstartup/shutdown/importlivecheckpoint or change active32cd13
frozenfiles. Trace actualoutputs as oldlabdiagnostic, separate from newlearning.
Stop on failure; uniqueimmutable run/partialreceipt, no resample/refund/grid.
