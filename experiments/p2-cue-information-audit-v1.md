# P2 cue information diagnostic — posthoc, no new candidate

Registered2026-11-09 before these measurements, after completed P2diagnostic
showed poor offlinejoint/refit accuracy. No new primary endpoint, training,
modelselection, hidden-rule tuning, changed task or gate. Same fixed existing
3seeds×offline_joint_full/frozen checkpoints=6models; no cherry-picked winner.

Measure stored information at prefixes2,6,12ticks, using a fresh4classridge
diagnostic head with exactly same3840train and20×128heldoutitems as source.
Representation from each immutable model,8vs64independent MonteCarlofeature
rollouts. 64 is EXTRA inference compute for a diagnostic, not a production
method or equal-budget gain. Use bounded16rolloutbatches with disjoint seed
offsets; all conditions use same prefixnoise schedule where applicable.
No labels in forwardrepresentation or modelweights/calibration/RNG.

Input information controls: local ridge on (a) raw first2cue averages + whole
context averages (24features) and (b) their outerproduct(4×20=80features).
Labels from trainonly. This is an engineered conjunctiveinformationcontrol,
NOT a neural algorithm, equal-capacitycomparison or generalization claim.
Purpose: unambiguous inputs may contain association information even if the
current neuralfeature/noise/memory doesn't make it linearlydecodable.

Dataset/mapping/sourcecheckpoint unchanged. Report all means/predictions and
perseed effects, same65%reference only as diagnostic context, not a new GO gate.
Interpret larger64rollout scores as sampling/noise evidence; early-vs-late
differences as time-dependent decodability, not an isolated causal neuron
mechanism or proof of unlimited retention. No testlabel selection/refitting
into main core or runtime. assert model.dumps unchanged throughout.

CPU1thread,sequentialmodels,2GBcap,180s/model,900sround. Stop on NaN/state
mutation/resourcefailure. Publish actual progress; no fake learningloss or
model/Qwen answer. Source run/records kept immutable, output new unique run.
