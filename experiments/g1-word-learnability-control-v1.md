# Isolated matched ALIF word learnability control v1

Registered2026-11-10 BEFORE implementation/performance/quality measurements.
Master§4.1/report26 explicitly proposes this diagnostic. Main ALIF/locallearning
architecture, word-score STOPv1/v2, P1/P2gates remain unchanged. No Transformer,
LLMproxy, pretrained weights, new localcandidate or servingimport. All backward
implementation belongs ONLY to evaluation/controls.py, isolated childprocess.

Question: under current fixed forward/input/head/data, does a stronger global/
temporal credit signal show learnability that local score did not? A negative
result does NOT prove any local method or SNN impossible. A positive control
does NOT fulfill SCAR-X's locallearning requirement or permit BPTTdeployment.

Paired MODEL seeds28001/28019/28027, SAME as wordinputv2, not new confirmation.
Two PREDEFINED encoders:legacy and known-digest128-v1 (not posthoc sparsewinner).
Six surrogateBPTTtrials,1024responseupdates each. Reuse six ACTUAL frozenhidden
v2trials/checkpoints/outputs only after frozencontrol parity passes (no wasteful
rerun to make UIbusy). No LR/optimizer/epoch/encoder/surrogate/grid selection.
Same181train/31dev/prior48 inspected diagnostics,8knownstyleresponses,train-only
vocab/IDF; no private/live weights, freshfinaltest or general language claim.

Forward authoritative simulation is UNCHANGED NumPy ScoreWord/MaskedScoreCore:
same initialseed,H48/K8/R12,coefficients,128unlabeledcalibration,MC8 decoder,
wordmask,12zeroboundaryticks,3ticks per previousword/control/promptprefix.
Same sampler seed+10200000,1024responses,calibrationseed+10100000,fullprefix
resets and independentnoise draws from core.rng. Four rotating blocks, two
trajectories per block, SAME hiddenupdate timing (weights change betweenblocks),
head delta once on mean8features. Current target never appears in input.

For each pair record observed NumPy spike tapes. Isolated Torch graph replays
the same analog recurrences with those authoritative binary spikes; forward
mask is forced, not separately resampled. Surrogate backward for a spike:
p=sigmoid(clip((v-threshold)/.4,-20,20)); dz/dv=p(1-p)/.4 inside unclippedregion,
dz/dthreshold=-dz/dv. This is a biased surrogate, NOT exact gradient through
Bernoulli sampling. Backward transports taskW_out error through all neuron/time
dependencies, including recurrence and adaptive threshold. Not locallearning.
Graph analogstate parity atol3e-5, featureparity1e-6; no tolerancerelaxation.
The conceptual surrogate framework is discussed in Neftci/Mostafa/Zenke2019,
https://arxiv.org/abs/1901.09948 ; this logistic/taped control is our explicit
adaptation, not a reproduction or their empirical/energy/efficiency claim.

Hidden loss=mean individual-path outputNLL of the pair, SAME perpathobjective
as score hiddenreward. Not NLL of meanfeatures. Constants: SAME current
normalized/clippedheadweights/bias/mask. Compute negative taskloss gradient,
take only chosen12rows and call existing per-synapse Adam update,LR.002,
moments.9/.999,eps1e-8,updateclip.025,weightclip3. No added normclip. Four
hiddenupdates/word,matching rows/optimizer/clocks; output-local delta lr.08
unchanged. Gradient estimator/information differs; equal trainingexamples/
updates/NumPyforward is NOT equalcompute: graph recomputes8trajectories and
performs4backwardpasses per word, plus retains NumPy eligibility overhead.
Report these extra counts/time/peakmemory, no resource superiority claim.

PRETRAINING preflight: fixed seed29011,64trainingresponses only for timing
(not dev/priorperformance), same2encoders; six frozenbaseline configurations
must match NumPy64responseprefix fullbytes, allarray/counter/RNG/head/calib.
Check local surrogate derivative, sourceforward parity, hidden changes,
no-target-leak, control-onlyformat rejection by productionloaders, fullstate
continuation and untouchedservingcorehashes. Coldprocess continuation is also
checked for all6finalartifacts. Torch never stays in pytest/mainserverprocess.
Control checkpoint has own wrapperformat, contains explicitBPTTprovenance and
full reference/moment/RNG/counterstate; cannot directly load as SCAR-X model.

Timing forecast from64responseprefight BEFORE longruns:6×estimatedtrialtime
(including calibration/evaluation), plus measuredheadroom, must fit6000s.
If estimatedtrial>900s,process-treecommit/RSS>2GB,NaN,parityfailure or model/
format/source mismatch: stop before formaltraining; optimize implementation
only without changing forward/objective/optimizer, then reverify parity. No
quality-based budget extension. Formalcaps900s/trial,6000s/round,CPU1thread,
ONE Torchchild at a time,2GB process-tree monitored plus Windowsprocesscap.

Evaluate final1024only with exact SAME v2NumPyMC8 decoder/seeds (no control
Torchdecoder/beam/goldprefix at inference). Train16/dev31/prior48 strict and
normalized freeutterance,teacher-token/NLL,firsttoken,EOS,ticks/resources.
Six frozenmetrics must reference original records rather than newtest scores.
Report paired surrogate-vs-frozen and surrogate-vs-localfull seedbootstrapCI95,
3clusters exploratory. No rewritten priorresults or selection on finaltest.

Descriptive interpretation per predefined encoder, no deploymentgate:
* mean train16normalized≥80%, devnormalized≥40%, paireddevgainvsfrozen≥10pp,
  positive onALL3seeds: CONTROL_LIMITED_LEARNABILITY_MAIN_UNPROVEN.
* train16≥80% but not above: CONTROL_FITS_TRAIN_GENERALIZATION_UNRESOLVED.
* otherwise: CONTROL_LEARNABILITY_NOT_DEMONSTRATED_IN_BUDGET.
Absolute/gain cutoffs diagnostic only; cannot replace P1representation,
P2retention or normalchat/vision criteria. Mainlocalfamily remainsSTOP even
if control wins. Keep all actual wrongoutputs/losses/ΔW in passive localhost,
clearly marked ISOLATED BPTT CONTROL, never report them as currentSCAR-X replies.
