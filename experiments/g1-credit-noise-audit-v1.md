# Read-only conditional credit-signal noise audit

Registered2026-11-10 BEFORE signal measurements/code. Master§4.1 credit and
representation diagnosis after isolatedcontrol27. Not a localrule adjustment,
newtraining, baseline selection, encoder/epoch/modelsize search or reopening
STOPword-scorev1/v2/localclassifier. Main ALIF/locallearning/CPU2GB unchanged.

Question: are existing local rawscore pair directions reproducible at fixed
word-learning inputs, compared with the biased surrogate control at the SAME
weights/tapes? A noisy estimator may explain poor sample efficiency, but this
audit alone cannot establish task-performance causality or propose deployment.
Surrogate gradient is NOT exact Bernoulli gradient/groundtruth direction.

Sources: ALL3 pairedseeds28001/28019/28027, denseinput only, THREE fixed states:
frozenhidden finalv2, full local-score finalv2, isolatedBPTT fullfinalcontrol27.
Nine immutable artifacts, no private/liveweights. This is not freshmodels or
independentconfirmation. Across states heads/weights differ, so do not assign
differences solely to learningrules; withinstate compare signals samepoint.

Eight cases: FIRST authoredTRAINprompt per fixed INTENTS family, with its
existing styletarget. Two teaching positions percase: firstword(BOSprefix) and
EOS(after true trainingresponseprefix). EOS uses goldpreviouswords intentionally,
never describes freegeneration/semanticreasoning. No dev/prior/finaltest access.
16conditionalinputs/source=144configurations; nestedcases are NOT144seeds.
Rows12 selected by ((familyindex+positionindex)%4)*12+[0..11], independent
of quality/gradient. All48rows covered over balancedfamilies/positions, but
this is oneblock perinput, not a full trainingstep/Adamupdate.

For each conditionalinput at fixedweights,128 independent noisepairs using
external RNG(seed+9120000+familyindex*1000+positionindex*100). Sameuniforms
across sources, own trajectory recurrenthistory. No checkpoint/core RNGdraws.
Exactly256actual NumPyALIFrollouts, no learning. For eachpath get finalfeatures,
currenthead log p(target)=R, and original forward localeligibility e on12rows.
Three predefined signaldiagnostics:
1. existing-score: .5*((R1-bEMA)*e1+(R2-bEMA)*e2), bEMA fixed from checkpoint,
   not updated by this read-onlyaudit.
2. crossfit-baseline-score: sameformula, b is mean of ALLpathrewards in the
   OTHER independent64-pairhalf. No withinpair/selfbaseline and no targeterror
   vector transported to hidden. Analysis variancecontrol only, not a proposed
   learner/optimalbaseline/newscore tuning or evidence overcoming oldSTOP.
3. biased-surrogate: existing evaluation/controls.py WordLearnabilityControl
   _gradient on samepair,tape,head,rows. No Adam/weight/headupdate; globalcredit
   remains ISOLATED CONTROL. Counts actualgraph/backwardcompute separately.

Flatten ONLY incominginput/recurrent signals on selected12rows; track blocks
separately too. Save raw128pairs arrays for oneinput at a time, fullaggregate
metrics for allinputs (not every highdimensional vector indefinitely).
For each method calculate64pairhalfmeans muA/muB and samplevariance trace vA/vB:
crossmeanenergy=dot(muA,muB) (maynegative, NEVER clamp into a confident signal);
noiseenergy=.5*(sumvarA+sumvarB). Ratio=crossmeanenergy/noiseenergy if >0noise.
Replicatecosine=dot(muA,muB)/(normA*normB), missing if zeronorm. This is split
reproducibility proxy, not SNRgroundtruth; finite128MonteCarlo uncertain.
Report per-case128pairmean norm, varianceenergy, rewardmean/std, storedbaseline
gap, cosinebetween128pairmeans ofscore andsurrogate (not exactalignment).
Report same estimates for input and recurrentblocks, missing/nonfiniteexplicit.

Primary descriptive contrasts perstate/modelseed: median crossfit/existing
noiseenergyratio and median surrogate-minus-existing replicatecosine over16
inputs; then seedbootstrapCI95 across3models/state. Individualratios where
existingnoise0 missing, not fabricated0/inf. No success threshold/selection/
training from these metrics. If crossfit lowersvariance, cannot conclude
performanceimprovement; if surrogate morestable, cannot prove it is unbiased.
No broad inference from 8knownstylecases or positive changedweights.

Integrity: sourcefileSHA and weights/moments/head/calibration/rowclocks/update
counters/coreRNG/referencevocab/encoding remainunchanged. Simulatedneuronstate/
eligibility live only in discardedclones; explicitly not identicalfullworking
state during sampling. Sourcecheckpoint bytes never overwritten. Frozenmode
may still compute signals but cannotlearn. Before audit tests fixeduniform
localpair signal equals sourceformula, identical surrogatealreadyparitychecked,
splitstatisticzero/negative/missing behavior and sourceweights/RNGunchanged.

One CPUthread/ONE Torchprocess,≤2GBmonitored/per-processcap,1200sroundcap.
144configs×128pairs; forecast from firstpredefinedconfiguration before rest,
not based on labelperformance. If forecast>1200s or RAM>2GB stop, no fewerpairs/
cases or budgetextension afterobserving signals. Stop any NaN/format/source/
math/parity violation. Publish actualprogress and technicalevidence on passive
localhost, not learningevents/fullanswers (nonearegeneratedorlearned).
Keep rawrun/source/protocol immutable. No newAGY/Qwencalls because no missing
data/reference for this signalquestion; existing actualcorpus/reference retained.

Mathreference only: Williams1992 doi10.1007/BF00992696 and Gerstner etal,
https://neuronaldynamics.epfl.ch/online/Ch19.S4.html . This audit measures our
implementation; it does not reproduce paperquality/energy/brain claims.
