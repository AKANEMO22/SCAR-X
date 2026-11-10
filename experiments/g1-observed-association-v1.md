# Observed-response covariance association v1

Registered2026-11-10 BEFORE implementation/training/performance measurements.
Master§4.1 representation, §4.2 Experience-to-Skill and §4.4 activity-based
learning/adaptive learning threshold. New bounded mechanism hypothesis, not a
rename of stopped score/localclassifier/target-rate/byte/decoder families.
Those STOPdecisions/protocols stay immutable. No scorevariance/baseline/LR/grid
adjustment, Transformer/pretrained proxy, backward/BPTT or hiddenlossgradient.
Production core/weights/format/Hermes never change in this pilot.

Hypothesis: explicit observed question→answer experiences can bind a decaying
presynaptic questiontrace to postsynaptic activity induced by the observed
answerwords, creating useful input→hiddenrepresentation with less reliance on
scalar stochasticreward. This is LABEL-CONDITIONED hetero-associative Hebbian
learning, NOT unsupervised/independentfactverification/brainnovelty. Teacher
answer is ordinary input only during a separate training-observation phase.
No current/gold target is allowed in prediction/probe/freegeneration input.

Different from closed LocalTargetCore: no predefined neuron/class target-rate
table, rate-matching error, surrogate derivative, localclassifier, fixedfeedback
matrix, scalar reward×likelihoodscore, or headW_out transport. Postsynaptic
activity comes from actual ALIF during observedwordstream. Functionally the
observedanswer is still privileged label-dependent teaching; do not conceal
that relationship or claim learning without externally supplied answers.
No claim of biological fidelity or novelty from assembling known components.
Covariance/threshold principles: Neuronal Dynamics19.2
https://neuronaldynamics.epfl.ch/online/Ch19.S2.html . Our temporal pairing and
signed numeric frontend are an explicit adaptation, not a paper reproduction.

One DETERMINISTIC SpikingCore ALIF with existing G1 forward (not stochastic
ScoreWord), H48/K8/R12. D=V+128+64+6,O=V+10; fixedtrain-targetvocab/digitreserved.
Core init seed, subtract.15 fromw_in, stablecopy. leak.65,threshold.5,
adaptstrength.02,adaptdecay.95,fdecay.25; ordinary3ticks/word and12zeroboundary.
All64visionchannels/digitoutputs remainuntrained, no new vision claim.
Fixed128densewordcode and IDF from ALL181trainprompts ONLY as priorv2; no
newencoderselection. No new neurons/parameters/recurrence/topology or optimizer.
Word/recurrent/control inputweights remainfixed. ONLY incoming questioncontext
w_in[:,V:V+128] is plastic in paired/cross-family branch. This is real hidden
synapse learning, but not evidence recurrent synapses learned relation skills.

Per trainingexample, same actual computation for all FOUR conditions:
1. Validate prompt/answer vocabulary/length BEFORE state mutation. Canonical
   reset neuronstate,12zeroevents, then3ticksBOS+1.6promptcontext with existing
   G1languagecontrol. For each querytick, e=.9e+(1.6context-mu_query), e starts0.
   mu_query is128inputmeans, initially0, updated .99mu+.01*(1.6context) once
   AFTER this observation episode; no dev/probe data updates means.
2. No neuron reset between query and observedanswer. Feed TRUE observedcurrent
   answerword IDs via ordinarywordinputgain.8, contextZERO. Exactly7slots×3ticks
   (maximum fixed8style target inclEOS); shorter answers endEOS then zeroinput
   padding. Slotcontrols existingG1wordcontrol for observedtoken,allzero forpad.
   This is observation, NOT prediction or a returned assistantreply. At every
   tick decayquestiontrace e*=.9, no newquestioninput. At slotend r=filtered
   ALIFspikes; use previous mu_post (48values,initial0), then .99mu+.01*r.
   Select rows ((slot_clock%4)*12)+[0..11], incrementclock allconditions.
   Paired update for contextsynapse only:
       delta_ij=clip(.02*(r_i-mu_post_i)*e_j,-.025,.025)
       w_ij=clip(w_ij+delta_ij,-3,3).
   No readoutupdate in observation, no derivatives/reward/Adam/anchors. Means
   are adaptive activityreferences, NOT importance/consolidatedknowledge.
3. Canonical reset neuronstate again,12zeroevents; teacher-forced output-local
   headlearning on same prompt with PREVIOUS answerwords only,3ticks/position.
   Currentword target only in local outputdelta lr.08,clip.025/weight3.
   Head uses rawfeatures like G1, no new normalization/calibration. Hidden
   weights and activitymeanstates FROZEN during this prediction-learning phase.
   Thus observationactivations cannot leak into head's predictionfeatures.

Conditions fixed before measurements, 3freshmodelseeds31013/31019/31033:
* frozen: simulate all phases/means/clock but no hiddenupdates.
* paired: correct observedanswer plus local covarianceupdates.
* cross_family: observedanswer of NEXT fixedINTENTS family cyclically (fixed
  prototypeanswer, not sampled by performance), while outputhead target stays
  the true answer. ALL21observationticks/7opportunities unchanged, even pads.
* erase_trace: correct observation but set e=0 before observedanswer; no hidden
  update possible. Must match frozen weights/predictions exactly given seed.
No condition/encoder/LR/epochs selection. Balanced8family sampler seed+11200000,
1024experiences each, actual AGY117+authored64TRAIN,31dev,48prior inspected.
Samples/publicsource SHA pinned, targets styleexamples not automatically facts.
No userfeedback/privateweights or ownunverifiedanswers. No new AGY/Qwen calls
because same actualsource95references already available, no missingdata.

Inference: canonicalreset+12zeroboundary then ordinaryG1wordsteps; only prompt
and previously NEURAL-generatedwords, max12words, same masksBOS/EOS/punct asG1.
No templates/intentlookup/retrieval/goldprefix/observationphase at inference.
Oldcontinuousstate app isn't changed; no automaticmigration or thinkingclaim.
Evaluate only FINAL1024checkpoint, not pick bestepoch/seed. Train16(2/family),
dev31/prior48 freeutterance strict/normalizedstyle,teacher-tokenNLL/accuracy,
firsttoken/EOS/ticks; actualloss,netΔW/changes and observationupdatecounts.
Common exactorder perseed, deterministic forward, same fixedcompute across
conditions. Compare to own matchedfrozen/crossfamily, not claim algorithmgain
vs old stochasticScoreWord with differentforward/head/objective/budget.

Representation diagnostic: canonicalBOS prompt-only48features after final
hiddenfreeze; fit SAME8classridge.01 using all181TRAIN only, compare dev31 to
fresh initial/frozen and crossfamily features. Head never selects templates or
answers; no hiddentraining from probelabels, no labelleakyteacherfeatureview.
Paired seedbootstrapCI95 descriptive3models, not originalP1confirmation/gate.

Informative ONLY if pairedmeanDEVfree normalized≥40%, meanpaired-vs-frozen
DEVgain≥10pp positive onALL3seeds, priorgain≥10pp, paired-vs-crossfamilyDEVgain
≥5pp positive onALL3, and meanprompt-onlyprobeDEVgainvsfrozen≥5pp positive on
ALL3. Technicalchecks/no-target-input/codec exact allpass. These descriptive
screencriteria neverreplace mainP1/P2/normalchat/vision/retention thresholds.
Positive=OBSERVED_ASSOCIATION_FRESH_VALIDATION_REQUIRED, no deployment/scale.
Negative=STOP_OBSERVED_ASSOCIATION_V1; no posthoc claiming train16ceiling,
ΔW/nonzeroactivity, or readout/retrievalgain is hiddenrepresentation success.
At most2registeredrounds perhypothesis per master; THIS protocol allowsONE
fixedscreen only, no automaticround2/LR/encoder/epoch/scale tune from failure.

Checkpoint distinctformat,≤3MBnonpickle/checksums, includes coreallarrays,
means/e/slotclock,outputmask,condition,vocabulary,encoding,allRNG/counters.
Beforetraining: initialhardforwardparity toG1, no-labelprediction path,
inputblocklocality/frozen/erasedtraceexact, observation-headseparation,
handcalculateddelta, corruptionbounds/RNG/coldcontinuation. Fullpytest/native
parity. CPU1thread/worker,2workers,2GBtree,120s/trial/1800sround,12trials.
StopNaN/malformedsource/persistence/leakage/locality/budget. Publish actual
answers/loss/ΔW/neurontrace to passive localhost with clearlyexperimental
observational-learning label. No training-like event for synthetic/mockdata.
