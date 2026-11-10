# Isolated ALIF visual-word learnability control v1

Registered2026-11-10 BEFORE implementation/timing/quality. Master§4.1/report32
permit this separate control. NOT reopening STOP_VISUAL_WORD_READOUT_V1 or
proposal30, not production/Transformer/pretrained/proxy/scaling/local-rule tuning.

Question: can the SAME48ALIF/input/head fit image-question words with isolated
surrogate-BPTT hidden credit? This diagnoses control learnability, not mainlocal.

## Locked data, forward and matched baseline

Pin datasetSHA0fb04abefe67451952a2d73634fded9c86243d5874356d68299ff047deecf6a2
from artifacts/visual-word-dataset-latest.json:192ownedPNG,864train/864devQA,
16train/16dev questionwordings from32actualAGYquestions, same novelcompositions.
Renderer targets ONLYsupervisedlabels, never inputs. No newteacher/model/API.
ExistingQwen3-4Btext-only is NOT visionbaseline; no image-as-text substitute.

Pairedseeds33113/33119/33133, FULLcondition only,1536experiences each. Reuse
actual frozen full/no_image/no_question results from
runs/20261109T222354Z-visual-word-readout-v1-a11424, never retrain/selectthem.
Samebalanced kind/language/target/case sampler seed+12100000;128unlabeled
calibration seed+12200000. Compare sampler, targetwordupdates and NumPyticks.
Dev already inspected: exploratory, NOT freshfinaltest. No qualityselection.

Unchanged evaluation/visual_words.py authoritativeforward:48ALIF/K8/R12,
leak.65/threshold.5/adaptation.02/.95/features.25, D=V+198/O=V+10, w_in offset-.15.
Prepare33ticks, mean16RGBpatch neuralfeatures cache48; eachword3ticks,
.5*(cache+currentfeatures), same questionhash, inputflags/coords, wordmask8max.
Previousgoldword duringtraining ONLY; previousneuralword at inference. No
intent/class/language parser, targetmask, metadata, retrieval or othermodel.

Wordhead remains sequential output-local normalized delta.08/clip.025/weight3.
Snapshot PRE-local-update weights/bias for eachword hiddenloss. Calibration
mean/scale frozen. No autograd headoptimization or alternative head schedule.

## Isolated hidden update, fixed before any training

Only evaluation/controls.py imports/uses Torch backward. Record authoritative
NumPy input/spike/state tape perresponse; Torch replays samehardobservedspikes
in forward. Existing WordTapeSurrogate derivative p*(1-p)/.4 with
p=sigmoid(clamp((v-limit)/.4,-20,20)), zerooutside rawabs<20. Backward through
recurrentedges/reset/adaptation/features/visualcache is GLOBAL/TEMPORAL,
biased surrogateBPTT, NOT exact hardspikegradient or local/successofproposal30.

Loss=meanword -log(max(softmax(maskedlogits)[gold],1e-12)), using each pre-head-
update snapshot. Abort if maxstateerror>1e-4, featureerror>1e-6 or losserror>1e-4
against actualNumPytape. Targets only in supervisedloss, not neuralforward.
One hiddenupdate at ENDresponse, ALL48rows (not matched localrowupdatecost):
Adam lr.001, beta1.9/beta2.999/eps1e-8, globalgradientL2clip1, resultingdelta
perentryclip.002, weightclip3. Moments0, persistentoptimizerstep. Stablearrays
unchanged; no replay/consolidation/headinjection/reinitialization/parametersearch.
Earlyzerograd measured, never repaired by randomizinghead. Allweights CONTROLonly.

Separate bounded checksummed nonpickle wrapper: reference + Adamstate + counters/
grapherrorstatistics. Production/VisualWords loaders reject wrapper. Save/load
workingstate, optimizer/RNG and freshprocess nextlearn/freeanswer exactbytes.
No control/error/tape/weights import into main; no proposal30 approval inferred.

## Preflight and resource budget

One preflight, no dev qualityscoring: for3pairedseeds compare initialhidden/
indices/calibration to source32; recording-only wrapper(no hiddenupdates) must
match64TRAINresponses/reference losses/fullbytes. Never modify VisualWords to
makeparitypass. Verify graphstate/feature/loss parity, finitegradients, code/
protocol/datahashes and codec/resume. Run targetedtests/fullpytest/nativeparity.

One training-only64response timingcopy seed34103, fixedrule. Forecasttrial=
1.25*(trainingseconds/64*1536 + setupseconds +40evaluationseconds). Proceed ONLY
if <1500s/trial and3*forecast+90<6000sround, no2GBtreebreach. Preflightcap300s.
OneCPUthread/oneTorchprocess atatime. Timing is not model/hyperparameterselection.

Formal3sequentialtrials,1500s/trial/6000sround INCLUDING evaluation/coldverify.
2GB/process Windowscap whereavailable AND process-tree sampling. Disclose graph
ticks/backwardpasses, optimizerbytes, time/RAM; sameexperiences != samecompute.
Exactly1536experiences, evaluate FINALcheckpoint ONLY. No intermediatequalitypeek.
Technicalfailure stops/preserveslogs; only diagnosed non-neural IO/reportrepair
with unchangedmath/protocol/data and prioractivebudget accounted. Timeout never
automatically means processdead or permission torestart.

## Measurements and interpretation locked before results

Evaluate ALL864train and864dev, freegeneration<=8words; macro overkind/language,
perkind/language, novel/seencompositions, strictstyle, EOS, teacherNLL/accuracy,
actualticks/time. Causalblank_image/blank_question/swapped_image EXACTLYprotocol32;
interventionassignment targets never enter inference. Reusedfrozenmissingmodality
comparisons are context, NOT matchedBPTTmissingmodality controls.
Paired95%seedbootstrap interval() as32;3models !=864independentseeds. No broadclaim.

* CONTROL_JOINT_LEARNABILITY_SUPPORTED only if EVERYseed devmacro>=65%, EACHkind
  >=50%, novelmacro>=60%, mean gainvsfrozenFULL>=20pp, allgainspositive/lower95>0,
  mean drop for EACHcausalintervention>=20pp. Still NOT main/local/normalvision.
* CONTROL_FITS_TRAIN_GENERALIZATION_UNRESOLVED if EACHseed fullTRAINmacro>=65%
  but precedingjoint/devcriteriafail.
* CONTROL_LEARNABILITY_NOT_DEMONSTRATED_IN_BUDGET otherwise. Not proofALIF/local
  impossible, not justification tochange mainarchitecture. No postresulttuning.

P1/P2retention/representationgates unchanged. Mainnormalchat/vision/actions/
lifelong andproposal30 remainunproven/pending even ifcontrolpasses.
Actual loss/ΔW/changedhidden/headsynapses/neuronframes/image-query-words-target
are passivelocalhost with CONTROL ONLY/BPTT/NOT MAIN labels. Complete864devcases
includingerrors automaticallyshown; no selector/runbutton/mockoutputs/GETlearning.
Mainrelease/privateweights/topology/vocab/source unchanged. Historicalreports intact.
