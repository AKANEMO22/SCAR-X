# Proposal30 → unchanged32 visual-word path, directeligibility bridge v1

Registered2026-11-10 BEFORE implementation/performance. Protocol35 and report36
explicitly name a separately registered visionbridge within approved30. This
is a new application of approved readouterror/own-neuronforwardeligibility, NOT
reopening32/33 or word36 by encoder/LR/epoch/cache/modelsize tuning. MainALIF/
weights/Hermes unchanged, no Transformer/pretrained/proxy/networkBPTTcandidate.
P1/P2 gates remainfailed and do not become satisfied by this applicationpilot.

## Locked source/forward

Use EXACT dataset32 SHA0fb04abefe67451952a2d73634fded9c86243d5874356d68299ff047deecf6a2:
192ownedPNG/864train864devQA,16/16actualAGYwordings withreviewsource. Targets
ownedrenderer manifest, not teacher/Qwenanswers; nolabels in image or inference.
Historicaldev/compositions have been inspected: exploratory, NOT freshfinaltest.
No newteacher call needed; existing32wordings and actual64Qwentextrefs remain.
ExistingQwen4Btext-only is NOT visionbaseline, no image-as-text substitution,
VLMdownload/9B/APIrequest or use of Qwenanswers as labels.

UNCHANGED VisualWords32forward:48ALIF/K8/R12,D=V+198/O=V+10,G1leak.65/
threshold.5/adaptstrength.02/adaptdecay.95/featuredecay.25,w_inoffset-.15;
question InputEvents128sumhash/L2,33prepareticks incl12zero/3BOS/18imageevents,
RGB16patches and48neuralfeaturecache mean. Word3ticks previousword/constant
question, feature=.5*(cachedvisual+currentfilteredspikes),8wordlimit/allvocab
mask/EOS/firstpunctuation as32. No intent/language/class/router/answertemplate.
Do NOT change mixture/inputencoding/positions/calibration to improve thisscreen.

## Approved signal mapped onto actual cached presynaptic feature

Reuse ReadoutEligibility35 UNCHANGED neuron-local sensitivity equations, logistic
psi.4, no incoming-spike derivatives, no temporalreverse/networkJacobian. Keep
own v_e/a_e/f_e forward arrays H×(D+K); sameLOCALAdam.001/.9/.999/eps1e-8,
entrydelta.002/weight3, all48rows. Stableweights unchanged, no normreduction.
OnePRE-head-update W_out.T@error step, normalizedhead strictclipderivative and
scale as35. Headoutputlocaldelta.08/clip.025/weight3; frozen128unlabeledTRAIN
calibration as32, no devstats. Never import controlweights into candidate/main.

Cacheeligibility=mean f_e over16imagecontentticks (1based17..32), obtained
whiletheSAMEneuronforward runs. Pooledeligibility=.5*(cacheeligibility+currentf_e)
at EACHwordendpoint, matching actualpresynapticfeature. Accumulate endpoint
directions, divide bynumbertargetwordsinclEOS, applyhiddenupdate ENDresponse.
No targeterrorat imageinput, no backward overpatches or wordtime. Held trace
is a bounded local sensitivity, not rawpixelretrieval/anotherneuralcore.
Changingthisderivative mapping vs numeric35 is mathematically required by
existingcache32; forwardcachepolicy/mixture remain unchanged, not a newgrid.

Threefreshmodelseeds37313/37321/37339 ×7conditions=21trials:
correct, shuffled, zero, frozen, no_image, no_question, bptt_control.
Firstfour SAMEfullinputs; missingmodalityconditions usecorrecthiddenrule but
zero the sameRGB48 or question128channels EVERYphase exactly32. Shuffled
permutesneuronL perword using existing seed+35100000, outputlabels correct;
zero computes traces but nohidden/momentupdate, frozen skips tracelearning.
Zero/frozen must yield identicalneural/headweights/answers on matchedstream.

BPTT ONLYevaluation/controls.py: same observedNumPyhardspiketape/input/cache/
headsnapshotperword, meanendpointloss, same allrowssynapseAdam/clips NOglobalnorm.
Biasedglobaltemporalgraph isCONTROL only, extra graphcompute disclosed. Actual
NumPystateparity<=1e-4/features<=1e-6/loss<=1e-4 required; not an exacthardspike
gradient or proof that candidate is fullybiologicallylocal.

All7conditions perseed same1536balancedTRAINexperiences/sampler32 seed+12100000,
same128unlabeledcalibration seed+12200000, samewordcount/ticks. No main/session
weights or labels/history used. Everycontrol isfreshrandominit, no bestseedpick.

## Engineering before formal

For eachcondition trace-disabled/zero-hiddenforward/calibration +16TRAINresponses
must match original32VisualWords weights/head/features/outputs EXACTbytes. Do
not modify32/35math or31renderer to makeparitypass. Preparedsegment/cachedtrace/
wordendpoint/synapsebank/Adam/RNG/context/vocab/fullworkingstate checkpoint with
separateboundednonpickleformat/checksums<=6MB. Loaders reject main/controlmixup.
Midimage/prepare/endword continuation exact, rollbackafterupdatefailure exact;
freshprocess nextlearn/freeanswer/RNG/fullhash proof for everyformalcheckpoint.

Independentfinite-difference of anchoredSMOOTHown-neuronsystem with fixed
presynapticimage+wordtape, matching16samplecachemean and wordendpoint mixture.
Atw0 surrogate matches hardtrajectory, not derivative of discontinuousmodel.
64synapses (4neurons×8input+8recurrent),eps1e-5,float64,atol1e-5/rtol1e-3.
No label needed for sensitivityoracle. No currenttarget in prepare/generate.
Feedback off at inference cannot change outputs withsameweights.

Preflight oneTRAIN-onlytimingseed37303,64experiences EACHcorrect/BPTT. No devscores.
Forecasttrial=1.25*(trainingseconds/64*1536+setupseconds+40evaluationseconds).
Formalallowed if eachforecast<300s and18*correctforecast+3*BPTTforecast<2400s,
no2GBtreebreach. Preflightcap180s; fullpytest/nativeparity before formalscreen.

## Measurements and fixed gate

OnlyFINAL1536checkpoint, all864TRAIN/all864DEV, actualfreewords max8, same
macro(kind,language)/perkind/lang/novelcomposition as32, teacherNLL/EOS/firstword
separate. No teacherforcing in freeoutputs. Correcttrainedhead causalblankimage/
blankquestion/swappedimage exactly32, assignments outsideinference only.

Samequalityrequirements32: EVERYseed macroDEV>=65%, EACHkind>=50%, novel>=60%,
mean correct−TRAINEDno_image and correct−TRAINEDno_question >=20pp,allgains>0/
bootstraplower95>0; mean drop for EACHcausalblank/swap>=20pp. Added contribution:
mean correct−frozenfreeDEV>=10pp,allgains>0/lower95>0; report shuffled/zero and
restorehiddeninit lesion. No success from100%recall/trainfit or changedweights.

ResetheadBOSprobe on SAME48pooledactualfeature, fixedridgealpha1/floor.05/
trainonlynormalization, exact1536trainexposures, same as diagnostic36. Report
firstwordprobe trained−initialhidden mean/lower95, not wholeutterance orP1pass.
No alternateviews/probeselection/epoch/headgrid. Hiddenrepresentation benefit
unproven unless this probe gain>=5pp/lower95>0, still onlyexploratory3seeds.
Positive=FRESH_DIRECT_VISUAL_VALIDATION_REQUIRED, NOT normalphotos/main/P1/P2;
failure=STOP_DIRECT_VISUAL_BRIDGE_V1. No tuning/rerun/scale/autoapplicationv2.

## Resources / trace / integrity

CPU1thread/worker,max2NumPyworkers,oneTorchchild atatime after NumPyworkers exit;
2GBprocesshardcap whereavailable +treeRSS/commit sampling.300s/trial/2400sround
includingevaluation/coldproof. Includecore/head/trace/cache/Adam/checkpoint/rollback
copies, no offload/iGPU or energyclaims. TechnicalIO/reportrepair only diagnosed
non-neuralfailure with unchangedmath/data/gates, originalbudget charged; no
retrainingcompletecheckpoints. Never restart from observationtimeout alone.

Publish actualimage/question/neuralwords/targets, loss/ΔW/synapseupdates/neuronframes
to passive localhost, source/time/condition/BPTTlabels; complete864DEV+96owned
images from latestcompletedfile automatically, not bestscore or userselector.
GETonlyreads. No fakeQwenvision; existingactualQwentextcomparison stays separate.
Main/privateweights/topology/vocab/encoding/config/source remain unchanged;
preserve old STOPresults and protocols. Goalnormalchat/vision/lifelong staysopen.
