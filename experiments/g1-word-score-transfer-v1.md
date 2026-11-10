# Locked paired-score rule transfer to next-word pilot v1

Registered2026-11-10 BEFORE model measurements; proposalreport25/masterP1/P2.
No thirdscorefamilytuning, stoppedfamilyreopening, biggermodel, pretraining,
privateimport or newservingID. Existingselected algorithm transferred unchanged
to a different small wordtask with outputmask; not inheriting originalP1pass.

ThreeFRESHindependent modelseeds27103,27109,27127. SameALIF48/K8/R12/layout
G1(wordV+128context+64reservedimage+6control,D; O=V+10reserveddigits).
Forwardleak.65/threshold.5/featuredecay.25/adaptationstrength.02 as existingG1;
otherConfigdefaults. Stochastictemp.4, independentpairedrollouts, hiddenAdam
.002,moments.9/.999,eps1e-8,updateclip.025,weightclip3,headlr.08. These learning
settings from lockedP1, no new LR/grid/hiddenhyperparameterselection.
InitializeSpikingCore same seed then subtract.15fromw_in likeG1, stablecopy.

Conditions frozenhidden, selectedfullscore, shuffledsign. Sameinitialweights/
data/noiseschedule, headupdates and fourblock/tworolloutcost for ALLconditions.
Shuffledsign perblock as source, headtarget remainscorrect. Ninefinaltrials,
1024labeledresponses each (uniform8intentgroup then uniformtrainexample).
Train=64publicauthored+117AGYreviewed(181), dev31, priorAGY48alreadyinspected.
No privatefeedback/mathproofs/userweights used. Closedstyletargets known, not
generalchat/novelknowledge. Vocabulary trainresponsesonly, targetsexpressible
checked; reserved10digitheadrows masked/untrained, no visionqualityclaim.

For currentword t, construct 12zero-boundaryevents then3ticks perteacherprefix
position0..t. Positioninputusespreviousword(BOSatt0), staticpromptcontext and
G1controlsignals. TRUEcurrenttarget only at finalscalarreward/readoutupdate,
never in input. Parent learn_example on wholeprefixsequence, sameweights for
pairedtraj, own recurrences, priorEMA baseline. Hiddenrotatingblocks cover48.
Wordheadmask excludesBOS/digits always, EOS/punct atposition0. Mask is ordinary
outputtaskconstraint, saved/validated, not a target/intent-to-template lookup.

Calibration128UNLABELEDtrainingprompts with randompreviousword histories of
length0..5, randomIDs fromtrainvocab (not trueanswers),8rolloutfeaturesmean;
stats frozen beforelabels, sameinitialcore/statistics acrossconditions. No
development/priorprompts or labels. Fixedcalibrationrngseed+10100000.
Train group/sampler seed+10200000, shared allconditions. Decoder MC8 fullprefix
replay from NEURAL-generatedpreviouswords, max12words; seed+10300000+caseindex
fixed for comparableinference. Inferenceclones canreset betweenprefixrollouts;
no maincontinuousstate/clockcompatibilityclaim. No beam/newdecoderpenalty.

At256/512/768/1024responses record actualcurrentloss, updates,ticks,weightdelta
and3neurontrace fields. Evaluate onlyFINAL1024checkpoint, no earlystopselection.
Scorefreegeneration strictstyle, existing normalizedstyle, teacherforcedtoken
accuracy/NLL, firsttoken,stops/repetition. Split train diagnosticsfirst16fixed
cases(2perintent), dev31, prior48. Same fixednoise percondition/seed.

Exploratorypilotinformative only if fullscore meanDEVnormalized>=40%, paired
mean gainvsfrozen>=10pp with positivegain onALL3seeds AND meanpriorgain>=10pp.
No P1representation/P2retention/normalchat/vision pass from this descriptive
criterion or8knownresponseintents. Report seedbootstrap95 exploratory, shuffled
and allbadoutputs. Failure=STOP_WORD_SCORE_TRANSFER_V1, no grid/tokens/model
budgetincrease or deployment. Positive=FRESH_VALIDATION_REQUIRED, no mainimport
or automaticconfirmation opened inthisrun. Any follow-up requires new protocol.

Codec wrapper/coremasks:RNG/neuronstate/calibration/moments/vocab/decoderRNG;
checksums/nonpickle/sizebound/partialcontinuation tested. Basepaired-scorev2
format/bytes remaincompatible; maskedwordcore has DIFFERENTformat. Inputmask
and targetbounds tested BEFOREweightschange. One actualmodelcore pertrial,
no pretrained/tokenembedding teacherweights, targetlookup or answerselector.

CPU1thread/worker,max2workers,2GBprocess-tree,240s/trial/1500sround. Actual
rollouts/ticks/arrays/buffers/time reported; no equal-cost advantage vs oldG1.
Stop on NaN, sourceformatmixing, labelleak or budgetfailure. Publish realloss/
synapseupdates/generation, scopeexperimental48neuron, not modelinHermes. Raw
checkpoints underartifacts with publicdata only; histories/protocols preserved.
