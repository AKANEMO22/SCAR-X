# Read-only audit of wordtraining objective / prefix reliance

Registered2026-11-10 BEFORE code/measurement. Responds to user's question whether
language-style supervision fits ALIF/SNN. MasterP1/§4.1 permit taskvalidity and
credit diagnostics. NO neuraltraining/optimizer/headrefit/newlearner/encoder/
stateviewgrid, no reopening STOP35/36/37, no objectivechange/deployment/scaling.

## Sources / invariants

All15FINAL36checkpoints (3pairedseed36313/36319/36341×correct/shuffled/zero/
frozen/BPTTcontrol), reviewed64train/64dev prompts and8ownedstylelabels from
datasetSHA585fd541cdc2651e554e7cd9b7a0ae6d04d0dd5fc0acc1104add773a92864d0b.
ExistingactualAGY128promptgeneration/1languagefix and Qwen64textrefs reused;
no newteacher/model/APIcall or use of Qwenanswers as labels. Historicaldata,
not freshfinal/generalchat/lifelong evidence. MainALIF/privateweights untouched.

Candidateforward UNCHANGEDDirectWords36,48ALIF/G1hash128 and3ticks/word. Read
checksummed BPTT wrapper only to extract its DirectWords researchreference,
without Torch/controlimport/backward. No controlweights import into main.
Sourceweights/stable/topology/headstats/Adam/moments/learningcounters/RNG
unchanged. Discardedcopy workingstate/traces/bank/ticks necessarily change during
read-onlyforward; sourcefileSHA unchanged. Do not claim allstates unchanged.

## Corpus structure (descriptive, not modelanswers)

For8canonicaltargets, compute prefix→possible nexttokens with empty vs nonempty
GOLDprefix. Count unique firstwords, laterpositions/punctuation/EOS fraction.
Uniformmass over8families; retain actualtokens/ambiguitycounts. This is an
external finitecorpus property, not learnedneuralrepresentation or deployed
lookup. No classifier/templaterouter may answer userprompts from this table.
Flag PREFIX_STRUCTURE_SHORTCUT if every nonemptygoldprefix determines one
nexttoken among these8targets. Even true, not evidence all modeluses thiscue.

## Actual forward interventions, ALL128cases/15sources

1.FREE: originalsingleprompt, BOSthenpreviousNEURALtoken only,max8words/EOS.
No target accepted by FREEforward function; score its text/tokens afterward.
Recordedtext/stopped mustmatch source36saved output for EACHcase, abort mismatch.
Capture per-position predictedtoken/probabilities/actualticks and neuronstate.
Goldtoken/target only assigned outsideforward when scoring positions.

2.GOLDPREFIX: sameprompt but previousGOLDtokens, reproduce teacher_metrics36
exactwordNLL/accuracy. This is explicitly DIAGNOSTIC teacherforcing, not free
utterance. Storedsource teacher metrics mustmatch (1e-7 loss tolerance).

3.GOLDPREFIX + WRONGQUESTION: use firstotherpubliccase with DIFFERENT canonical
target, samegoldpreviousprefix/positions. Targets selectquestion OUTSIDEforward
only. Model receives wrongpublicquestion, no family/intent/answer oracle.
Keep originalcase gold nexttoken only for scoring conflictintervention.
This may be OOD contradictory context; no generalcausalityclaim from droponly.

4.GOLDPREFIX + ZEROQUESTION: zero only128questionchannels by scoped _context
hook while keeping actual36inputword/controlflags/steps. Validate original
prompt first, restorehook afterward; no otherencoder/normalization changes.
OODblanking distinguished from wrongquestion. No parameters fitted/tuned.

Report FIRSTWORD and LATERwordinclEOS accuracy/NLL separately (macro over8families),
and nonpunctuationcontentlater vs punctuation/EOS to expose easytargets. Do not
compare tokenaccuracy to wholeutterance as if identicalmetrics. At matching
positions actually reached in FREE (including predictedEOS), compute teacher-
prefix vs neural-prefix gold nexttokenaccuracy on SAMEpositions. If model
stoppedearly, latergoldpositions are censored/missing, not automaticallywrong
or silentlyexcluded: report evaluatedcount/censoredcount and scope separately.
Also verify teacher-all-argmaxcorrect iff freeoutput tokens+EOS exact, because
with samegreedyforward this musthold; teacherforcedtokenaccuracy alone cannot
replace wholeansweraccuracy. Outputs beyondgoldEOS retain freeexactfailure.

## Read-only candidatecredit quantities, CORRECTsources only

On all128cases at each GOLDPREFIX endpoint, track existing own-neuronforward
eligibility, calculate POTENTIAL unpermuted direction
(W_out.T@(onehot(gold)-p))*interiorclip/scale × e_f. Fixedcheckpoint/headweights,
no accumulate()/head.learn()/finish()/Adamupdate, no RNGdraw or gradienttape.
Thus these are signals at finalcheckpoint, NOT pastapplied weightdeltas or
actualtrainingtrajectory. Allactualtrainingweights/head changed during36;
do not retrospectively claim this reconstructs their history.

Group incomingdirections into previouswordV, question128, reservedimage64,
controlflags6, recurrentK8. Report per-positionL2norms/firstdirection vs meanlater
directions, cosine(first,sumall) and cosine(first,sumlater), median bysplit.
Normlarge is NOT usefulcredit or proofcausalresponsibility. No directionused
to update/freeze/select synapses, no signalrule changes from thisaudit.

## Pre-registered diagnostic flags / interpretation

* PREFIX_STRUCTURE_SHORTCUT asabove. Closed8templates remainclosed even if100%fit.
* POTENTIAL_PREFIX_RELIANCE per source/split if GOLDPREFIXlateraccuracy>=90%,
  wrongquestiondrop<=10pp and FIRSTWORDwrongquestiondrop>=20pp. This is a narrow
  fixedcounterfactualflag, not proofquestionirrelevance under normaldistribution.
* EXPOSURE_SENSITIVITY if on SAMEuncensoredLATERpositions goldprefixaccuracy−
  free-neuralprefixaccuracy>=10pp, with counts/censoring disclosed.
* DIRECTION_CONFLICT if median cosine(first,sumlater)<0 on CORRECTsources;
  finalstate diagnostic only, not proofthis caused generalizationfailure.
Allthresholds descriptive, NOT projectP1/P2/qualitysuccessgates. 3seedbootstrap
acrosspairedmodels,15conditions not15independentseeds. If unsupported, report
UNRESOLVED; do not inventselectivelearner or tune views tofindeffect.

Result can motivate a SEPARATE selectivecreditproposal within approvedsignals,
not automaticallyauthorize it or override STOPv1/2roundlimits. Need explicit
objective/ablations/protocol before any newlearningexperiment. Genericnexttoken
supervision is architecture-independent; thisaudit cannot certify it fits
SNN/biologicallearning/generalAI or explain failuresuniquely.

CPU1thread/2GBprocess+sampledtree,300sround, no Torch/iGPU/offload. Stop source/
format/forwardparity/nonfinite/protectedstate/budgetfailure; preserve logs.
Meaningfultests and fullpytest/nativeparity for observerchanges before audit.
Passivelocalhost typedmetrics with GOLDPREFIX vs FREE labels, realpublicanswers/
archivedQwensource/time, no learningloss/ΔW fakeentries. Main/private invariants
and all15sourcefiles checked afterward. Do not modify historicalreports/results.
