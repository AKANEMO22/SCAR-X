# Review of incoming DeepSeek product changes / technical evidence v1

Registered2026-11-10 BEFORE independent reruns. Incoming GitHEAD27b6c10, changes
since8337d60 and untracked interference_evidence_v2.py are the review subjects.
User requested read/compare/process the new work. No learningrule/LR/topology/
neuroncount/signal/decodeobjective change, no real-session teaching/restore,
no replay/consolidation/modelscale. Keep user's restored current64ALIF state;
do not revert it to prior128state or assume old codehash still describes HEAD.

## Fixed technical checks / intended fixes

1. Guard-on-manual path, before-probeclone and fullneuralrollback isolation.
   Remove livegenerate fallback when clonecreation fails: fail/rollback instead.
   Verify with faultinjection and fullstateregression; no guardthreshold changes.
2. Copy liveSQLite session using online backup, not independent DB/WAL filecopies.
   Verify coherentcommittedsnapshot and sourceunmodified, never copy lockfiles.
3. Interference parametertransplants must apply EXACT requested AFTERarrays and
   retain unselected BEFOREarrays/state. Existingv2source forgot afterbias/hidden.
   Unit-test everygroup, sourceimmutability and shape/method rejection.
4. Productacceptance v2 uses exactcanonicalcontent for knownfixture and real
   sentence terminator, not mere substrings such as negatedidentity or "nothello".
   Keep v1explicitlyavailable for historicalreproduction; rows/versionlabels and
   legacy-v2 results sidebyside. Do not reinterpret historicalnumbers/gates asv2.
   This is measurementbugrepair of declaredcomplete/correctproductcriterion,
   NOT lowering or retuning P1/P2/researchsuccess thresholds.
5. Report newskillacquisition separately from acceptedtransactionrate. On each
   explicitpubliclesson, scorebefore/after on isolatedcanonicalclone, distinguish
   alreadycorrect, newlyacquired, accepted-without-acquisition and rejected.
   Eligiblebaseline-failingcases denominator0 => missing, never fakepass.
6. Reused6"HELDOUT"phrases are developmentexamples, not freshfinaltest. Grade
   them with isolatedresetclones and explicitfullreference in v2; retainlegacy
   keyword/statefulmetric with label forreproduction. No reruns until higher score.

## Independent causal diagnostic (unchangedrule, disposablemodels only)

Freeze currentlive /api/export into private local/<run>/ snapshot once. Extract
modelpayload only, never write realDB or restore/reteach usermodel. Publicrelease
checkpoint is secondsource. Optional thirdsource is OLDEST chronological DB
checkpoint with DIFFERENT weights but SAME topology/vocab/encoder as current;
compatibility only, no choosing on quality. Ifnoneavailable, reportsources2.
Startingcheckpoints are NOT independentnetworkseeds, no inferentialCI from them.
No privateprompt/weights/fingerprint in publicreceipts/sourcepackage.

Fourfixedpubliclessons as incomingv2: hi→Hello.; cảmơn→Rấtvuiđượcgiúpbạn.;
chào→Chàobạn.; bạnkhỏekhông→Tôiđanghoạtđộng. Actualstrings retainspacing incode.
For eachsource/lesson: from same clonedbeforestate, full-local and readout-only,
exactly6train_pair calls. Do NOT call realGeneration.teach/guard or change LR;
diagnostic intentionally bypasses servingguard on throwawayweights to reproduce
interference. No candidatelearnedweights saved/deployed or claim of newskill.

Measure fixedpublic10probeexpectations (8productguardprompts +activity/teaching
fixtures). Reset a DISCARDEDclone perprobe and use same availableactivebudget
min64/allocated, never growmodel, userruntimecontinuity untouched. Baseline
already-failing pairs, newlylost content, style-onlyloss and gain separate.
Changinglesson prompt excluded from "oldskill"protectedset; report scope.
Fullstep0..6loss/readout/bias/input/recurrentdelta and probe changes recorded.

From EXACTbaselinepayload, transplant aftergroups:
none, w_out, bias, w_in+w_rec, w_out+bias, w_in, w_rec, allfour.
Allgroups mustreproduce trainedcanonicaloutputs, none reproducesbaseline;
assert selected/unselectedarrays exactly and rejectdifferentvocab/indices/encoder.
Record head-onlysufficientforTHISlostset, hidden-onlysufficientforTHISset;
normratios/LRratios never imply negligiblehiddencontribution or universalcause.
No loss in giventrial => causalflagnotapplicable, not evidenceabsenceofforgetting.

## Scope, resources and reporting

No preregisteredneuralrule/qualitygate changed. Newknownproduct8cases and6reused
paraphrases are finiteengineeringfixtures. No claim commercial-equivalence,
generalvision, solvedP1representation/P2forgetting from theseexamples.
Audit39signflip beforeAdam withstoredmoments is not exactopposite applieddelta;
document limitation, preserve oldresult/source. Do NOT rerun/tune39 or call its
approximateeligibility "not a gradientapproximation" merely for noautograd.
PRODUCT_DEBUG_REPORTA4 is superseded by isolatedrollbackproof, markas such
without deleting earlierbody/results. Gitexistsnow; no init/commit/push byreview.

CPU1thread/2GBprocess+sampledtree,300s independentdiagnosticround. Fullpytest/
nativeparity afterruntimefixes. Source/currentexport/protectedweightchecks before/
after, no trainingjob running or modelrestore. Publish actualwork/probepredictions
and completecausalrows in passivelocalhost with COPY ONLY/DIAGNOSTIC labels;
no fakeQwen/controlmeasurements. Keep currentAPI one scarx-generation1.
Finaldeliverable is reviewreport40, reproducibleevidence, fixes/tests, updated
master/navigation/status and packageverification; goalnormalchat/vision staysopen.
