# SCOPE P1b — locked experiment, no production permission

Date16/11/2026. User FULL directive SHA2e0fa1fd…849c6e5. Research branch only,
no push/main/Hermes/checkpoint live/reset servingP. Prior P1/6e08a361 remains
PREFLIGHT_NO_GO; independent saved-data audit confirmed it without replay.

## Prerequisite evidence before first learning update

Intake `runs/20261115T174514Z-scope-p1b-v1-b30f0b`: lockedpreflight commit
0e2d20cc,942newCPUforwards/0updates. All preregistered gates passed: teacher,
S-only witness, oldvalid, ROcollision, conditionaltracevariance, physical
proposaldifference and causalprefix. DeltaB−deltaC median≈28,9%, not just
nonzero targets. Corrected audit independently reconstructed exactphysical
delta; first audit JSONnp.bool serialization failed, preserved; recovery is
data-only, no newforward/model/dataset/gate change. Gate-adjudication records it.

CPU median2episodeforward+trace+proposal3,9087ms versus CUDA237,8319ms,
paritymax2,98e−8. Freeze **CPU NumPyFP32,1BLASthread/worker** for experiment,
max3parallelworkers. Not a claim all SNNs are faster on CPU. GPUprofile uses
CuPy14.2.0, CC8.9, runtime12090/driverAPI13040; WDDM runtime free-memory query
does not equal whole-card freeVRAM. NVIDIAglobalbefore/aftersamples≈2885/2800MiB
free, no exactGPUcontextpeak claim. No more profiling or CUDAtraining.

## One hypothesis and fixed mechanism

H: aligned causal temporal Pfiltered-spike features improve useful S hidden
learning beyond label-only and matched same-class/readclock teacher traces,
including adverse initial readout. Adaptation of featureKD/P&C/localeligibility,
not novelarchitecture, newShannoninformation or proof Plearned its skill.

Task/teacher/encoder/generator/initialization/localphysicalrule are **exactly**
the locked preflight and `scope_p1b_common.py` numeric equations used there.
Only diagnostic arrays were added to learn()'s returned dictionary; its entire
computational prefix and forward/proposal AST must match archived preflight.
Teacher
hand-engineered; S/P64+64=128, fan8, D64,32tick, FP32, Rfrozen0. S8oldrows
fixed, S56newrows learn Wincolumns0/1/3/4; same correspondence P64..119 to
S8..63 provided a priori. Newhead2classes, oldhead2classes, fixed total2heads.
No hiddenR/out-of-mask/statistical-normalizer updates. No BPTT/RTRL/autograd.

Candidate direction D=m_label E_f,T+bank_hint, bank_hint=mean_t((qP_t−qS_t)/.1)
clipderivative E_f,t, coefficient1. A uses D_label only; B aligned teacher,
C permutation within(label,readclock), D readout-only. Mom=.9/energy=.99,
step.008,updateclip.01,bound3; newheaddelta.04/clip.025,bound3. Input-only
forward teachertrace uses no futuretarget. Candidate labels available at end.
Readout feedback uses pre-updatehead. Oldhead frozen. Eval has no teacherarg.

## Data lock and exactly24 trajectories

Seeds261161/261167/261173 × initial readoutsign+1/−1 × A/B/C/D =24arm-runs,
not24eachcondition. Pair same Winrandominit/trainorder/data/labels byseed/sign.
Teacher construction fixed for all. Signconditions and configs are not
independentseeds. Differentseed studentnoise/order/amplitude draw realvariation.

Perseed TRAIN192, DEV128, untouchedTEST192 from source salt-separated generator
as preflight. Labels positive iff orderedABgap1/2; negatives includeBAshort
andlonggaps. TEST has unseen clocks and amplitude bands andlonggap7; not all
axes are disjoint fromDEV, do not claim arbitrary OOD. Inputs do not contain
label/taskanswer or futureevent metadata. Model receives only sequence.
Data bytes/hash generated before first update. TEST only saved/sealed, no
forward or selecting winner until G2. No oldTEST accessed.

Each trajectory **192 learning updates, one locked order**, not bestepoch.
DEV curve snapshots at updates0/48/96/192, oldvalid12cases same checkpoints.
Curves only report, no early stopping/parameter selection. Every update saves
loss/delta/eligibility diagnostics. Checkpoint at0 and192; hidden-revert
ablation on finalDEV keeps learnedhead but reverts Win to initial. No fitting
new probes or adding controls after results. Predicted ROceiling75% raw, not
neural skill. Require minclass recall≥65% in addition to acquisition≥65%.

Teacher TRAINcache computed once/seed, fullP filteredtrace andspikes stored.
All arms run same teacher-bank arithmetic and same count studentupdates/
forwards; A/D discard auxiliarydirection, C gets matched permutation. Cache
setup shared cost reported separately. Real-use A/D could omit cache/bank;
do not call deploymentcompute exactly matched. Teacher manualprior is explicit
extra knowledge, not free selflearning. C preserves teacher information amount
and class/time distribution, breaks samplealignment. D freezes hidden and all
arms preserve old stable subcircuit by mask, not by learned antiforgetting.

## G2, uncertainty and no failed-run rescue

FinalDEV: for **eachseed andbothsigns**, B must exceed eachA/C/D by≥.05,
acquisition≥.65,minclassrecall≥.65, olddrop≤.03 and hiddenrevertdrop≥.05.
Compute paired stratified episodebootstrap10.000 draws, fixed3trainedseed
models and both signs paired on samecaseindices. Stratify (seed,order,gap,
readclock); report95% intervals and Bonferroni-adjusted98,333% percontrast
intervals for3comparisons. Require adjustedlower>0 for B−A,B−C,B−D.

These are **input-conditional pilot intervals for these trained models**,
not a95% claim over the population of modelseeds. With nseed3, population
uncertainty remains INCONCLUSIVE (minimum exacttwo-sided signp=.25). No
project-levelCI/lifelong qualification; scientificgates unchanged. G2 can
qualify only a finite-pilot signal and conditional technical stages below.

If any required finalcontrast fails, label LEARNER_NO_GO/NO_TEACHER_BENEFIT/
INCONCLUSIVE by evidence; no TEST, restart/reuse continuation. Finish the
registered24paired cells for finalestimates, no newconfigs or seeds. Runtime/
resource/harnesserror stops affected study, never reruns learning to replace
bad results. All partials kept.

## G3: only after G2 conditionalpilot qualified

Fixed checkpoint S64 loader accepts student kind, noP arrays. Independent
inferenceprocess receives only studentcheckpoint and X-only DEV/TEST/old
inputs; labels stay in scorer. Openaudit hook denies access to any other
files within studyroot. No teacherweights/spikes/logits/hints/archive/RAG at
inference. Verify exactDEVpredictions before/after restart; then open new
sealedTEST once for all24registeredcheckpoints to retain controls.
Require BnewTEST acquisition/minclass≥.65, olddrop≤.03 and all3TEST contrasts
≥.05 with conditional adjustedlower>0 before G4. No repairing primary models
afterTEST; G4 creates a separate pre-registered S2candidate and keeps S1immutable.
This is oneoldskill retention, not20stage3/5pp validation.

## G4: registered continuation inside the same B trajectories, no new arm/seed

Only afterG3 qualified, continue the6B coupledS/Ptrajectories. Same64+64neurons,
same2heads and masks. Reset sameP64 with seed+fixednamespace (no seedsearch),
initialweights .12±.002 onnewobservedchannels3/4 and head sign as original.
TRAIN2/DEV2/TEST2 saltcycle2 independent, rule reversed: BAshort ispositive,
notABshort; this is two sensor/event contexts provided, not taskfree discovery.
Labels never encoded by context alone. Teacherwitness union feasible onpaper:
detector Win3=.358,Win4=+.158, retaining Win0=.358,Win1=−.158 and samehead.

First P learns192label-only localupdates, then qualifies on DEV2≥.95 and
minclass≥.65. If fail STOP itscontinuation, no retry/reconstruction teacher.
If pass freeze P2; consolidate into S with384updates interleaving all192TRAIN1
and192TRAIN2 by fixed seednamespace50000. Oldhints are cached S1features;
newhints from actual learnedP2, same localrule/constants. No newdata/oracle.
Keep bounded384replay only, not archivepertask. Before resetP2 require both
DEVskills acquisition/minclass≥.65 and old/new1drop≤.03. Save S2, repeat isolated
restart with X-only(TEST1,TEST2,old); score once. No thirdcycle/20stages.

Continuation is phase of each existing B trial, not added comparative arms;
counts all P/S forward/updates and runtime, max576extra updates/Btrajectory.
If remaining executionbudget insufficient, NOT_RUN_RESOURCE, never increase.
Do not claim initial constructedP learned; G4would be the first actual resetP
acquisition check. Skill2 may fail; preserve it. Gates cannot be weakened.

## Resource/custody

Total1800s activeexecution including saved-dataaudits/profiling/preflight/
training/conditionalstages/audit; intake≈10s plus conservative60s reservation,
no timetransfer by hiding setup. Coding/literaturereading not modelruntime.
Pertrial hardbudget45s; total24×45=1080s primaryupper; allcontinuations share
remainingglobalbudget and stopbefore exceeding. NumPy1thread ×max3workers,
experimenttreeRSS/commit sampled50ms and perprocessWindows2GBcap. Aggregate
resourcecheck<2GB; OSpeaks perPID used as conservativebound, avoid doublecount
reusedworkers. ProfileGPU context has exited, no CUDAtrainingcontexts.

Fullsource/protocol/data hashes, dirty status, allpredictions/labels/losses,
actualhidden deltas, eligibility/hint norms, checkpoints/RNG/fullstates,
PID/RSS/wallandcounts stored unique paths. No overwrite oldintakeerror/P1raw.
Audit separately recomputes metrics/contrasts/bootstrap/revert/checkpointhashes
without training. Identicalbitfloat / np.bool report bugs do not permit new
learning runs; pure reportrecovery must preserve failures and sourceversion.
