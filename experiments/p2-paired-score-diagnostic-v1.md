# P2 paired-score acquisition/forgetting diagnostic v1

Registered2026-11-09 BEFORE measurements. MasterP2 follows the narrow P1 pass
of paired-scorev2, with no new hidden rule/thirdscoreadjustment, model growth,
language run or serving import. This round diagnoses causes before optimizing
replay/importance. It is not a new retention method or an efficacy GO claim.

## Fixed stream and budgets

Development seeds15217,15227,15241. Historical context_task with20observable
contexts,D24/O4, ALIF48/K8/R12, paired-score selected independentAdam.002,
same headlr.08/clip.025/weights3 and frozen training-only calibration.
This tests the algorithm at historicalP2dimensions, not transfer of incompatible
D8/O2 P1weights. All conditions start from the same initialization perseed.

20windows×192freshobservations=3840fresh. Windowk<19 smoothly mixes contextk
andk+1 with next probability(offset+.5)/192; finalwindow19usescontext19 only.
No boundary/index/family metadata supplied to learner or reservoir (stored
context field always0). Context one-hot cue is still an OBSERVABLEinput, so
do NOT call this a fully unmarked task-agnostic natural-world benchmark.
Recurrent state resets per independent12tick diagnostic episode as P1; not
the never-reset state policy of the serving conversation/idle core.

Calibration128UNLABELEDcontext0episodes seed+7200000 only, no future stats.
Each context pool384examples seed+7300000+131*context; cursor consumes each
once, at most two adjacentwindows. Mixture RNGseed+7400000. Test128/context
seed+7500000+131*context, fixedcommon8rolloutnoise seed+7600000+131*context.
These heldout inputs/labels never update weights, calibration, RNG or replay.

At every secondfreshsample, one extra training slot: reservoir replay if
condition enabled, else repeat that just-observed sample. First slot may replay
itself; all data are already observed. Total5760headupdates,46080rollouts,
552960forwardticks/condition. Capacity128rawepisodes allocated in ALL modes,
same shapes/dtypes/metadata; reservoirseed+7700000. No context-balanced/taskID
replay or retrieval. Report bufferbytes and all time/RAM, not just weights.

Six conditions×3seeds=18trials:
stream_full_repeat, stream_frozen_repeat, stream_full_reservoir,
stream_frozen_reservoir, offline_joint_full, offline_joint_frozen.
Offlinejoint is a diagnostic upper-reference with FUTUREtraininglabels: shuffle
all3840fresh+1920repeatedtrainingitems with RNGseed+7800000, same5760updates.
It is NOT a deployable lifelong strategy, task-boundary-free learner or an
equal-buffer algorithmic baseline. Its larger rawpool is disclosed/accounted.
No forgetting/acquisition diagonal is invented for offline training.

## Measurements and fixed interpretation

Stream model receives x,label only, no update policy changes at evaluator
window boundaries. Everywindow: evaluate all20contexts in statelessdiagnostic
clones/feature batches. Primary acquisition=mean nominalwindowdiagonal,
final=mean finalrow, forgetting_j=max scores sincewindowj minusfinal_j,
mean/worst and signedBWT. Contextk+1 may have appeared duringwindowk; disclose
that diagonal is an operational milestone, not a hidden oracle firstexposure.
No rollback/modelselection on heldoutscores. Perseed joint gate UNCHANGED:
acquisition>=65%,final>=65%,meanforgetting<=3pp,worst<=5pp. Passing3development
seeds is not8seedconfirmation or full P2 success/equal-time superiority.

Head/hidden interventions for oldcontexts0..k-1 compare four combinations
using SAME heldoutrawfeatures/noise and frozenstats: oldhidden+oldhead,
newhidden+oldhead, oldhidden+newhead, newhidden+newhead. Report signed head-only
and hidden-onlydrops and interaction, do not add them as independent causes.
Headcopy includesweights+bias; statistics common/frozen. No intervention
weights are fed into subsequent training. Frozenhidden intervention must have
zero hidden-onlyeffect under commonnoise.

At end, a fresh diagnostic multiclassridgehead(.01,train-onlycenter/scale)
refits hiddenfeatures on the3840TRAININGitems and scores the sameheldoutdata.
Probe is not a runtime algorithm; no testlabels/weights imported. Distinguish
poor onlinehead from feature insufficiency. Offlinejointfull/frozen and refit
scores diagnose learnability under this budget, not an absolute capacity bound.
No claim newskillgeneralization from simple context recall, even if100%.

Report paired full-vs-frozen/replay-vs-repeat effects, seedbootstrap95 with3
clusters explicitlyexploratory. No chosen winner or gate adjustment. If even
offlinejoint+probe fails65%, don't spend this round tuning importance to freeze
an unlearned skill. If acquisition adequate but retention fails, use head/hidden
interventions to design separateP2importanceprotocol. No automatic expansion
or additional same-familyhidden tuning from either outcome.

## State, source and observation

Checkpoint model+complete replay+cursor and immutable streamhash at end; verify
same next replay sample/update and full state after load, no originalartifact
mutation. Keep head/hidden snapshots and all matrices/predictions in run. API
observer receives realloss/ΔW/windowmetrics and tensor outputs with isolated
research scope; no mock chat/Qwen. Model live/privateweights remain unchanged.
CPU1thread/worker,max2workers,2GBaggregateprocess-tree monitored,180s/trial,
1800sround. Stop on NaN, bad replay/checkpoint/forwardparity or memory/deadline.
Sampling is not a hardaggregatecap. No paidAPI/new teacher/model server.
