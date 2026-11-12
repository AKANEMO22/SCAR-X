# ACR DEV v1 — đăng ký trước đo chất lượng, 2026-11-12

User attachment SHA256: 17b3b4253672c4d5b846b893acd10b4a4ec4ad88a5dc74ac9f272bb706b5cf68.
Base 93b5b96bebaf22dbc39a3f96b2cac7c917d5e377; equations pinned
bc6865003cc63dd846b4321d5050a0efbe27ef24; peer607ffca unchanged/clean.
Ponytail unavailable. Separate research branch; no main/Hermes/live/port8767.

## Scope and mathematical lock
One unchanged H48 D8 K8 ALIF forward. Two independent (e_v,e_a,e_f)
streams; stopped drops both reset terms only in next e_v. Mix AFTER each
stream's existing sensory gate and heterogeneous filter. Copy exact endpoints.
Readout, pre-update signal, momentum/energy, clips and update count unchanged.
Own/stopped can diverge before a hard spike because surrogate derivative is nonzero.
The label arrives at episode end: alpha(t) does not imply online weight updates;
only final alpha contributes directly to the update. Router weights are FIXED
heuristics in v1, not learned. q depends on trace history, not previous alpha.

P0: exact endpoints including active stream, optimizer, RNG, stats and resumed
next update; scalar recurrence oracle; dynamic mixture ordering; one forward;
snapshot bounds/checksums, full state/RNG, rollback; applied-norm feasibility.
Any endpoint mismatch blocks DEV. Reviewer checks real source/tests.

## Finite DEV schedule, no confirmation
Discovery seeds 140011,140017,140023 (old exploratory DEV, explicitly not fresh).
Exclude all twelve 20261112xx confirmation seeds and the R9 bank.
Four unchanged tasks strict_binding,xor,distractor_binding,order. Each has
1024train/256refit/512ID/512OOD; calibration first32TRAIN only.
Fifth environment `joint`: one shared checkpoint/head, 256TRAIN+64refit from
each family, deterministically shuffled using seed+900; 512ID/OOD each family.
One pooled refit head; no per-family probe head or task marker to learner.
Pooled shuffled training is NOT a continual retention experiment.

Stage AB: static_0,static_25,static_50,static_75,static_100;
linear_up/down (.1 to .9); groups_up/down (.25/.75 halves);
intrinsic sigmoid(4*(gamma-mid)/range). Uniform aliases static, not extra evidence.
All 10 profiles plus readout_only at all 3seeds x5environments =165jobs.
Frozen-hidden aliases readout only after P0 proves exact predictions/state.

Stage C opens ONLY after AB unless one uniform profile meets all broad DEV
checks below. One fixed router, no coefficient search:
logit=intrinsic_logit -1 -.5*clip(|u-theta|/(theta+1e-6),0,4)
 +1.5*z +a_pre/(1+a_pre)+q.
q=.9*q_old+.1*(||Fstop||-||Fown||)/(||Fstop||+||Fown||+1e-6).
Router receives no labels, task, loss, accuracy or future state.
state, state_shuffle (permute feature vectors, keep neuron bias), random_alpha
(permute computed alpha, preserving distribution exactly), state_signal_shuffle.
Shared RNG draw schedule, dedicated PCG64 seed+43100000, persisted.
4profiles x3seeds x5environments=60jobs maximum.

Selection: ONE global nominee, never per-task. Rank isolated primary cells only;
joint checks are a separate veto. Rank by minimum of mean direct
and mean probe differences from the better endpoint in each primary family,
then global mean direct difference; ties prefer simpler static then name.
Joint results are reported separately and must not hide family regressions.
Always run nominee signal-shuffle on15cells (reuse state_signal_shuffle if same).

Additional paired actual-update-norm controls: static_50 and global nominee
(deduplicate), each vs own on15cells; maximum30 paired jobs/60models.
Each gets1024updates; same source optimizer; actual postcast/postclip hidden
delta radius R=.5min(raw norms), bounded64-step ray bisection, tolerance1e-5.
If a positive radius cannot be realized, restore BOTH pre-update states and
mark INVALID; never silently drop rows or claim norm-controlled benefit.
R=0 is reported, not a plasticity gain. Optimizer histories remain distinct;
this controls amplitude but cannot prove direction is the sole cause.

Equal CPU secondary: own,stop,nominee,nominee shuffle,readout on15cells each,
0.5process CPU seconds per model (checked at episode boundaries), cap4096updates;
maximum75jobs, never substituted for primary fixed1024update results. CPU is
measured per model for raw arms; paired norm controls report total pair CPU
and half-allocation only, never claim that allocation is individually measured.
Own/stop here are dual-stream ACR endpoints, not cheaper original single-stream
implementations. This isolates routing at matched instrumentation/compute;
it does NOT establish CPU efficiency relative to the pinned single-stream code.
No optional extra seed validation bank in v1. No fresh confirmation.

## Mechanism/OOD analyses (no additional candidate fitting)
Every actual training update logs loss, applied L2norm, cumulative CPU time.
Initial/final full checkpoints and all ID/OOD direct/probe predictions saved.
Revert input/recurrent/both hidden weights separately, keep learned head;
refit probe on TRAIN-derived refit only. Report direct causal drop and probe
gain separately; a probe does not establish usable direct skill.
Diagnostic clones only: fixed TRAIN indices0,63,511,1023 replayed from final
checkpoint, per-tick u,theta,z,a,own/stop/mixed eligibility,alpha,q; one real
local update per clone logs pre-update signal and actual hidden vector.
These are labelled clone diagnostics, not logs of the original training steps.
Compare stream cosine/norm and update direction; final alpha only intervention.

OOD stress on nominee+endpoints, first128 original ID examples each family:
insert12zero ticks before final query/readout; multiply nonfinal channels6:8
(background AND distractor pulses) by3; swap nonfinal channels6/7, preserving
the final channel7 response marker; semantic event rows/channels0:6 x.7;
binding query flip4/5 with labels
recomputed from original writes and paired flip consistency reported. Query
flip is a counterfactual intervention, not an unseen query distribution.
Order of semantic writes protected. No task
name is sent to router. Primary input encoding/splits unchanged.

## DEV decision (exploratory, no P1 claims)
Broad checks: direct>=.65 in all3primary families; both direct/probe mean no
worse than better endpoint by>.03, including XOR; OOD direct/probe no worse
than own by>.03; order direct>=.65 and regression<=.03. At least2/3 families
must improve direct OR probe by>=.02 vs better endpoint, with probe gain>=.05
over readout and direct causal drop>=.03. Additionally >=2/3 families need
mean ID probe advantage>=.02 over nominee signal-shuffle, and >=2/3 need
mean ID probe advantage>=.01 over own in actual-norm-paired trajectories;
neither control may show >.03 direct/probe regression in any primary family.
These are exploratory effect thresholds, not significant confirmation claims.
All prescribed control cells must be valid; incomplete/invalid cannot advance.
Joint checkpoint must preserve order and avoid >.03 direct regression vs better
endpoint in each family. Confidence: seed-level effects and descriptive95% CI,
exact8sign-flips; 3seeds cannot establish significance (minimum p=.25).
ACR_DEV_ADVANCE only if all conditions hold plus independent integrity audit.
Otherwise representation-only if >=2families probe +.02 without direct gain;
specialized if only one improves; no-gain if none; invalid mandatory control
=>BLOCKED_INVALID_CONTROLS, even if raw accuracy is good. Keep negative evidence.
No changing thresholds or tuning coefficients after scores.
Future P1 must keep approved 2/3,65%,CI5pp,Holm,OODNI/protected gates unchanged.

## Resource/stopping lock
One worker, one numerical thread. Max150minutes total from DEV runner launch,
each job120seconds. RSS AND committed process/tree checked against2,000,000,000
bytes, peak separately recorded, no swapping weights to SSD. Reserve output
directory once, source hashes including imported peer/core/generator locked;
stop on changed source/dirty peer, invalid data oracle, endpoint failure,
nonfinite state, memory/time limit. Partial run cannot advance; report exactly
which cells are missing. Observer on8771 only, passive and bounded.
Maximum375 trained models plus144 diagnostic clone updates (3profiles x4tasks
x3seeds x4TRAINepisodes),
not an open-ended search. Commit code/protocol before opening DEV outcomes.
