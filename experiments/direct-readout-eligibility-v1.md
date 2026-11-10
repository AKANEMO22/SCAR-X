# Proposal30 / forward-local eligibility + direct readout signal v1

Registered2026-11-10 BEFORE implementation or performance. User explicitly
approved proposal30 research, not deployment/scaling/Transformer/pretrained/
proxy/whole-network-BPTT/RTRL/autograd in candidate. MainALIF weights/source
remain untouched. This is a NEW approved credit mechanism, not renaming/tuning
closed score/classifier/byte/visual families. Maximum2registeredrounds/hypothesis;
no automatic parameter grid or moving gate after results.

## Scope of this first stage

Engineering and temporal/relational feasibility FIRST. No language10Mtokens,
modelgrowth, C++/iGPU or claim normalconversation/vision/lifelong. Word/vision
bridge using the same mechanism requires its own registered corpus/ablation
after mathematics/state integrity; approval30 remains research authorization.
Antigravity/Qwen data are not needed for these numeric tests; existing real
teacher/publictext references remain, never fakeQwen on tensors or images.

## Fixed equations and locality

Authoritative NumPySpikingCore ALIF, H48/K8/R12 metadata, D8/O2, originalConfig
leak.85/threshold.65/adaptstrength.08/adaptdecay.95/featuredecay.92, init unchanged.
These are Phase2.1 matched task parameters, not G1serving timescale changes.
All methods same initialization/input/forward/head/calibration/experiencebudget.
No surrogate neuron replaces hardspikes in candidate FORWARD.

For row j, pre_j = [currentinput x, previousspikes[indices_j]]. Incoming neural
activity is treated as observedconstant for sensitivity; no derivative of other
neurons, including self-presynaptic recurrence, is propagated. Own neuron's
voltage/reset/adaptation/filteredspike dynamics remain in its local trace.

Before tick: V_j=postresetvoltage, A_j=adaptation, Z_j=spike, F_j=filteredspike.
U=alpha*V+sum(w*pre); theta=threshold+beta*A; Znew=hard(U>=theta),
Vnew=U-Znew*theta; Anew=rho*A+Znew; Fnew=kappa*F+(1-kappa)*Znew.
Surrogate only for sensitivity: raw=(U-theta)/.4,
psi=sigmoid(clamp(raw,-20,20))*(1-sigmoid(...))/.4 if abs(raw)<20 else0.

Per incoming synapse, initialized0 at segmentboundary:
u_e=alpha*v_e+pre; z_e=psi*(u_e-beta*a_e)
v_e_new=u_e-theta*z_e-Znew*beta*a_e
a_e_new=rho*a_e+z_e; f_e_new=kappa*f_e+(1-kappa)*z_e.
No T×N×N tape, reverse sweep or full RTRL Jacobian. Arrays H×(D+K) only.

At supervised endpoint: error=onehot(target)-p from CURRENT localhead;
q=(F-headmean)/headscale, headscale=max(trainstd,.05), transformed=clip(q,-5,5).
L=(W_out.T @ error) * indicator(abs(q)<5)/headscale.
direction[j,i]=L[j]*f_e[j,i]. PRE-head-update W is used. Output-local delta
remains unchanged .08/entryclip.025/weightclip3. Feedback is ONE readoutstep,
explicitly non-local relative to scalarreward; NOT fully biologicallocal/exact
networkgradient. Per-neuron eligibility is forward-only and ignores recurrence
credit paths through other neurons. No error hidden→hidden.

If multiple supervised endpoints in later wordbridge, accumulate directions and
average over endpoints; do not alter weights inside a segment. This stage has
oneendpoint/12ticks. After endpoint apply all48rows with synapse-local Adam:
lr.001,beta1.9,beta2.999,eps1e-8, moments0, positive learningdirection;
delta perentryclip.002, weightclip3. NO globalgradientnorm reduction or anchor/
replay/consolidation/firing-rate regularization/optimizer selection. R12 is
metadata of old STDP, not a claim only12rows update here. Moments and counter
persist. Measured signalnorm may be logged, never used to couple rowupdates.

Conditions: correct, shuffled, zero, frozen, isolatedBPTTcontrol. Shuffled
permutes the neuron-specific L with RNG seed+35100000 at everytarget; head label
stayscorrect. Allcandidateconditions consume same permutations, including zero/
frozen. Zero calculates eligibility but suppresses hiddenupdate/moments. Frozen
skips traces and hiddenupdates. Zero/frozen MUST match neural/headweights and
predictions for the same stream, though trace state/cost differ. Feedback off
at inference cannot change forward predictions when weights are heldfixed.

MatchedBPTT ONLY evaluation/controls.py: SAME NumPyobserved hardspiketape,
WordTapeSurrogate(.4), globaltemporal graph, same frozenheadstats and preupdate
headW, same per-synapseAdam/lr/clips/all48rows, no globalnormclip. Head still
output-local. Disclose graphcompute/cost. Candidate never imports Torch/control.
Freshcontrol wrapper intentionally incompatible with main/candidate loaders.

## Engineering must pass before formal screen

Check derivative on a fixed PRESYNAPTICtape using independent forward finite
differences of a differentiable anchored surrogate: z(w')=observedz +
sigmoid(clamped margin(w')/.4)-sigmoid(clamped baseline_margin/.4). At w'=w
this matches actual hardtrajectory; at perturbedw it is mathoracle ONLY, never
modelinference. With other-neuron presynapticspikes heldfixed, finite-difference
localF matches f_e; eps1e-5,float64, atol1e-5/rtol1e-3,64synapses across4neurons,
input and recurrent connections. NOT a finite-difference claim on hardspikes.
Also verify forward bytes vs unmodifiedSpikingCore, locality/zero/shuffledsign,
no target in forward, calibratedfeedbackclipderivative and stableweights intact.

Own boundednonpickle/checksum codec: core/head, all v_e/a_e/f_e, directionbank,
Adam moments/counter, learningcounter, RNG, endpointmask/segmentcursor/trackflag.
Partialsegment→save/load→nexttick/endpoint/update exact; freshprocess continuation
on discardedcopy exact. Corruption/unboundedshape/nonfinite/negativevariance
rejected. Mainloaders reject prototype/control wrappers. Count every array and
checkpoint/rollback buffer in memory report. No serving migration.
Run targetedtests/fullpytest/nativeparity and registered P0audit before training.

## Locked temporal screen

Three fresh modelseeds35113/35119/35141 × order/xor/strict_binding ×5methods=45
trials. Use existing audited generators from tasks21/binding_audit21 unchanged.
Order is a weak delayed-order/recency task, not complexreasoning; always disclose
this and never claim success from its100%recall alone. XOR and strictbinding
require interaction of events; report bag-only, time-permuted and state-reset
readouts. Bag-only>60% makes a task invalid for representation gate. No hidden
trace may consume taskID, labels in input or futureexamples.

Per model/task: unlabeledcalibration128 sequences offset+1360000, train1024
offset+1370000, refitprobe128 offset+1380000, development256 offset+1390000,
OOD128 offset+1400000 for order/xor only. Refittaskhead ridge_probe(.01) is
diagnostic, not inference or learnedhidden signal. Datasetseeds=offset+modelseed.
Headcalibration on initialhidden TRAINcalibration ONLY, freeze before labels.
Only finalcheckpoint; no devpeek/epoch/LR/parameterselection. Refitprobe on
separate data, clearworkingstate/resethead, never use gold in core.forward.

Screen gate stays historical P1: mean refitprobe gain vs INITIALhidden>=5pp and
directaccuracy>=65% on at least2VALIDtasks; correct vs frozen probe advantage
positive on those tasks. Additionally include XOR OR strictbinding in any
positive interpretation; order-only recall never opens furtherclaim.
Screenfailure=STOP_DIRECT_READOUT_ELIGIBILITY_V1; no confirmation/scale/autoLRgrid.
Positive=CONFIRM_DIRECT_READOUT_ELIGIBILITY_REQUIRED, not learnedlanguage/vision.

If screen passes, preregistered confirmation modelseeds35201/35219/35227/35233/
35239/35243/35257/35261, SAMErule/dataoffsets/1024training. Compare correct/frozen/
shuffled/zero/BPTT, all3tasks, finaltest512. P1 unchanged: lower95probeGain>=5pp,
direct>=65%, correct−frozen positive and paired signflip/Holm across3tasks p<=.05
for gain and frozencontrast on>=2VALIDtasks. Report OOD/lesions. No selection.
If confirmation budget infeasible, leave confirmation pending, not successful.

Temporal zero/frozen equivalence and control learnability reported separately.
Failing BPTT also means task/forward/budget unresolved, not proof localimpossible.
P2acquisition>=65%/meanforget<=3pp/worst<=5pp unchanged; no retentionclaim here.
Normalchat/vision, experience-to-skill and mainweights require later tests.

## Resources / observation

CPU1thread, one Torchchild at a time, at most2 NumPyworkers, process-tree2GB
AND Windowsperprocesscap. P0cap180s. Temporalscreen180s/trial/1800sround;
confirmation180s/trial/3600sround. Forecast from TRAINonly64cases before formal
longrun, include initialization/evaluation/coldverification. Stop nonfinite/
source/format/locality/budgetfailure, retain logs, no restart just on timeout.
TechnicalIO/reportrepair only with unchangedmath/protocol/budget and no retraining
completedmodels. Same examples != equaltime; measure extra graph and tracebytes.

Publish actual loss/ΔW/changedsynapses/hiddenupdates/neuronframes and numeric
input/prediction/target to passive localhost, label PROPOSAL30 RESEARCH / NOT MAIN.
No fakeQwen or freechat on numeric tensors. No user selection/start required.
No main/privateweights/config change, no automatic control import into Hermes.
