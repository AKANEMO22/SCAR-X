# P1 auxiliary classifier: final conditioning/timing adjustment v2

Registered2026-11-09 BEFORE new measurements. Final permitted adjustment of the
local-auxiliary-loss family after v1 failed. Not another local-target-rate or
random-main-error round (those remain NO-GO). Master P1, same ALIF48/K8/R12,
D8/O2, topology/forward. No serving import, increased model or language corpus.

Hypotheses (not established causes): raw-rate auxiliary logits/eligibility have
poor conditioning compared with the normalized main head; giving binding's
target at ticks8/9, before query10, adds ambiguous early supervision.
DECOLLE source: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00424/full
This remains a manual ALIF adaptation, NOT a reproduction or a new theory.

## Frozen design

Development seeds12101,12109,12113; confirmation12503,12511,12517,12527,12539,
12541,12547,12553. Existing order/XOR/strict_binding generators. Offsets
calibration+4200000 (128UNLABELED), train+4300000 (2048), refitfit+4400000(128),
development/finaltest+4500000(256/512), OOD+4600000(128order/XOR only).
No previous test used to select/tune this run. Main head uses the same frozen
calibration mean/std floor0.05 and clip5 for ALL conditions.

Two auxiliary views: raw features(v1) or
`r=clip((f-mean)/scale,-5,5)`, scale=max(train-only std,0.05).
Fixed G drawn with seed+3100000 as v1. `p=softmax(G r)`;
`m=G.T(onehot(y)-p)*I(abs((f-mean)/scale)<5)/scale` for normalized view;
raw remains v1. Frozen statistics are not learned from targets/test or updated
after hidden changes. Local EPSP/eligibility unchanged fromv1. No task W_out
feedback, cross-neuron gradient transport, reverse-time tape, autograd or BPTT
in the candidate. G transpose transports the local auxiliary error (nonlocal
teacher information); disclose this, do not call it strictly synapse-only.

Timing: LAST4=one hidden update after ticks8,9,10,11; FINAL4=four hidden updates
after tick11, with no intervening neural step. Both4hiddenupdates/episode,
12forwardticks, same targets/weights/RNG and local main-head1update/episode.
Repeated final updates do not constitute4additional thinking ticks.

12 configurations/task/seed=108developmenttrials:
readout-only, scalar-STDP(LR.002), raw-LAST4(.002), raw-FINAL4(.002),
normalized-LAST4(.0002/.002), normalized-FINAL4(.0002/.002), and shuffled
auxiliary-label versions of each normalized condition. Shuffled condition
keeps task-head label correct. No replay/retrieval/reset between inference
tokens (tasks are discrete diagnostic episodes, reset at their boundary).

Select normalized timing/LR by mean development refit gain over3tasks/3seeds;
ties smallerLR then LAST4. Screen: mean probe gain>=5pp AND directaccuracy>=65%
on at least2tasks. If fails, STOP family with NO_GO_LOCAL_CLASSIFIER_FAMILY;
no confirmation, third adjustment, relabeling drift as representation or scale.
If passes, freeze selection file, then evaluate8NEWconfirmationseeds on all
3tasks/5conditions: selected, corresponding raw, shuffled, readout, scalar.
Gate unchanged: lower95 seed-bootstrap gain>=5pp, directmean>=65%, positive
paired advantage vs readout and exact sign-flip Holm(3tasks) p<=.05 for both
gain and readout contrasts, on>=2validtasks. Report OOD/permutation/reset lesions
and initial headroom. No P2retention/language/vision/commercial parity claim.

## Learnability and accounting

Regardless of candidate outcome, isolated controls in evaluation/controls.py:
same ALIFforward/topology/initialization/main normalized head, readout-only
vs surrogate BPTT hidden learning.3devseeds×3tasks×2=18controls,2048examples each.
Output head uses existing local clipped delta(.08); hidden BPTT uses Adam.002,
gradientnorm1, perweight updateclip.025 and weightclip3. Full reverse-time graph
exists ONLY in this control. Its surrogate is psi=.3/A*max(0,1-abs((vpre-A)/A)).
No feedback through G in controls; task W_out feedback is explicit BPTT.
Compare actual end-to-end/refit accuracy, input statistics, weight norms/time.
Same examples/readoutupdates do NOT mean equal compute; no efficiency advantage
claim from these counts. Hidden update count/time/arrays+optimizer buffers/RSS
are reported;4local vs1BPTTupdate/episode, allhidden vsrotating12rows disclosed.
If controls also fail, negative conclusion applies to this implementation and
budget; no claim that published DECOLLE/local learning/SNN are impossible.

CPU1thread, localworkers2; isolatedTorchcontrols sequential after workers exit.
Pertrial180s, whole2700s, process-treeRAM2GB and bounded buffers. Stop on NaN,
bad checkpoints/locality/leakage or budget. No paid teacher/Qwen server.
Forward parity, local-proxy finite-difference, frozen stats, checksum/state/RNG
continuation tested BEFORE trial. Local codec includes G,stats,EPSP/eligibility,
neuron state,rows,RNG,counters. Control artifacts are diagnostic weights/head,
not a production learner/optimizer-resume claim.

Every512trainexamples publish measured loss/ΔW/update counts to localhost;
publish actual first3heldout predictions per trial with tensor input/target and
explicit classification scope (not chat). Qwen field missing for these tests;
never fabricate natural-language/Qwen output. Preserve all predictions/source
in unique run. Receipt marks whether any confirmation was opened. No target
from diagnostic corpus enters live private session; public release unchanged.
