# P1 paired score eligibility v2 — final score-family adjustment

Registered2026-11-09 BEFORE training measurements. Second/final adjustment of
the original Phase2.1 stochastic score family, not a third local-classifier or
local-target-rate/random-feedback round. Those families stay NO-GO. MasterP1,
same stochasticALIF48/K8/R12,D8/O2. No serving import/model growth/language run.

Preflight p1-score-integrity-audit-v1 matched exact expected reward derivatives
on finite2neuron paths. Antithetic pairing reduced sampled variance only about
10–13% in those fixtures; no promise at48neurons/representation tasks.
Context: https://doi.org/10.1007/BF00992696 and
https://neuronaldynamics.epfl.ch/online/Ch19.S4.html . This is a manual discrete
BernoulliALIF likelihood-rule adaptation, not a new biological proof.

## Rule and signal audit

Maintain forward presynaptic sensitivity `q=leak*q+pre` and accumulate
`e_ij += (z_j-p_j)/.4*q_ij`, zeroing contributions at clipped-logit saturation.
No temporal derivative graph, taskW_out transpose or error feedback intohidden.
At each block, two trajectories start from canonical episode state, same weights
and fixed head; inputs include no targets. Each has its OWN recurrent presynaptic
history/eligibility. Pair uniforms independentU,V or antitheticU,1-U.
Always draw U,V, discardingV in antithetic mode, to align samplerRNG consumption.

RewardR=-main-head NLL. Prior EMA baseline b is frozen for BOTH trajectories;
`g=.5*((R1-b)*e1+(R2-b)*e2)`. No pair-dependent baseline, reward-difference
shortcut or shared recurrence trace. Update EMA after pair. Label enters only
local readout error/scalar reward, not neural input. Raw score estimator has a
likelihood-gradient interpretation under ideal probabilities; finiteprecision,
EMA/history, clipping and adaptive optimizer mean the FULL LEARNER is not
claimed to be an unbiased exact gradient, e-prop or BPTT.

Four consecutive rotating12row blocks per observed example cover all48rows;
2trajectories×12ticks perblock=96forwardticks/example. Block weights update
only after pair; head stays fixed throughout4blocks. One local delta head update
on mean of8features afterward, fixed normalized statistics from unlabeled
calibration. No neuron-count increase; report8×rollouts vs historic1rollout.
All conditions below match these forward/head/hidden-update opportunities.

Legacy-matched conditions clip e to10 eachtick and advantage to2, use SGD.002
or per-synapseAdam.002. These are Phase2.1-inspired matched baselines, NOT its
old numerical results (normalization/coverage/rollouts differ).
Candidates use untruncated e/advantage, AdamLR.0002/.002, mean/secondmoment
.9/.999,epsilon1e-8 and per-row active-update clock. Local running moments use
only that synapse's estimated signal; no task gradient or cross-synapse mixing.
All updates clipped.025,weights3. Shuffled control randomly flips scalar
gradient sign perblock; readout label remainscorrect. No replay/consolidation.

## Locked data/design

Development seeds14207,14221,14243; confirmation14503,14519,14533,14537,14543,
14549,14551,14557. Existing order/XOR/strict_binding generators. Offsets:
unlabeled calibration+6200000(128); train+6300000(2048); refitfit+6400000(128);
development/finaltest+6500000(256/512); OOD+6600000(128order/XOR only).
Feature evaluation averages8independent stochastic rollouts with fixed common
noise, stateless and not mutating learnerRNG. Initial and trained hidden use
same fit/test/probe/noise. Calibration initial-hidden8rollout means frozen
beforelabels; scale floor.05, normalizedfeatureclip5. Probe ridge.01 diagnostic.

12configurations×3tasks×3seeds=108development trials:
frozen-independent, frozen-antithetic, clippedSGD-independent(.002),
clippedAdam-independent(.002), unclippedAdam independent(.0002/.002),
unclippedAdam antithetic(.0002/.002), and shuffled versions of BOTH unclipped
pairing conditions/LRs. Same initialization/data/noise schedule where meaningful.

Select one unclipped pairing/LR by mean dev refit gain across3tasks/3seeds;
tie smallerLR then independent. Screen unchanged: gain>=5pp AND direct>=65%
on>=2tasks. Failure means STOP_SCORE_FAMILY_V2, no confirmation/thirdround or
largerdata/model. If passes freeze selection before8freshconfirmationseeds:
selected vs matchingfrozen, clippedSGD, clippedAdam and matchingshuffled,
all3tasks. Representation gate unchanged lower95gain>=5pp,direct>=65%,
positive pairedprobe advantagevsfrozen, exactsignflip/Holm3tasks p<=.05 for
bothgain andfrozencontrast on>=2validtasks. Report reset/permutation/OODlesions.
No retention/language/vision/commercial-parity claim from this screen.

## Validation, limits and compute

Before trials: fixed-uniform forward matches Phase2.1 ALIF, oracle-inspired
score tests, independent baselines, inactive rows/moments frozen, permutation
control, checksum and exact partial-neuron/eligibility/moment/RNG continuation.
Own bounded nonpickle codec, no migration into public/private main session.

CPU1thread/worker, at most2workers, process-tree2GB,180s/trial,3600sround.
Matched baselineforward learnability is stochastic; prior deterministicBPTT
controls are NOT matched for this run. If candidatesfail, conclusion limited
to this family/budget; no claim all local learning impossible. No new BPTT
control imported into main. Record actual time/updates/buffers; no efficiency
advantage from8rollout budget. Full gate/retention remain unresolved until
separate confirmed experiments; do not tune after unseen test opening.

Publish actual512example loss/weight deltas, hidden update counts and real
heldouttensorpredictions with scope; no fake chat/Qwen. Corpus/Qwen reference
from prior work remains retained, not automatically imported into synapses.
