# P1 local classifier error / eligibility reference adaptation v1

Registered 2026-11-09 before implementation measurements. Master P1 compares
local representation learning; no larger core, new framework or release. The
stopped local-target-rate/random-main-error family stays NO-GO. This distinct
objective minimizes a local auxiliary classifier's error, NOT neuron target
rates or the task readout's error. Report its distinct signal path and timing.

## Source and mathematical review

DECOLLE: Kaiser/Mostafa/Neftci2020,
https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00424/full
and official https://github.com/nmi-lab/decolle-public . Original code/paper use
their own neuron/filter dynamics and multilayer networks. This is a MANUAL
NUMPY adaptation of the auxiliary local-loss idea to SCAR-X's existing ALIF,
NOT a faithful reproduction, a new theory or a claimed published result.
No copying/importing pretrained weights, autograd or external runtime.

Same deterministic ALIF48,K8,R12,D8,O2. A FIXED random local classifier G[O,H]
drawn N(0,1/sqrt(H)), RNGseed+3100000. It reads filtered spikes f, computes
`p_aux=softmax(G f)` and local postsynaptic signal
`m=G.T*(one_hot(y)-p_aux)`. G is never copied from task W_out, trained, or used
at inference. The supervised label is feedback, never part of input x. This
broadcast is nonlocal teacher information and must not be called unsupervised.
It is an auxiliary within-group local error, not task readout backward or
propagation to other neurons' states/time steps.

Forward presynaptic EPSP trace q=leak*q+presynaptic_activity, where activity
is current input or previous source spike. For selected rows j:
`psi_j=0.3/A_j*max(0,1-abs((v_pre_j-A_j)/A_j))`
`elig_ji=feature_decay*elig_ji+(1-feature_decay)*psi_j*q_i`.
At feedback: `dw_ji=clip(lr*m_j*elig_ji,+-update_clip)`, then weightclip3.
Stop-gradient through adaptation, refractory reset, other neurons and past
states; derivative is an approximation, not an exact global network gradient.
No W_out transpose or main-error feedback. Same incoming indices/rotating rows.
Local tangent finite-difference oracle tests update sign/scaling against this
specified auxiliary objective, not derivative of the hard threshold itself.

## Design before data opening

Fresh development seeds11113,11117,11131. Existing order/XOR/strict-binding
generators, 2048train/128refitfit/256development, offsets+3300000/+3400000/
+3500000; OOD128offset+3600000 for order/XOR, binding OOD not supplied.
Same existing Config defaults except H48,K8,R12,D8,O2,anchor0. Main readout uses
local normalized features with mean/std from128UNLABELED calibration episodes,
seed+3200000. Statistics frozen before ANY labeled learning; same calibration
and initial hidden/output weights across all conditions. No ridge runtime.
Scalar-STDP baseline uses the same normalized task head's pre-update target
probability for its original scalar EMA advantage; no raw/normalized double
readout update. It is explicitly scalar-STDP + normalized head, not the old
unconditioned Phase2configuration. Initialhidden is a shared refit control.

Conditions: initial-hidden/readout-only, original scalar-STDP (hidden_lr0.002),
aux_local_lr0.0002/0.002 and shuffled_local_label for BOTH LR. SIX conditions,
3tasks*3seeds=54developmenttrials. Auxiliary feedback during last4ticks of each
12ticktrainingepisode; readoutoneupdate at end for all conditions. Shuffled
labels independently drawn from candidate RNG, task head keeps correct label.
Candidate has4extra hiddenupdates/episode vs scalar1; report compute and update
counts, do not claim same hidden-update or equal-time advantage.

Refit head: train-only centered/scaled ridge0.01, same fit/test across conditions,
measures trained vs INITIAL hidden; actual task head measured separately. Temporal
permutation/resetstate lesions only in diagnostic clones. Measure hidden-change
norm, main vs aux objective, direct and refit accuracy, and resources. No replay,
retrieval, generation/caption templates or targets during inference.

Select LR by mean DEVELOPMENT probe gain across3tasks/3seeds, tie smallerLR.
Screen qualifies only if gain>=5pp AND directaccuracy>=65% on at least2tasks.
Order can be near ceiling; report initial headroom and do not substitute easy
recall. If screenfails stopv1; at most one further separately registered
adjustment, never unlimited tuning. Do not openconfirmation if fails.

If screenpasses, NEW8confirmationseeds11503,11519,11527,11549,11551,11579,11587,
11617, same settings but512test. Freeze selection before test. Compare selected
aux/readout/scalar/shuffled, seed-cluster bootstrap95%, exactsignflip/Holm3tasks.
Historical P1gate remains lowerCIgain>=5pp,direct>=65%,positive gainvsreadout
with adjusted significance on>=2validtasks. No P2retention or normalchat/vision
claim; a P1pass requires separate P2 before release.

CPU1thread,2workermax,180s/trial,1200swhole,aggregateRAM2GB, no Qwen concurrently.
Own boundedchecksummedcodec includes G,EPSPtraces/eligibilities, neuron state,
head stats, RNG, aux/mainupdate counters. Prove forwardparity/readoutindependence,
local tangent derivative and exactcheckpointcontinuation before running. No
serving import/weights migration. Missing matched learnability control limits
negative conclusions; failed adaptation is not proof DECOLLE/SNN impossible.
