# Contrastive output-error eligibility, GPU-only DEV pilot

Authorization: user15/11 continues necessary isolated changes toward real SNN
advantage; latest request explicitly forbids CPUtraining and asks autoGPU.
Every neural forward/local update in this study is CUDA0; CPU only loads data,
serializes, rescores and bootstraps. Do not rerun old CPU/GPU trajectory tests.
ALIF H128/input64/fan8/FP32 equations/topology unchanged. No whole-network/time
backward, Transformer replacement, scaling/quantization, LoRA or live promotion.

## Evidence and gap, literature before code/train

0748ea prototype rule TEST62.64% vs BM25 78.47%, ridge83.33%; mean forgetting
10.83–16.67pp, worst35–70pp. HistoricalFAIL unchanged. Read-only b71d81 found
common-head DEV63.89% vs frozen62.50%, gain1.39pp CI[-5.83,8.61]: no convincing
general representation improvement. Increasingticks32 to62.78% helps coarse
spiking but does not match semantic rawfeatures; analog prototype67.50% also
weak. Old-targetpool restore with final competitors does not recover earlier
scores: forgetting includes competing class boundaries, not just targetoverwrite.

Read author DECOLLE paper https://pmc.ncbi.nlm.nih.gov/articles/PMC7235446/
equations6–8 and https://github.com/nmi-lab/decolle-public/blob/master/decolle/base_model.py
(LIFLayer rather than documented legacy BaseLIFLayer dynamics) plus utils.py
state_detach. Author implementation uses PyTorch surrogate backward locally;
we do not copy GPLv2 code or add Torch. Adapt only principle: explicit task-error
feedback times local surrogate/presynaptic trace, with all temporal state forward.
Reuse SCAR-X LocalALIF.step's already audited neuron-local ALIFeligibility; no
cross-neuron Jacobian recursion. This is a newheuristic, not DECOLLE reproduction
or exact gradient. Existing e-prop/one-step-error userapproval30 reused, don't
rerun paperbenchmark or claim published scores. SoftHebb authorREADME/paper read
as alternative: unsupervisedWTA+separate classifier not clear solution to this
supervised negative-evidence gap, defer without another experiment.

Hypothesis: positive-only prototype imprinting ignores class competitors and
maxpool spike ties; a fixed pooled readout with signed output-error eligibility
can train hidden synapses to discriminate useful margins. No momentum/feature
stat drift so this tests one error-alignment mechanism. No new dataset encoder.

## Exact rule and registered two DEV candidates

Freshseed260113 only, original normalized128inputrows and recurrence, fixed
f_mean0/f_var1. Use8constant-input ticks, track forwardeligibility. Readout G is
fixed6x128, each of21neurons perlabel has +20/21 to its ownclass, rest0;2reserve
neurons no output. Scores G*actualALIFfeatures; softmax and local output error
d=onehot(label)-p. Hidden signal=G.T*d, same approved one-step-error relaxation.
Input-onlydelta=clip(eta*signal[:,None]*e_f[:,:64],-.025,.025), clipweights[-3,3].
No hidden normalization afterupdate, no adaptive optimizer/momentum/importance
in this pilot. Recurrentw_rec constant. Never substitute analogcosine answer.

Two predefinedrates eta=.05 and.20 (onlydifference); fixed hidden control uses
same G and initialization. Readout-only control learns G/bias with lr.04/clip.025,
same spikefeatures/fixedstats/input/recurrent. Output-onlycontrol shares initG.
Labels known only duringTRAINupdates. Allarms same20stage TRAIN/replay64 schedule
as priorseed260113,24attempts/stage.120DEV onceaftereachstage,0TESTcalls. Two
candidate rates are a small DEVmechanism screen, not independent3seed evidence.
Stop family if neither acquisition>=65% DEV and finalmeanforget<=3pp/worst<=5pp;
also report strongest control gap, unchangedgain5pp andCI gates still required
for any laterconfirmation. Do not call pilotPASSfullquality. No selecting/rerunning
more rates iffail. A subsequent study needs a genuinely new hypothesis/protocol.

## Technical P0 then pilot, hard budgets

Max1500updateattempts includingGPU-only P0/continuation,12000forwards/180seconds,
PythonRSS2GB/512MiBdedicatedreserve/artifacts30MB. Newledger separate hypothesis,
prior4374update/37722forward budgets remain closed and charged, norefund.
P0: twoGPUclones execute6synthetic updates total, coldcheckpoint exact, next
GPUupdate exact, strict type/duplicate/researchformat rejection via puretests.
No CPUmodel.learn or CPUforward. Nonfinite/guard/persistencefailSTOPbeforeDEV;
preservefailreceipt and ledger, don't hide by retry. Every stage actualloss/
scores/deltas/spikes/device posted tolocalhost; live counters/weights unchanged.

## Final test lock

Official240TEST rows already consumed for originalstudy: never use their new
predictions for model selection or freshconfirmation. This pilot uses original
TRAIN240/DEV120 only. Before any laterconfirmation, lock disjoint unused human
utterances from the original pinnedTRAIN source (remaining normalizedtexts),
excluding all600priorselectedrows, without seeing modeloutputs. If insufficient
or affected by overlap, report limitation rather than manufacture independence.
