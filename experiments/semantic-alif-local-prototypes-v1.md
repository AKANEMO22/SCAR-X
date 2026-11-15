# SNN representation pivot: local prototype synapses on real utterances

User15/11 requests continued isolated changes until a genuine SNN advantage is
demonstrated. Keep all scientific gates; never guarantee a positive result.
No main/R13/liveweights, model scaling, Transformer replacement, precision drop,
new LoRA or automatic promotion. This is a module hypothesis, not a full-system
claim. Two closed six-macro tool banks stay closed; LoRA gains are not SNN gains.

## Literature-first / discriminating hypothesis

Read CLP-SNN authorREADME + models/CLP_SNN.py and Replay.py: supervised
prototype allocation, winner-only local three-factor normalization and
per-neuron learning-rate stabilization, with fixed pretrained feature encoding.
Paper2511.01553/OpenLORIS scores cannot transfer to ALIF64D or natural language.
No code copy/Loihi/quantization; use the idea as a new explicit algorithm.
Existing e-prop/DECOLLE/source-pins and old retention/protection failures are
reused, not rerun. Static representation quality is a prerequisite to temporal
reasoning, not proof that the network learned relations across video events.

Banking77 original paper2003.04807 + authorrepo PolyAI-LDN/task-specific-datasets
licenseCCBY4.0, pin57ec275d8078af65b7731c2a98be812d844a6d6b read before download.
Natural annotated utterances avoid another hand-coded tool-error table whose
oracle is already matched by a constant action. Six confusable fee/exchange
intents chosen by taxonomy before reading samples/scores:
card_payment_fee_charged,cash_withdrawal_charge,transfer_fee_charged,
card_payment_wrong_exchange_rate,wrong_exchange_rate_for_cash_withdrawal,
exchange_rate. No selecting another set if results disappoint.

Per intent40TRAIN+20DEV from officialtrain,40TEST from officialtest, deterministic
shuffle seed260216 per label.360train-derived +240officialTEST utterances,
exact/normalized duplicates reported and disjoint selection enforced before
features; don't remove model failures. Human dataset labels are a limited intent
rubric, not an oracle for financial facts or general language understanding.

## Shared frozen semantic encoder / information path

Original nativeQwen0.8B BF16/F32 encoder, no training/adapter. Private embedding
worker uses same pinned model with --embeddings --pooling last, not liveHermes.
One live endpoint probe already returned501requiring this flag; keep receipt.
E0 must verify finite1024D FP32 vectors, norms/nonconstant outputs and physical
GPU512MiB reserve before corpus extraction. Max610embedding calls/10minutes,
no generation/search paidAPI/private usertext. If unsupported STOP, no silent
new model/package. Preserve rawvectors and request/source hashes.

PCA64 fitted TRAIN vectors only, no labels/testfit; sample normalizationFP32.
All SNN/readout/classifier controls use the same64D codec. RAGsemantic1NN/5NN
also gets original1024D controls so compression doesn't cripple retrieval.
BM25, class centroid, regularized linear readout controls included. RAG memory
can keep all seen TRAIN examples onSSD, no artificial64record cap. Testseen
context from pretraining is possible, disclose scope and no general AI claim.

## New explicit local plasticity, same ALIF kernel/capacity

ALIF inputs64,hidden128,fan-in8,outputs8, nativeFP32 unchanged.6fixed label pools
of21neurons,2reserve neurons; no neuron growth, synapse deletion or fakeRAMclaim.
Every neuron/synapse remains allocated. Train a prototype by changing a hidden
input synapse row locally; sparse recurrent weights/kernel preserved. Fixed
ordinary readout pools spike features, no readout fitting in prototype arm.

On a supervised sample, label broadcasts a pool eligibility gate. A new
prototype can be imprinted into an unused target-pool neuron at the current
presynaptic feature (explicit supervised plasticity pulse, not a spontaneously
discovered rule). Once allocated, target-pool winner uses local normalized
competitive update: w_j <- normalize(w_j + alpha_j*(x-w_j)), alpha_j=1/(1+n_j).
No reverse graph/time gradients; per-neuron count stabilizes learned filters.
Select winner from the neuron's own input drive in TRAIN only, not hidden gold
features during inference. Prototypegeometry/labels are inductive bias and must
not be advertised as general deep representation or temporal credit solution.

At inference every answer comes from an actual recurrent ALIF forward and
spike-feature readout; never return a separate nearest-prototype cosine argmax
while pretending neurons decided. Count/trace nonzero spikes and actual GPU
placement. Initialization, fixed pooling and codec version are saved explicitly.

## Gate and fixed comparison

Before SNNtraining, measure TRAIN→DEV retrieval/NCM/ridge controls. If strongest
practical control exceeds95% so5pp advantage is impossible, STOP this study
without training/tuning or changing intent subset. Keep best registered control.
If headroom exists, seeds260113/260137/260171,20external stages focused on six
labels cyclically,12new exposures +12replay attempts/stage, reservoir64, no stage
ID in input.240current exposures perseed are repeats of240uniqueTRAIN candidates
only as stream dictates; log actual unique counts. No TEST/reward leakage.

Arms: locally adapting prototype-hidden+fixed-readout, same initial hidden
weights frozen+local learned readout, frozen model, shuffled-label local hidden
control.6headwidth uses already allocated readout slots2:8; same128neurons,
same input/recurrent initialization/source/samples perseed. Save actual perstep
weight changes and outcomes. Fixed prototype pooling vs learned readout is a
declared algorithm difference; don't call it an exact-credit/magnitude match.
Hidden-reset and no-plasticity ablations on clones, normalization fixed/shared
so resetting hidden doesn't carry a postfitstatistic confound.

Max4500local update attempts,40000CUDAforward attempts,20minutes wall, worker
2GBhost/512MiBdedicatedGPUreserve/artifacts100MB, single precharge ledger with
no reset/refund/reseed. No originalHistory177call budget reuse. Numerical/source/
device/nonfinite/persistence failure STOP, retain failedreceipt, don't retry.

DEV trajectories20stages and original held-outTEST once final; no choosing a
winner fromTEST. Acquisition>=65%; meanfinalforgetting<=3pp and worstintent<=5pp
with positive acquisition; gain>=5pp/lower95%pairedseed/taskCI>0 vs strongest
retrieval/rule control and readout-only; hiddenreset/shuffle must support actual
hidden-learning contribution. Report samplecount/seed uncertainty and population
limits honestly. Same-GPU save/load inference exact; no CPU/GPU trajectory
parity certification or relabeling of prior hard-thresholdFAIL.

Only a qualified intentmodule on this declared subset may be called better;
full assistant/video/long-term personal learning remains a separate goal.
Publish actual embeddings/proposals/spike activity/loss/weightdelta/scores to
localhost, marked as labwork not live neural counts or readable thoughts.

Source-split preflight before embeddings: first source-only intake rejected a
TESTrow overlapping the selected TRAIN text, so it could not collect40unique
TEST for transfer_fee_charged. Preserve that failed source-only run; no model
score or update occurred. Correct split construction reserves all official
TEST normalized texts before selecting TRAIN/DEV, keeping all official40TEST
rows/intent. Report withinTEST normalized duplicates and cluster them inCI,
not treating duplicate official rows as independent. No intent/gate/model is
changed to fit scores, source cache is reused rather than downloaded again.

Encoder resource compatibility amendment before any quality score: v1 run84f936
performed one public E0 embedding, then stopped at the2GBworker resident guard.
It unnecessarily inherited the live vision projector and16K context. Preserve
that STOP. V2 is text-only: remove unused --mmproj, use512-token context and
batch/ubatch512 (no truncation; oversized input rejects). Same original native
text weights/precision/source, same data and2GBworker/512MiBGPU guards. No neural
training. Charge original1call/1.937seconds against combined610call/600second
budget; no reset/refund. No quality selection has occurred. If this reduced
buffer configuration also fails, stop encoder study, not increase its cap.
