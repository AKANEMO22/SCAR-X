# Final word-score application round: fixed input codes v2

Registered 2026-11-10 BEFORE implementation/training/label measurements.
Master §4.1/P1 representation and numeric tokenizer permission. One ALIF
recurrent core with local synaptic learning; no Transformer, pretrained weights,
template answer selection, backward or BPTT. FINAL bounded application round.
Preserve STOP_WORD_SCORE_TRANSFER_V1 and STOP_INPUT_IDENTITY_CONTROL_V1.

Different question from the stopped numeric intent-head control: can a fixed
distributed input code help actual next-word local neural training? No further
numeric intent-classifier, vocabulary-size/hash-seed/feature-performance search.
The299-D reference is not a neural solution; neural dimensions stay unchanged.

Hypotheses fixed before measurements:
H1: dense code FULL yields ≥10pp mean DEV free utterance gain over legacy FULL,
positive on ALL three model seeds.
H2: dense FULL beats dense FROZEN by ≥10pp mean DEV, positive on ALL three seeds,
and ≥10pp mean on PRIOR diagnostic.
H3: sparse-vs-dense with identical words/IDF distinguishes coverage/selection
from code geometry. Descriptive only; never select an encoder after results.

Three fixed structural ablations, no hyperparameter sweep:
* legacy: ScoreWord v1 byte-identical numeric encoding/checkpoint behavior.
* known-sparse128-v1: ALL train-prompt alphanumeric word types, no truncation;
  document IDF log((1+N)/(1+df))+1. BLAKE2b digest8 signed bucket128 (existing
  formula), IDF weighted counts, divide by max(1,L2).
* known-digest128-v1: EXACT SAME words/IDF/counts; BLAKE2b digest16, unpack
  big-endian bits to ±1/sqrt(128), sum weighted vectors, divide by max(1,L2).
No char trigrams in known-word variants. Their difference isolates geometry;
legacy comparison also changes inventory/IDF/char treatment. No semantic or
pretrained embedding, labels or answers construct codebook/IDF. Unknown words
ignored, coverage reported. Bag representation still loses order.
Metadata ≤65KB/4096 word types; fail rather than truncate if over. Exactly128
input channels; same trainable arrays/parameters/H48, no extra embeddingweights.

Fresh MODEL seeds28001/28019/28027 ×three encoders×frozen/full/shuffled=27trials.
Same181train (64authored+117reviewedAntigravity),31dev and48prior inspected
diagnostics; not fresh final test or semantic-template-disjoint data. Pin source
files/SHA256. Existing actual AGY data and archived exact-prompt Qwen outputs;
no new teacher call. Output vocabulary train-targets only,8style responses;
no general language/vision claim or privateweights/feedback.

Reuse unchanged ScoreWord v1 core/noise/calibration/decoder and runner:
H48/K8/R12,leak.65/threshold.5/fdecay.25/adaptstrength.02; MC8,temp.4;
four rotating rowblocks×two independent rollouts, hiddenAdam.002/.9/.999/eps1e-8,
clip.025/weightclip3, output-local delta lr.08. Uniform8family sampler,1024
responses,seed+10200000. Calibration128unlabeledtrainprompts/randomprefix,
seed+10100000; different encoding changes measured stats, not the algorithm.
Same samples/noise schedule per seed across all conditions. No hiddenrule/LR/
epoch/grid tuning. Current target never enters input; teacherpreviouswords only
in trainingprefix; inferenceprefix exclusively neuralgeneratedwords. Canonical
prefix replay experimental, different from servingcontinuousstate.

Final1024checkpoint only, no early selection. V1metrics unchanged: train16,
dev31,prior48 strict/normalized freeutterance exact, teacher-tokenaccuracy/NLL,
firsttoken,EOS/repetition,ticks,loss,ΔW,updatecounts,checkpointcontinuation.
Paired seedbootstrapCI95 exploratory: three seeds do not prove general ability.
Predefined DENSE informative ONLY if full devnormalized≥40%, H1 AND H2 pass,
every coldresume/continuation/target-noleak/shape/parity check passes. Do not
substitute sparsewinner, teacherforcedtokens, trainingexact, intentclassification
or Qwen selfclaims. Report every outcome.
Positive=FRESH_VALIDATION_REQUIRED; no servingimport/P1/P2/vision/normalchat
success or scale permission. Negative=STOP_WORD_SCORE_INPUT_TRANSFER_V2;
close word-score application after two rounds. No extra encoder/vocabulary/
epochs/LR/beam/hiddenrule/modelsize search or renaming these same rounds.
Future directions require a distinct cause/rationale. Retentiongates unchanged.

BEFORE training test legacybytes/exactfuturecontinuation from priorv1artifact,
train-onlycodebook, targetfreeinputs, sameweightsizes, frozenhidden, nonpickle
codec/metadatabounds. Save inputmetadata/RNG,allneural/moment/calibrationstates.
Do not change servingUnifiedCore/SpikingCore/PairedScoreCore or oldformats.
Optional encoderfield in experimentalwordwrapper only; absent stayslegacybytes.

CPU1thread/worker,two workers,240s/trial,2400s/round,2GBtree monitored plus
existing Windows per-process hardcaps. Stop NaN/malformeddata/formatmixing/
leakage/budgetfailure. Expect10–15min fromv1 timing; no >72h run. Publish actual
answers/learning/neuron samples on passive localhost as measured. Publicmodels
only; never overwrite live userweights or start extra job just to fill UI.
