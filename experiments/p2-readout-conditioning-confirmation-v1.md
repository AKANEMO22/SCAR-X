# P2 conditioned-readout confirmation and continual stress

Registered2026-11-09 AFTER development screen p2-readout-conditioning-v1.
Only normalized condition was informative; freeze scale floor0.05, clip5,
lr0.08/updateclip0.025/weightclip3, initial fixed deterministic ALIF48/K8/R12.
No LR/parameter/model selection after confirmation. No new hidden rule.

## A: independent confirmation

Seeds4103,4111,4127,4133,4153,4159,4177,4201. Same order/XOR/strict-binding generators
and data offsets+1,500,000train/+1,600,000fit/+1,700,000eval, but512eval;2048train,
512unlabeled calibration then1536labeled updates. Raw vs normalized, hidden
identical/frozen. Keep per-example predictions and refit-probe diagnostic.

Necessary conditioning evidence: mean direct>=65%, lower95CI paired gain>=10pp,
Holm-adjusted two-sided exact signflip p<=.05 on>=2tasks. This is an output-local
conditioning gate, NOT a hidden-representation gate or permission to deploy.

## B:20context continual stress, no future calibration

Same8seeds, existing `context_task` generator with20contexts/D24/O4,H48,K8,R12.
Each context32new training examples,128held-out test, total20stages. Data seeds
= seed+2,100,000+context*101 train; +2,200,000+context*101 test.
Calibration128unlabeled samples from CONTEXT0 ONLY, seed+2,000,000; freeze stats
before ANY labeled updates. No future-context data/stats/labels used in learning.
All heads have same initialized output weights/bias and fixed hidden features.
Context cues are already observable inputs of the historical task; do not claim
this is a task-boundary-free benchmark. Boundaries are supplied to scheduler.

Conditions raw, normalized, raw_replay, normalized_replay. Exactly128output-local
updates/stage. Without replay, repeat the32new samples4times. With replay,
slots k%4==3 use an earlier observed replay example if present, remaining slots
new sample. First stage uses only new samples. Replay capacity128, ordered observed
examples with context,label metadata; trim oldest duplicate from largest bucket,
keep>=1per observed context-label group where budget permits. Uniform over
observed groups, then over that group's items. Never add test items. RNGseed+2300000.

Evaluate every observed context after each stage, accuracy matrix. Report diagonal
acquisition, final average, max-to-final forgetting per context, signed BWT and
mean/worst per seed. Absolute master gates: acquisition>=65%,final>=65%, mean
forgetting<=3pp and worst<=5pp on EVERY seed. A model not acquiring fails even if
forgetting0. Do not claim complete P2 relative superiority without equal-time
confirmation; this trial matches update count only and reports all runtimes.

Check exact readout serialization/continuation, no hidden changes, no retrieval.
Mean/variance are not knowledge bank; old weights remain ordinary neural weights.
No good aggregate can hide a failing seed or context. No live replacement of
user weights, language pretraining, bigger model or iGPU/C++ optimization.
CPU1thread, sequential,2GBhard cap,<=180s overall; failure saved explicitly.
