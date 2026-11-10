# Read-only conditional input reliance audit

Registered2026-11-10 before intervention measurements. Supports master §4.1
causal representation/credit assignment diagnosis; not a learning adjustment,
encoder search, stopped-family reopening or deployment. v2 trial continues
unchanged. Use ALL nine immutable word-score v1 checkpoints27103/27109/27127
×frozen/full/shuffled. No v2 winner selection or private/live weights.

Exactly95publiccases: train16(2/family),dev31,prior48 from v1 sourcefiles.
At first generated word only (BOS, no previous/gold target word), compare three
predefined interventions on the SAME actual input event arrays:
1. real: actual promptcontext.
2. zero: set only128contextchannels to zero; BOS/control/boundary unchanged.
3. cross_family: replace context with firstcase of next of the8fixedfamilies
   cyclically IN THE SAME split; use intent metadata only to assign this input
   intervention, never as neuralinput/readout/templateanswer.
No changing token length, noise, weights, neuron count, codec or inference rule.
Batchall95 with existing purefeatures_many MC8,15ticks; common RNGseed991000
within each model across interventions. Batched RNG differs from per-case v1
decoder; report a NEW diagnostic, never rewrite v1 scores. Singlecase forward
parity test with existing distribution atsame seed before measurement.

Report firsttoken accuracy/NLL vs actual targets, argmax changes, probabilityL1
and hiddenfeatureRMS vs real. Real-vs-zero/cross accuracy gains and seedbootstrap
CI95 descriptiveonly (3independentmodels/condition). No pass threshold/learner
selection from this audit. Not fullutterance/normalchat/vision/retention scores.
Probability changes prove sensitivity, not useful semantics; similar accuracy
alone cannot prove the network ignores context, since many targets share a
firstword. Keep all raw public firsttokenpredictions and caveats.

No parameter/RNG/calibration/state mutation within the diagnostic after setting
existing firstwordmask on each discardedclone; sourcecheckpoint hashes must
remain unchanged. Actual coreoutputs only, no diagnostic fittedanswerhead.
CPU1thread,2GB cap,120s total; batched input/feature memory counted. No new
teacher call/data/labels/training. Publish technicalprogress/actualmetrics,
do not label a one-word probe as a whole model reply in answertrace.
