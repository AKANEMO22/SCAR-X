# Technical audit of completed fixed-input neural transfer

Registered before audit measurements. Read completed27trials, no rerun,
selection, new qualitythreshold, trainingdata or servingmigration. Keep source
protocol/metrics/STOP decision immutable.

For every checkpoint: validate saved seed/condition/encoding/1024responses,
H48/K8/R12 and equal neural arraybytes; known-word metadata must equal a fresh
UNLABELED train-prompt-only fit. Frozeninput/recurrentweights must equal initial
model; full/shuffled hiddenweights must actually differ. Inspect finite arrays.
Check all saved artifact/source/protocol/data SHA256 against manifest, restricting
code equality check to files consumed by neural training (UI may change).
Verify every trial<240s, total<2400s and measured memory<2GB; report actual caps.

Cold continuation:27 separate fresh Python processes, one at a time. Load source
checkpoint, learn ONE known public example on a DISCARDED copy, then generate
one reply with saved decoderRNG; compare fullbyteshash/loss/reply to same
operations on a discarded parent copy. Never write modified model or report
this training point as quality. Sourcecheckpoint SHA unchanged after audit.
Serving core.py/unified.py hashes must match neural-core-boundary receipt.

CPU1thread/child,2GBcap,120s audit limit. Failure invalidates technical readiness,
never overturns failed quality. Publish actual check results, no fakeanswers.
