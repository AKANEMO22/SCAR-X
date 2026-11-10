# Decoder state retention / causal input audit

2026-11-09, before measurements. MasterP1 working-memory/credit precondition.
No updates, new learning rule, architecture replacement or semantic pass claim.

Fresh byte model seeds8101,8111,8123 and final byte_raw checkpoints6101/6113/6121.
Generate32new order-distinct prefix pairs seed39107 using the existing last-write
counterexample generator. InputEvents sum summary is identical within each pair.
Process both complete prefixes from identical canonical state on clones, capture
voltage/adaptation/features after input; then force the SAME previous-byte stream
(BOS followed by cyclicASCIIa/b/space/dot) under identical pooled context.
At byte positions0,1,4,8,16,32,64,127 measure exact feature equality, featureL2,
adaptationL2 and decoder probabilityL1. Differences measure influence, not correct
reasoning. Exact collapse means that particular readout can no longer distinguish
the counterfactual prefix in that trajectory; no global impossibility statement.

Also erase only encoding-end state on a diagnostic copy (voltage/adaptation/
spikes/features=0; leave all weights, cursor and common previous bytes) and
report same measures. Never reset/alter private serving core. No tests/labels
used for choosing condition or adaptive budgets. Raw array/output statistics
saved,<=60sCPU1thread/2GB. Frozen activation snapshot is observed only; it is not
called learned long-term memory or evidence of continual retention.
