# Scope of older-round verification

Every branch/fullSHA and source/run path is in BASELINE_INVENTORY.json. The audit
does not claim to have retrained every older campaign. Source/report and saved-row
arithmetic are distinct from runtime reproduction. Current R6B/D and R6C evidence
must not be transferred backward to different samplers/tasks/heads.

| Campaign | What exists / what improved | Evidence level in this audit | Limit / decision |
|---|---|---|---|
| DeepSeek R1 | Head conditioning and frozen temporal features; normalization alone not enough | Source/report inspection | No general hidden-learning result |
| Grok R1 | Slow trace makes delayed state readable before hidden learning | Previous reviewed correction noted; source/report | Dev-scored confirmation corrected71005; hidden gain gate still fails |
| Antigravity/DeepSeek MTB |240wide temporal basis;297screen trials | Source/report + previous Codex checkpoint counterexample | PIVOT; scalar score mislabeled proposal30, no hidden advantage shown |
| Gemini DC-ALIF R1 | Learned-hidden improvement on strict_binding/XOR under protocol deviation | Saved row/source checks; previous Codex tests, not full retraining here | Order/pair-binding regress; P2forgetting fails |
| DeepSeek R2 | Fixed-hidden same-head spike sum .971191vsdecay.506348 |32trial-row aggregate recomputed | Useful readout finding on sim_xor, not useful learned hidden |
| Grok R2 | DC-ALIF transfer/lesions/equation audit | Source/report checked | Transfer screen STOP; composite gate causality limited |
| DeepSeek R3 | Binned trace helps order1.0, temporal XOR only.673584, strict.500732 |200trial-row aggregate recomputed | Confirmation fails broad-win/non-regression gates |
| Grok R3 | Decodability vs trainable online head; shifted/noise tests | Source/report checked | STOP; do not call ridge a trainable capacity ceiling |
| DeepSeek R4 | Query-conditioned bilinear diagnostic, later corrected pre-query stored-value interpretation | Source/report checked | First target was queried label withoutquery; correction matters; complementary values only |
| Grok R4 | Floor/order/geometry diagnostics | Source/report checked | No cross-task candidate promoted |
| DeepSeek R5 | Independent two-key generator; pre-query vA/vB probes .996745/.997396 |3stored-probe rows recomputed; R5B formal head reproduced through exact Grok copy | Frozen memory, not hidden acquisition; caseA formal gate fails |
| Grok R5 | Wrong query tick corrected; variance-safe head helps order but regresses query | Source/report checked | Corrected replay is not retroactive originalGO |
| DeepSeek R6A | Data ladder with scaler drift/raw-stress bug/hard-coded gates | Relevant source defects checked; not retrained | Do not use invalid stress as model failure; R6D supersedes limited questions |
| Grok R6B | R5B exact reproduction and failedNoClip | Runtime reproduced + independent prediction scorer | STOP confirmed |
| DeepSeek R6D | Corrected data/scaler study, positive readout effect | Runtime reproduced + independent prediction scorer/source/data lineage | G6recorded fail, null aggregation and changed-noise pair limitation |
| Gemini R6C | Strong context-gated readout/Oja query gate | Runtime historical replay +344pair arithmetic checks | CKCERaccuracy matches, not all baseline/margin bytes; post-hoc/protocol attribution defects |

The larger historical main sequence is indexed in the supplied handoff and main
docs/research08–44. Its narrow P1-score gains and failed P2/word/vision studies are
retained as historical evidence, not relabeled independently reproduced in this
audit. See CLAIM_LEDGER.json for23explicit claim verdicts.
