# Output codec P0 V1 — approved engineering scope

User approved **P0 isolated codec only** on2026-11-11, replying to the explicit
proposal at docs/research/BASIC_KNOWLEDGE_CODEC_PROPOSAL_V1.md. This does not
authorize dataset training, P1/P2, a new learning rule, main/live migration or
Hermes changes. All implementation stays in evaluation/research modules.

## Fixed design

- One copied UnifiedCore/SpikingCore ALIF population, unchanged equations and
  local update rule. No Transformer/LLM/embedding/backward/autograd replacement.
- Legacy output codecV1 and output-only numeric-atomV2. Preserve prompt context
  encoding byte-for-byte; changing global input tokenization is forbidden.
- Decimal grammar: ASCII signed integers/fixed-point literals, decimal dot;
  terminal punctuation separate. Preserve all other word/punctuation content.
  Encode/render normalization is NFC/NFKC-compatible casefold/whitespace as already
  declared by BASIC_KNOWLEDGE_ACCEPTANCE_V1. Report raw surface differences too;
  no semantic grading or historical score changes.
- Vocabulary from verified train targets + predeclared UNKNOWN phrases + public
  legacy vocabulary. Expected complete inventory55, cap256. No final prompts or
  quality results select vocabulary/format. CodecV2 numeric atoms are tokens, not
  a fact-ID answer router/template/table.
- Migration preserves old word IDs and hidden topology/state; remaps context/
  image input columns and digit output rows consistently. New outputs masked
  during old-probability parity. Missing codec metadata loads as legacyV1;
  foreign/research formats fail closed in serving loaders.
- Snapshot includes weights/stable weights, neuron/eligibility state, vocabulary/
  codec/mask, capacity RNG/clocks, bounded replay fixture/controller cursor/RNG,
  and a partially taught fixture. No actual automatic replay/consolidation loop.
- Only exact UnifiedCore+SpikingCore supported. PlasticCore/importance/optimizer
  wrappers must be rejected, not silently stripped. Their migration is outsideP0.

## Required engineering checks

1. Roundtrip all320target strings without neural quality scoring, and standalone
   Unicode/decimal/sign/EOS/bounds/malformed-input cases. Same55target coverage
   is checked for legacy-expanded control and numeric-atom candidate.
2. Old word probabilities and neuron trajectories exact with new tokens masked;
   digit readout/trajectories unchanged under row remap. Compare to the copied
   public release, not private/live model. If float arithmetic prevents exact
   parity, report it and diagnose; never relax the gate after viewing a failure.
3. Full mid-segment save/load and fresh-process continuation exact, including
   active mask, eligibility, replay/controller/RNG draws and bounded tiny fixture
   updates. These are state tests, not training a dataset or measuring skills.
4. Atomic save and rollback restores whole snapshot after injected rejection.
   Corrupt checksum/type/shape/duplicate ZIP or oversized declared NPY arrays must
   be rejected before large allocation. No pickle.
5. NumPy/native existing regression as appropriate; no source changes in neural/
   app/learning/production release descriptors. Record source/hash/commands/logs.
6. CPU one numerical thread, RAM/commit+process-tree≤2GB for migration/load/save/
   fixtures/rollback; per-process cap when available. No iGPU or performance
   optimization. Check peak buffers, not only stored model arrays.

P0PASS means codec/state compatibility under the stated fixtures. It does not
mean acquisition, unknown calibration, generalization, retention or production
readiness. Any P0 failure stays visible in a unique immutable run. No model
selection, final-test inference, AGY/Qwen call or benchmark training in this phase.
