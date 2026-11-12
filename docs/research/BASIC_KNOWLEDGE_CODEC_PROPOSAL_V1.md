# Basic-knowledge codec / migration proposal V1

**Approval update:** user approved **isolated P0 only**. See
[P0 handoff](OUTPUT_CODEC_P0_HANDOFF_V1.md) and
`experiments/output-codec-p0-v1.md`. No dataset training, P1/P2 or serving/live
migration is authorized. The pending notice below records the proposal's original
pre-approval state; the original design and acceptance thresholds are retained.

**PENDING USER APPROVAL. No tokenizer, encoder, renderer, vocabulary, checkpoint
or serving model was changed.** This proposal addresses observed I/O blockers,
not credit assignment, semantic understanding or retention by itself.

## Evidence and choice

Frozen320source-verified rows cannot be used unchanged by the public24token G1
checkpoint. Complete legacy token coverage+predeclared unknown policy needs55
tokens, within256;128decimal rows fail exact surface roundtrip. Expanding to55
does not require extra ALIF neurons. Keep current48–128capacity policy and2GBlimit.

Proposed **output-only codecV2**, with two separate P0 controls:

- **Legacy-expanded reference:** keep v1tokenization/rendering, initialize a
  research-only complete55token vocabulary. This isolates coverage from I/O
  formatting; it is not a candidate that can satisfy exact decimal output yet.
- **Numeric-atom outputV2:** keep a finite decimal literal as one output token,
  separate terminal punctuation/EOS, and otherwise retain existing word rules.
  Numeric vocabulary is derived from verified **train target strings only**;
  fixed unknown-policy words come from the already declared policy. An inventory
  calculation also gives55tokens. No final answer/prompt drives vocabulary design.

The candidate must encode arbitrary supported decimal strings lexically, not
use a fact-ID→answer table or a task-specific answer formatter. It still generates
every token from the neural readout. Current source examples `0.25.` and `20.0.`
would be represented as `["0.25", "."]` and `["20.0", "."]`. Exact raw text and
typed semantic metrics remain separate; do not rescore old outputs as improvement.

## Preserve input encoding

`UnifiedCore.context()` currently calls legacy tokenization. Globally replacing
that function would silently alter the encoder. **Not allowed.** Version only
answer encoding/rendering; preserve prompt normalization/hash/trigrams/lexical
encoding byte-for-byte, including prompts with numeric text. No pretrained
embedding or new encoder is introduced. Autoregressive previous-token one-hot
channels are explicitly versioned because output IDs/layout change.

## Research-only migration plan

1. Only public release/discarded research copies; never private/live SQLite or
   `local/generation1`. Save original payload/hash before any operation.
2. Preserve EOS/BOS and old word IDs. Append approved new tokens deterministically.
   Shift context/image input columns and digit output rows consistently; copy
   input/recurrent/output/stable weights, topology, neuron state and learning state.
3. New input columns/readout rows are initialized under a declared fixed policy.
   **New output tokens stay masked during legacy parity checks.** Merely setting
   bias−6 is not exact preservation: their softmax mass changes old probabilities.
4. Store explicit output-codec version, vocabulary/mask/layout and state policy
   in a new research checkpoint wrapper. Missing codec metadata means legacyV1;
   do not reinterpret old IDs/numeric fragments. Unknown/foreign formats fail closed.
5. Rollback restores the complete old model/controller/RNG/replay/cursor payload,
   not just hidden weights. No silent in-place overwrite of the source checkpoint.
6. Any deployment/migration later needs independent approval. Hermes model ID
   remains a single SCAR-X model; this research does not add another serving model.

## P0 verification before any quality training

- Token/EOS/Unicode/range limits, decimal signs/precision/punctuation and old-word
  cases; malformed numeric-looking text cannot be silently changed into a fact.
- Roundtrip all supported train/policy strings, then schema/representability
  recheck final data without model quality scoring. Preserve frozen source pool.
- Legacy old-token-conditioned probabilities and neural trajectories exact with
  new tokens masked; digit outputs unchanged under remapped indices.
- Full state checkpoint/partial-segment cold restart/exact continuation and
  rollback, including masks, normalization, optimizer/importance if present,
  RNG and replay. Reject corrupt shapes/types/duplicate members before allocation.
- Count snapshot/rollback/serialization memory under2GB; CPU reference; no iGPU,
  C++optimization, scale, API call, source lookup or learned model needed forP0.

This P0 is technical compatibility, not an intelligence test. No P1/P2 or new
teacher calls are included in the proposed approval scope.

## Subsequent research, separate approval

Coverage/format changes alter token count and output classes. Any later quality
comparison must report equal-experience **and** equal-update/compute conditions,
not credit a shorter target to hidden learning. Keep frozen/readout-only and
correct/zero/shuffled hidden controls, common refit and actual free generation.

Current train has no unknown targets. Before a neural unknown-detection claim,
register a source-verified negative-training curriculum using a separately
approved AGY budget/version, or evaluate controller abstention/calibration as
controller behavior. Do not train on unknown development/final examples.

Acceptance remains factual≥80%, heldout paraphrase≥75%with required contrasts,
unknown rejection≥95%; acquisition/final≥65%, mean forgetting≤3pp/worst≤5pp
under the registered seed/stream. Do not open retention pilot, large corpus or
production import from a codecP0pass. If hidden learning remains weak, report
the codec success separately and keep G1 unchanged.
