# Open-vocabulary byte dialogue prototype v1

Registered2026-11-09 before codec/training measurements. Follows masterP1/P2
and event-dialogue-v1 failure; not a third local-target-rule adjustment, bigger
hidden model, language pretraining or live migration. Local-target family stays
NO-GO. Same existing scalar-STDP/ordinary output-local rule only.

## Codec and source review

Fixed256UTF8byte values, token IDsbyte+2;EOS0,BOS1. No learned vocabulary or
development-label-derived output types. Text supports strict UTF8 and excludes
ASCIIcontrol chars except tab/LF/CR and DEL. A deterministic cursor enforces
RFC3629 validity (https://www.rfc-editor.org/rfc/rfc3629.html), including no
overlongs/surrogates/>U+10FFFF. BOS masked, EOS only at character boundary and
after at least1byte. Byte-length budget masks starts that could not finish a
codepoint; no replacement characters/answer templates inserted. Syntax validity
comes from the HOST MASK, not learned language/semantic ability; report separately.

Keep48ALIF,K8,R12;256previous-byte INPUTchannels,128text/64image/6control =454.
BOS signaled by a control channel, no previous-byte one-hot. Outputs258+10digit
slice=268. All byte conditions matchedD/O; historical word experiment D/O differ,
so never attribute cross-experiment gains to learning-rule superiority. Report
actual parameter/memory changes due codec. Hidden topology/population unchanged.

Same chronological InputEvents prototype and three ALIFticks per byte step,
12quiet boundaryticks; no neuron reset between responses/passes. Input history
prefix excludes current target. RGBpatches can feed the same core/byte head,
but natural-photo learning is NOT tested or declared here.

## Data, conditions and locked budget

Use immutable curated Antigravity data directory
data/candidates/antigravity-events-curated-20261109T094142Z (96train responses,
24development). No new teacher generation, target edits, output-vocab fitting,
finaltest or live/private session import. All synthetic/rubric labels; not
independently factual-verified. Compare codecs' representability only before
training, not semantic correctness.

Fresh seeds6101,6113,6121. Conditions byte_frozen,byte_raw,byte_normalized,
byte_normalized_full. Raw/full output-local lr0.08,clip0.025,weightclip3; full
uses current G1scalar advantage/STDP,hidden_lr0.0001,weightclip20. Frozen keeps
all weights unchanged. All heads receive the same teacher-forced training walk
for per-neuron calibration, then frozen statistics. Byteposition normalized/128,
no adaptive-computation claim. Teacher forcing supplies only PREVIOUS targetbyte,
not current/future byte to the core; syntax mask uses prefix only.

4passes through32dialogues, preserve turn order inside dialogue, seeded shuffle
dialogue order each pass. Same number of byte-steps and labeled targets across
conditions; bytes increase steps vs word trial and costs must be shown. Max
answer256UTF8bytes for training; max128newbytes at inference. Evaluation at0/2/4
on cloned state, train-final/dev exact after tokenization plus raw exact, valid
UTF8, visible/nonempty response rate, byteNLL/byteaccuracy, repeated-character
rate, byte/character lengths and all outputs. Target representability must be
100%without using development vocabulary. Valid-but-gibberish output fails.

Stop after both2and4checkpoints show <10ppdev exact improvement over frozen
matchedseed; no longer training, broader data/model or best-checkpoint selection.
No result here changes historical representation/retention gates or proves
normalchat.3seeds exploratory; independent8seed/task/retention checks required
before a release. No candidate deployment into Hermes.

Per-condition180s,whole1200s,CPU1thread, sequential, sampledRAM<=2GB and Windows
process commitcap2GB. Save full core/head/codecformat/ticks/counters/RNG used for
ordering plus explicit training schedule. Mid-trial neuron state persists.
Cold-process inference and continuation must match. P0 must reject cross-format
G1loads/checksum faults; do not silently import byte weights into word G1.

Unit checks: heldoutUnicode/emoji/negative numbers roundtrip; every byte in
their teacher-forced paths allowed by cursor, invalidUTF8 rejected; no half
codepoint at outputbudget; no weight updates during inference; readout/full
ablation and checkpoint continuation. No new C++optimization; run existing
baseline native parity for shared SpikingCore.
