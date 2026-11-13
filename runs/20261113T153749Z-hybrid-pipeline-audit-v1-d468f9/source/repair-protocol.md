# Pipeline repair acceptance — 2026-11-13

Engineering fixes following audit `20261113T152708Z-hybrid-pipeline-audit-v1-af1dd5`.
Keep that failed endpoint answer, original outputs, WER and source/protocol.
Do not reinterpret fixtures as scientific model or hidden-synapse improvements.

Approved task scope is pipeline inspection/normal reversible bug fixes, not new
architecture, training or LoRA (the subsequent LoRA request was explicitly withdrawn).
Preserve ALIF, weights, original precision, 16K and whole-system4.5GB.

Repairs: apply explicit first/last temporal constraint before learned ranking;
pair a requested historical screen observation with its own retained frame;
let non-resource ASR errors record missing audio and continue vision. RAM breaches
remain fatal; an expired past frame must not silently become a current frame.
Mark degraded coverage separately from source capture completion.

Re-run the SAME fixed engineering scenarios/labels and limits from the audit
in a new isolated run using its unchanged runner. This is development acceptance
on known failures, not held-out confirmation. Expect blue at start, yellow at end,
normal tool/chat/image functionality, exact persistence/rollback and preserved
guard. Structural tests must reject a relevance score that prefers the wrong
time and still propagate MemoryError. No release-quality or 8h claim follows.

Optional diagnostic after the repeated run: decode the independently labelled
21s Vietnamese clip as one chunk with the same FP32 Whisper; compare edit counts
to the existing three-chunk transcript. Do not change ASR based on the result
within this acceptance run or call one sample a corpus benchmark.
