# Video reply presentation — actual Hermes acceptance

The completed canonical Hermes turn on the previously uploaded owned clip
returned **“Hình đầu tiên là một hình vuông màu đỏ.”**, with zero tool events.
The runtime separately appended **“Dữ liệu tham chiếu: [E1 · 00:05]”** from the
observation actually supplied. The stored event preserves `native_answer` and
`runtime_reference_footer` separately. This is not a claim that the VLM learned
citation, that its answer was independently verified, or that arbitrary videos
now pass a quality benchmark.

## Change and boundaries

- Put supplied images before observations and the original user question. Keep
  the original user words; add a bounded language instruction. Vietnamese hints
  do not treat every non-ASCII language as Vietnamese.
- For local-media calls, which already have no tool schemas, omit only the
  integrity-pinned, exact unused Hermes steering briefing in a system message.
  The baseline native diagnostic reproduced marker echo with that briefing.
  User content, history and generated output are not stripped or replaced.
  Other calls retain the existing host steering contract.
- Append source metadata only to completed plain-text answers with supplied
  media evidence. JSON/schema responses remain untouched. Streaming puts the
  footer before completion; native usage counts remain provider counts.
  Screen timestamps mean time since sharing, not video-player position.
- Discard client-supplied internal evidence fields before deriving metadata.
  Source references describe supplied observations, not support for every claim
  in the answer. There is no fabricated native citation or learned reward.

## Verification

**263 Python checks, 29 Node checks and three actual isolated HTTP feedback
scenarios passed.** Python includes CPU/native/CUDA parity, checkpoint/rollback,
media ownership/concurrency, exact-note scope, JSON preservation, SSE delivery,
timestamp rounding and questions in other scripts. CuPy emitted one CUDA_PATH
discovery warning; all four CUDA cases actually passed without skipping.

The canonical Hermes acceptance resumed the owned fixture session. It used the
persisted media binding and completed normally; this was not a new Desktop UI
upload test. The preceding RC6 first-draft UI acceptance remains recorded with
its English/no-timestamp failure. Diagnostic candidates, including a candidate
that echoed the host marker, are kept in private logs. No thresholds or old
receipts were rewritten.

Saved neural arrays, RNG and replay matched across the presentation activation,
with five existing local updates. ALIF, weights, precision and learning rules
were not changed; no laboratory weights were imported. The source footer is
runtime presentation, not hidden-synapse improvement.

[Machine-readable audit](evidence/MEDIA_REPLY_PRESENTATION_20261114.json) can be
re-derived by `scripts/audit_hybrid_media_presentation.py` from existing private
receipts/SQLite without additional model calls. Full regression receipts are
under `local/media-presentation-final-release-20261114/`.

This remains a limited RC. General video understanding, speaker-loopback,
lifelong retention and SNN superiority remain unqualified. The cancelled eight-
hour test stays cancelled.
