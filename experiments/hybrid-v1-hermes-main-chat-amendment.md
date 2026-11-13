# User-directed main-chat/screen amendment —2026-11-13

User clarified: SCAR-X is selected as a normal Hermes model, using normal chat,
images, tools and video attachments. The SCAR-X tab is debugging only. User also
requires opt-in viewing of a video already playing in a selected screen window,
plus file attachments. Never capture an arbitrary window or microphone by default.

Implementation: one existing OpenAI-compatible gateway, preserve system/user/
assistant/tool history and native tool_calls, normal Hermes provider registration.
Hermes minimum-context exception is scoped to exact local SCAR-X route, not a
false64K declaration. Actual working buffer16,384tokens; weights remain native
BF16/F32 and ALIF remains unchanged. Runtime resource qualification is repeated.
Soft background threshold3.85GB; hard target remains4,000,000,000bytes including
Hermes. This changes engineering I/O configuration before P1/P2 measurement;
all scientific quality/hidden/retention thresholds remain unchanged.

Final UI clarification: no permanent Video/Screen/RAM toolbar. Video uses Hermes'
existing attachment menu and scoped composer middleware. Screen viewing is a
normal model tool (`scarx_video` or Hermes computer-use), gated on a recent user
request and an identified window; system-audio loopback requires a request to
listen to video/audio, never a microphone. Capture timestamps are
relative to sharing start, not a guessed video playback position. Raw observations
are local, with an indicator and bounded queues/SSD retention. Screens/images are
perception evidence, never automatic verified labels. Public fixture checks do
not count as P1 or lifelong success. Original video engineering runfd5931 exposed
real errors (Vi ASR hallucination and poor evidence selection); retain that receipt.

Resource repair after a measured4.039GB violation: frozen FP32 ASR is loaded only
for an audio chunk and yields its memory before VLM generation. No weight paging,
quantization or changed learning thresholds. Capture/analysis frontiers show lag.

Final engineeringrepair after another measured4.045GB audio overlap: run frozen
Whisper via bounded CLI, and if projected audio headroom is insufficient, unload
the whole frozen VLM component and restore it before vision/generation. Never
page individual layers/weights, no dtype/model switch; ALIF/evidence persist.
Reload latency, I/O and lost prefixcache are reported. 21s engineering receipt
4003aa completed at3.629GB; this does not pass P1/P2/P3 or the TTFTquality target.
