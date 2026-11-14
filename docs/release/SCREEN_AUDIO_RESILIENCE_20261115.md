# Screen audio failures no longer automatically kill visual ingestion

Confirmed on the previous RC7 source: a real FFmpeg decode failure propagates
out of `ScreenMemory._ingest`, marks the share failed and terminates its worker
before processing a valid frame. Optional audio import/PyAudio initialization
also happened outside the worker's audio error-reporting block.

The repair keeps non-resource audio errors local to that channel, records failed
intervals and continues visual ingestion. A later successful chunk can resume
ASR; it cannot bridge a failed/preempted predecessor with an overlapping label.
After three consecutive failures ASR pauses for this share to bound retry work.
Stop and start a new share to retry; video remains usable. Foreground work gets
priority and records an explicit gap without adding an error penalty.

Resource `MemoryError`, a budget violation discovered during error handling,
and source paths outside the owned archive remain fatal. No silent CPU fallback,
microphone substitution, failed transcript or automatic learning label is added.
Historical errors stay recorded even after ASR resumes. State keeps at most 128
audio failures and 128 skipped intervals; the model receives their coverage
limits rather than an implication of continuous audio understanding.

## Actual fault reproduction

`scripts/accept_hybrid_screen_audio.py` replays the old and fixed ingestion code
on separate disposable owned journals. FFmpeg really rejects a malformed WAV;
two subsequent static images are sent to the existing native GPU worker.
The before source is pinned to commit `1e8f4534`, not installed into live Hermes.

| Measurement | Previous source | Fixed source |
|---|---|---|
| Whole ingestion status | failed | stopped normally |
| Frames analyzed by actual native VLM | 0 | 2 |
| Actual native answers | none | `red`, `blue` |
| Visual frontier in fixture time | 0 | 10 s |
| ASR frontier / ASR calls | 0 / 0 | 0 / 0 |
| Failed audio interval | fatal before coverage recording | [0, 1] s, degraded |

The source is an **owned journal producer**, not WGC or speaker-loopback capture.
Fixture times do not measure real continuous viewing. The two answers test the
pipeline continues, not general video quality or learned skills. Physical GPU
free-memory point samples were at least 2,785 MiB; these are not a continuous
peak measurement. Owned producers exited; live SNN update count stayed five.
Raw before/after outputs and source remain in the private acceptance directory;
[public receipt](evidence/SCREEN_AUDIO_RESILIENCE_20261115.json) includes hashes.

**284 Python checks passed**, including CPU/native/CUDA parity, persistence,
actual FFmpeg failure, recovery, three-failure pause, preemption, fatal resource
errors, source confinement, unavailable loopback and coverage delivered to the
model. One CuPy CUDA_PATH warning remains; the CUDA cases were not skipped.
No renderer/plugin change is needed: existing degraded-state rendering and
normal source frontiers apply. Prior Node/Desktop/HTTP receipts remain historical.

This improves runtime fault isolation. It does not teach SNN the cause of tool
loops, validate transcripts or satisfy lifelong/production gates. Live loopback
acceptance and general model quality remain open; the cancelled eight-hour test
is not restarted.
