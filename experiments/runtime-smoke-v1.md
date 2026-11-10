# Runtime smoke v1

Registered before execution. 64-neuron random byte model; 512-byte ASCII prompt and 64-byte output, 10 repeats each on NumPy and native C++. Report prefill plus first-byte p95, end-to-end generation duration and bytes/s. These metrics do not indicate language quality. CPU only; no energy inference.

SQLite corpus: 1000 exact 1000-byte ASCII episodes, thus 1M byte tokens for this prototype's byte tokenizer, NOT 1M common subword tokens. Query every 37th exact identifier. Record ingest and search latency; no semantic retrieval claim.

Default 10-second alternating learn/consolidate/checkpoint loop; configurable duration. Only an actual >=28800-second completed loop counts as 8h endurance. Large-scale language gate remains independent.
