# Basic knowledge baseline V1 — registered before inference

2026-11-11. Directive V2, evaluation only. No learner, vocabulary, encoder,
renderer or live-state change. This does not reopen STOP35–38 or the P2 pilot.

## Frozen inputs

Use the source pool SHA256
`13bb051467b27caa10bf8d16039d52fa830baf77a5e87bd63e3520d8b96ed062`.
The controlled-query verifier is revised before model scores: definitions,
derived quantities and units are rejected rather than labeled by a literal
oracle. Revalidation leaves original candidate bytes unchanged.

SCAR-X: the public G1 release checkpoint identified by
`artifacts/generation1-release.json`, SHA256
`0c1e8bab2c4ca2ec10578bd4ee0e745366ce1f0b483b4b72b418e09fc5fe10a1`.
It has not been taught this new dataset. Load a fresh in-memory public model,
use `canonical_reply` for independent request-state probes, and prove its
serialized bytes and all weight hashes unchanged after evaluation.
No private checkpoint or SQLite is read. No app server is started.

Run on all **64 development** questions and **32 unknown-development** questions.
Never load held-out confirmation prompts into model inference. Counts and IDs
are derived solely from trusted split metadata. Only `prompt` enters the model;
fact IDs, source, labels, language and split do not. Retrieval, tables, templates,
external models, teacher forcing and training are absent. Preserve generated
text, latency, EOS/stop reason and neural token confidences when available.

## Metrics fixed before outputs

- Exact: Unicode NFC, casefold, whitespace collapse; retain punctuation/content.
- Typed semantics: numeric answers must consist entirely of one independently
  parseable literal (optional enclosing code formatting); compare Decimal with
  the source value. No search for the right number inside an arbitrary sentence.
  Enum answers use the same exact normalization, allowing one final period.
  This is a deliberately limited rubric; longer explanations remain ungraded
  by semantic parsing rather than being accepted by keyword matching.
- Unknown: exact normalized Vietnamese/English abstention phrase from the
  registered policy. Record confidence separately; this is not calibrated
  uncertainty or proof of understanding.
- Report known and unknown counts separately; no pooled score that rewards
  always abstaining. No threshold tuning or winner selection.
- Acceptance thresholds remain A≥80%, B≥75%, C≥95%, D≥65% acquisition/final,
  forgetting mean≤3 pp/worst≤5 pp. A development-only untrained baseline cannot
  satisfy release gates; B/retention/causal contrasts are NOT_MEASURED.

## Independent Qwen reference

Optional bounded comparison on exactly **24 development** prompts: variants
5 and 6 of `config_inputs`, `config_hidden`, `config_neuron`, `config_leak`,
`config_ram_limit`, `g1_hidden`, `g1_feature_decay`, `g1_vocabulary_cap`, and
`api_model_id`, `api_port`, `api_host`, `api_models_route`.
16 known / 8 unknown-domain cases, balanced Vietnamese/English. Select before
reading either model's outputs; do not replace cases for score reasons.

Use existing `Qwen3-4B-Instruct-2507 Q4_K_M` descriptor/revision/SHA, verify the
file and CPU llama-server executable before launch. User prompt only, no source
document/target/system answer hints. CPU 2 threads, context1024, temp0, seed919,
max128 output tokens, no GPU, no external API. 4.5 GB reference budget separate
from the SCAR-X2 GB phase. Startup≤90s, request≤30s, whole reference≤420s;
at most one owned server, stop on timeout/limit and preserve partial results.
Never kill an unrelated server; if one exists, stop before launch.

These are version-specific repository facts. Neither model receives repository
source at inference; Qwen is not assumed to know them from pretraining. Report
raw paired answers, token truncation, latency/RAM and the same limited rubric,
not equivalence, commercial intelligence or a fluency score invented by Codex.
Qwen output is reference only, never a training label/verifier/SCAR-X backend.

## Resources and output

SCAR-X phase≤120s, CPU one numerical thread, hard per-process limit2GB when
available and sampled process-tree RSS/commit. Qwen phase is a separate process
with its own4.5GB cap/monitor; never claim its memory as SCAR-X usage. Save source
manifest, commands, logs, checkpoint hash, rows and decision in new run folders.
Only public run files are mirrored to the observer; no live SQLite writes.
No retries, new data generation, model fitting or final-test scoring in this run.
