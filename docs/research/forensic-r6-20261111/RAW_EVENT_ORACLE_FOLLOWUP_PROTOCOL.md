# Data-only follow-up — actual Gemini event/label contracts

2026-11-11, after audit commit24453bd. No training, neural forward, fresh seed,
new mechanism, performance selection or historical artifact overwrite.

Pin Gemini8f24a5abde0598201e558eee559564ccde889145. Regenerate only already-scored
121011/121017/121023/121029/121031/121037/121041/121043 raw condition data at the
same counts/offsets. Verify all train1024, held512, paired512 rows, including
shift, reversal, distractor, noise and both auxiliary generators.

Independent oracle: parse key/value marker channels0–3 in chronological order;
two base writes, or three writes for overwrite; latest value wins. Query comes
from channels4–5 and must follow the writes. Background6–7 cannot change target.
Never use candidate y/vA/vB/returned metadata to infer answers. Threshold.5 is
valid for exact marker tensors and ordinary.08noise fixtures. Treat ambiguous
raw noisy markers as unresolved observation, not a wrong training label.

For the known.16Gaussian-noise transformation, independently reconstruct the
background with the registered noise seed and shape, subtract it only in the
evaluator, and parse the remaining event pulses. This verifies the generated
latent event/label relation; it does not assert that arbitrary noisy observations
are perfectly decodable. The model never receives this de-noised oracle tensor.

Separate checks: row-label correctness, query ticks, chronology/overwrite, paired
facts, and same non-query noise. Do not turn valid labels into same-noise paired
evidence. Count mismatches instead of repairing inputs/labels. Inspect the meaning
of first-key baselines separately from chronological first-write diagnostics.

Use one CPU numerical thread, sampled process tree≤2GB and hard per-process cap
where available,120s total, new immutable run/source snapshot. Mutation tests
must catch wrong latest-write selection, treating background as a semantic write,
and changing target/query. Preserve the already delivered ZIP/commit unchanged;
publish an addendum if evidence resolves or adds a limitation.
