# Hermes false-empty read contract repair — before candidate execution,14/11/2026

P03ce157 actual dispatcher/publichook has14calls, locked nonempty file and true
empty file yield identical success/0byte envelopes. Do not deploy a metadata
classifier or train these reports as truth. Preserve original source/raw data.
Root source: compound read parser defaults unparseable byte/line counts to0;
shell pipeline exit statuses can mask earlier read failure. Current target
tools/file_operations.py SHA256d33e0bc43516f9c5da4dd1ab75bd57b03837034c5c4accb91df1d78b92131b51.

Engineering change, not a new learning mechanism or confirmation experiment:
fail closed on unavailable/negative byte size or line count, failed fallback
sample read, empty sample for nonempty source, and inconsistent zero-size/
nonempty content. Do not change schemas, allowed tools, source scope, UTF16
heuristic, model/precision, ALIF, feedback rules or longrun sources. Generic
source-error message must not claim a particular permission/root cause.

Use exact-source patch with byte-for-byte backup/compare-and-swap installation;
unknown live source means stop rather than overwrite another worker's edits.
First apply to an isolated module overlay in the existing private Hermes test
environment. No entire repository duplication or new background service.

Red/green regressions: invalid/negative size, invalid/negative line count,
readable real empty, nonempty source/empty sample, failed fallback sample and
valid readable/partial/BOM/Unicode/file-special guards. Run relevant existing
Hermes reader tests; separate environment/tooling failures from assertion failures.
Reexecute the same14owned file cases after repair, at most20actual dispatcher
calls and≤3minutes; no VLM/SNN calls or new training. Labels of fixture causes
are never neural inputs. New run/source snapshots, never replace3ce157 data.

Acceptance: locked-source output must be an error and public hook statuserror;
true empty still a successful0byte read. Source unchanged after closing handle;
all other fixed cases retain valid read/error/missing/partial behavior. Keep
UTF16/binary ambiguity explicit rather than broadening this patch. Hash-only
observer/privacy behavior and ordinary reader budgets unchanged.

Only install exact reviewed source after red/green+actual fixture checks pass.
No automatic gateway/model restart during8h; source installation alone is not
proof current already-imported Hermes workers use it. Identify the affected
Hermes process reload boundary before activating, and preserve user session/
nativeVLM/ALIF state. Active8hPID/source is independently checked before/after.
Report software-failure repair, not neural self-awareness or retention success.

Maintenance amendment before refactored candidate execution: v1inline repair
passed11target regressions and14actualcalls, but broad noncanonical reader
invocation has10failures requiring clean per-file baseline comparison. Preserve
all outputs, including an initial invalid red-harness sentinel attempt. Follow
Hermes AGENTS: canonical scripts/run_tests.sh, and extract compound-response
interpretation to a small sibling (reduce existing facade instead of growing it).
This refactor preserves the same intended fail-closed behavior. Only3additional
actual calls (true empty, exclusively locked, partial read) are allowed for the
refactor: cumulative17≤20, not a reset of the original engineering call budget.
Run baseline/candidate through the same canonical runner, no silent skipping or
fixing unrelated OS tests to obtain green. Record existing failures separately.
