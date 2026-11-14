# Independent read-query oracle P0 —14/11/2026 before execution

Parent descriptive qualification b77ce6:229exact,6order-only,6generatedID,
20superset,2keyschema,3unresolvedpayload across266calls. No training labels.

Build independent SQLite in-memory selection oracle for five documented read
APIs: FindAlarms, GetReminders, QueryCalendar, SearchInbox and SearchMessages.
Input is the actual simulator database state immediately BEFORE the call plus
explicit args/session/clock. No goldresponses, model text or futureturns enter
the oracle. Read Python officialAPI docs/source to resolve documented predicates;
implement selection/ordering independently in SQL, never call their checker.

Calendar interval intersection is inclusive; alarms inclusiveHHMMSS ranges;
reminders allownedrecords; email/message literal-keyword any/all search with
sender/time filters, currentclock excludes future, newest5. Preserve multiset/
duplicate and stable insertion order for equal timestamps. ASCIIcase matching
scope for the English sourcecorpus; no claim generic Unicode search equivalence.
Unknown tools/invalid auth/date/type must failclosed outside validquery scope.

Replay original78conversations and266annotatedcalls through the same qualified
private sandbox to reconstruct prefixes, never toolsource/live modifications.
For each validreadquery save statehash, actualstate JSON, oracle rows, actual
response and originalannotation. Compare actual and annotation separately under
order-insensitive multiset records where documented; do not erase changed fields.
Mutations in prefix still use original simulator, so this audit verifies query
selection on its current state, NOT independently every mutation/user intent.

Counterfactual tests before corpusrun: timeboundary and future exclusion; any/all
keyword and sender; extra/wrongrecord differs; duplicatepreservation; authdenial;
calendarcontains/intersects; equal-date stabletop5. No similarity/BERT/LLMjudge.
All mismatches preserved. Predicate match supports query-contract validity only;
cannot make annotation or final natural-language response a factual traininglabel.

Budget≤5minutes/10MB, no modelinference/learning/deps/download. Publish actual
oracle outcomes at8775 as orchestrator. No touching8hfrozenfiles/main/live/R13.
