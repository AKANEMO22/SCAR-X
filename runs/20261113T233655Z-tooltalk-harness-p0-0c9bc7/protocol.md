# ToolTalk harness P0 — registered14/11/2026 before isolated method execution

Intake08df3e verified pinned e05f4ce6132c80ed33392b81535b077d56ab28fd and78
conversations. Still no external model inference/training. Source inspection
found two issues that must be checked before using publisher success as evidence:
to_openai_doc places required outside parameters; success may ignore failed
extra calls if all gold calls eventually match. Also apis/utils imports sent2vec;
do not install it, load extra embedding model or silently alter matched RAM.

Allow ONLY AST extraction/compilation of the audited pure methods
API.to_openai_doc and ToolExecutor.evaluate_predictions, with decorators removed
for direct invocation, exact remaining body/source SHA retained. Restricted
globals include basiccontainers, deque/json/logger and a local comparison stub.
No API/module import, database mutation, embedding load, networking or upstream
generation/evaluation entrypoint. The stub compares owned fixture JSON only, not
LLM outputs or finalTEST answers. No eval/exec of model-created text.

Before execution lock fixtures: one correct call; that call plus24identical failed
reads; plus one successful unintended mutation; plus24failed mutation attempts;
no correct call. Keep originalmetrics unchanged, including any surprising PASS.
SCAR-X must report nonprogress/failed-call cost separately, not relabel upstream
success or count success-after-loop as evidence it learned to avoid loops.

Use installed jsonschema to check the original parameters schema on empty
arguments with a required field, then a research-only normalization that moves
required into parameters and rejects conflicting/unknown required declarations.
Preserve original upstreamsnapshot. No extra deps, edits to Hermes/serving/core,
candidate weights or gates. This P0 does not approve a learning run.

Save source methods, original/repaired schemas, fixture predictions/metrics,
counts/roles/import metadata and independent regression checks. Retain upstream
LICENSE/NOTICE in any committed extracted-code bundle. Stop on hash mismatch,
unexpected AST function/calls, missing validation package or uncontained effect.
Bound≤5minutes; publish real technical measurements to localhost as orchestrator.
