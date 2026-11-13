# Purpose-overview routing repair — registered14/11 during the v1 soak

The v1 actual app-z-fold job read local data with correct cwd/no web, but spent
87s in varying search_files/terminal attempts and returned an iteration-limit
notice. Existing exact-repeat guards do not recognize every query variation.
Keep that FAIL as-is. Do not interrupt or rewrite the frozen30minute v1 run.

After v1 CPU/restart checks finish, amend only the runtime workspace controller:
the initial bounded read also samples common manifest files one directory level
below root and representative root source files; no binary execution, writes,
secret-dotfile reads or unbounded recursion. Preserve the14000character source
budget, label truncation, preserve normal Hermes tool/approval execution.

For explicit project-purpose/overview questions (e.g. what the folder does),
after4tool results request a final answer with tool_choice=none while retaining
all schemas/history. This is an observable runtime budget rule, not learned
ability. Other coding/edit/debug workflows retain the original general budget.
Never replace model answers or synthesize project conclusions as if inferred.

Technical regressions: correct final runtimecwd; bounded actual reads, .env and
outside symlinks excluded;4-result boundary; unrelated/explicit tool directives
and coding-change requests unchanged. Native VLM/file-inspection E2E with an
isolated read-only test home on the actual project and on fixture. Lock3prompts
before measuring,90s/job, same model/precision/ALIF; preserve failures and private
transcripts with hashes. Must read local source, avoid web and return a factual
overview rather than a limit notice. No neural training/LoRA or gate changes.

The v1 endurance receipt stays tied to its source snapshot. New routing gets
separate regression/smoke receipts; do not call v1 a30min run of post-repair source.

DEV2 showed native output may remain empty/rawtoolXML under tool_choice=none,
and Hermes' exact empty-response nudge reopened tool work. Preserve DEV2 results,
including109s wall overrun (initial startup+retry; harness lacked outer90s fence).
Treat those known nudges as scaffolding for the same4-result boundary. At the
boundary return an explicitly labelled runtime controlled-failure notice and
verified read-result paths/format metadata, not a fabricated VLM analysis. Never
count expected words in this notice as model-quality PASS. New runner enforces
the outer90s/job and saves timeouts. This bounds the bug without pretending to
solve the0.8B semantic/reasoning limitation; a capable analysis model is still open.
