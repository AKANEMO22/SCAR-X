# P1B — actual error recovery, matched controls and retention

Approved15/11/2026: user requests P1B learning from actual errors vs RAG and
readout-only, retention, ALIF/local learning on GPU, unchanged criteria and real
localhost trace. Additional permission allows VLM LoRA, isolated as a separate
arm/protocol. P1B main uses the same frozen native0.8B; never attribute LoRA gains
to hidden synapses. No main/R13/Hermes serving edits or weights promotion.

## Literature-first / reuse

Reuse P1A-v2 input/action/receipt/persistence and local rule; don't repeat its
frozen qualification, old132 reference cells, failed history budget, or old
24-case bank with oracle margin4.17pp. Reflexion paper2303.11366 and ExpeL
paper2308.10144 + author reflect/update_memory code were read: experience-memory
controls are mandatory, reflection isn't a verified label. ToolSandbox source
blobaf9faacc and P0 exact effect checker supply state/outcome design, not soft
similarity truth. Original ToolTalk API code at e05f4ce6 supplies actual error
execution; retain MIT/LICENSE/NOTICE and prior schema patches. No paid API.

Unresolved hypothesis: do causal task/error observations with verified recovery
outcomes permit hidden local updates to improve held-out action/task success
beyond strong experience retrieval/readout-only and keep it after20stages?

## A. Data/action usefulness qualification before any SNN update

Build a fresh finite bank of owned alarm tasks from tool execution. Preserve
actual invalid-argument, unknown-argument, invalid-time and authorization error
JSON, immutable before/after states, source pins and valid corrections executed
with an independent exact alarm transition checker. Public user-requested times
are observable; expected effects, family IDs and future outcomes are checker-only.
No private user data or hardcoded successful VLM replies. Deterministic fixture
corrections are disclosed as teacher/executable contracts, not model answers.

Primary action arms use six research slots show_schema/inspect_status/retry/
clarify/defer/finish. Same frozen VLM/schema/response budget handles any argument
repair after macro; all arms have identical access. Raw errors alone do not
authorize factual learning. Include auth insufficiency and unsupported cases;
ask must address missing field/authorization, not just contain a question mark.

Split by public wording and independently selected time values before inference:
32TRAIN/8DEV/16TEST for bounded source/error pilot. TEST is excluded from retrieval
and training labels; shared task template across states is reported explicitly.
This narrow initial corpus is not a general language/video/tool benchmark.

Run pretraining data audit on allowed action coverage, causal input equivalences,
duplicate/template leakage, and action outcome tables. Compare strongest fixed
capability/error rule, BM25+verified episodic experience, vector-kNN experience,
and frozen selector. No rule may inspect private case ID/expected outcome table.
If strongest rule is at ceiling or oracle minus strongest control<5pp, STOP
advantage training on that bank; do not drop the baseline or invent harder cases
after viewing results in the same run. This qualification is useful even if
it stops SNN training. Independent LoRA pilot may still use valid TRAIN corrections.

## B. Only if A qualifies: fixed local learner comparison

Seeds260113/260137/260171, ALIF H128/64D FP32 on CUDA0; unchanged LocalALIF rule,
new versioned composition to bind six action semantics rather than importing
live weights. Arms frozen/readout-only/hidden/shuffled; RAG controls no neural
update. Same verified experiences, replay64, bounded evidence and task scopes.
20stages×12new experiences per stage with12replays max per learned arm/stage.
Task boundaries are evaluation scheduler metadata, not input to model.
No EWC/metaplasticity/new credit rule or history variant silently added.

Register exact task composition after qualification before first update.
Max20,000 SNN forwards,4,500 local-update attempts and1,200 native input calls,
60minutes wall including dataqualification; persistent attempts precharged, no
refund/reseed or automatic confirmation on failure. GPU physical reserve512MiB,
worker RSS/commit<=2GB, artifacts<=250MB. Any nonfinite/source/permission/device/
persistence failure stops that run and preserves receipts.

Primary gain>=5pp vs strongest matched control and lower95%paired task/seed
bootstrap CI>0; hidden must exceed readout-only with hidden-reset/shuffle causal
checks. Acquisition>=65% and20stage forgetting mean<=3pp/no family>5pp. Freeze
or unlearned/chance behavior cannot pass retention. Report accuracy curves and
full per-task heldout predictions; cluster by unique task/family, not updates.
Cold restart/retrieval-off and delayed old-task returns, full state/RNG/replay/
rollback exact. Same-device continuation, not a relaxation of failed historical
CPU/CUDA hard-spike trajectory parity. No inference into live/user checkpoints.

If A stops, B remains unexecuted, explicit result; do not claim user request
has been scientifically completed. Publish actual sources/calls/loss/synapse
updates/guards/rollbacks with origin in localhost; no mock thinking or scores.
