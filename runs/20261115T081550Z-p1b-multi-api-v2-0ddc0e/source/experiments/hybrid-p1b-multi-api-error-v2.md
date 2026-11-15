# P1B-v2 — disjoint multi-API error recovery qualification and learning

Scope: continuation of user-approved P1B actual-error learning on GPU, matched
RAG/readout-only and retention. No new learning rule, ALIF/capacity change or
LoRA in these arms. The previous AddAlarm action bank is CLOSED and remains
uninformative for the5pp advantage gate; do not rerun or tune it.

New hypothesis: six recovery actions may have different useful effects across
actual multi-API schema/state errors, whereas the old bank only needed alarm
argument repair. Test AddReminder, CompleteReminder and DeleteAlarm, with
task/ID/date/time-bound exact state effects. These are owned simulators, not real
notifications or external account actions. Source ToolTalk pinned e05f4ce6;
read relevant reminder/alarm API implementations before transfer, use actual
original API code, don't use semantic_str_compare/embedding scorer.

Literature-first: reuse ToolSandbox task/state/trace distinction and P1A-v2
binding/local-code/persistence; Reflexion/ExpeL require strong verified
experience controls. New continual-SNN intake records CLP-SNN's pretrained
semantic features/prototype assumptions; not a compatible ALIF replacement.
No upstream training benchmark or old history/P0 repetition is necessary.

## Frozen data and native calls

72fresh cases:36TRAIN/12DEV/24TEST,3APIs×4error modes (missing field, unknown
field, malformed value, wrong type). Explicit requested source IDs/values are
public task input. Setup object IDs seeded per case outside policy; private
expected before/after fields not input. Split templates and setup RNG/values
disjoint; source schemas/errors shared and disclosed. No IDs/family/split as
feature labels. Keep raw error+teacher proof and source hashes.

Each invalid call and independently corrected teacher call executes on the
owned source backend. CompleteReminder must mutate only exact target status;
DeleteAlarm removes only specified object; AddReminder creates exactly one
matching task/due_date object and no other changes. No weak question-mark or
self-verified status rubric. Dataset is controlled simulator data, not arbitrary
language/vision tasks or human-life experience.

Qualify12DEV first: six exact existing action macros followed by the same
frozen native0.8B response/execution; terminal actions cannot create mutations.
RAG arm retrieves2 TRAIN-only verified experiences via fixed lexical API/error
similarity. Fixed rule error→show_schema/elsefinish, all action-compatible
constant strategies retained as controls. Same response max128tokens/temp0.
Strongest registered control vs measured-action oracle margin must≥5pp before
training. If fails STOP, no new bank/grid created in this run. Guarded outcome
learning isn't justified just to display updates.

If DEV qualifies, generate36TRAIN+24TEST outcome table with same locked runner.
Before learning check TRAIN marginal achievable improvement≥5pp; don't train
if task/utility headroom is absent. Do not gate or choose winner on finalTEST:
TEST effects remain evaluator-only. The new data/semantic hypothesis was
registered before any v2 native calls, not posthoc changes to v1TEST.

Budget504native calls max (12+60cases×7arms),30minutes qualification+learning
wall; max1repair tool call after initial failure. No retry bad JSON generation.
Source/permission/nonfinite/execution failure stops and preserves raw output.

## If prerequisites pass: fixed local comparison

Reuse existing LocalALIF H128/64D FP32 CUDA0 rule exactly, modesfrozen,
readout_only,learned,shuffled, seeds260113/260137/260171. Public task and actual
tool error in two8tick blocks; six research action slots versioned in checkpoint.
No pretrained hidden features from LoRA, no backward or BPTT in SNN.

20stages×12common fixed-exploration experiences. At stage t choose API×error
family t%12;12samples rotate the3 TRAIN examples in that family, disclose
repetition. Candidate action schedule t*12+j mod6, actual branch effect gives
reward+1/-1 for that action. Shared off-policy behavior, not a claim that SNN
itself discovered the behavior; policy outputs evaluated independently.
12replay updates per stage,64reservoir, same RNG stream/steps among learned arms.
4320local update attempts total, maximum4500 including any diagnostics;
<=20000explicit+implicit neural forward calls, no rerun or reset/refund afterFAIL.

TEST24unique cases/12families, evaluated before learning and after every stage,
no labels/rewards added to memory. Compare same frozen VLM outcome tables with
strongest registered rule/constant-action and native verified-experience RAG;
include full data provenance, all arm/seed curves and probabilities. RAG has
access to all allowed TRAIN experiences on SSD; no artificially undersized
retrieval store to make neural memory win. No TEST experience retrieval.

Keep acquisition>=65%, primary gain>=5pp/lower95%pairedCI>0 vs strongest
registered control; hidden must exceed readout-only and pass shuffled/reset
attribution. Hierarchical seed/task-family bootstrap, no per-token/update
independence. Mean forgetting<=3pp, no family>5pp after20stages; best previously
measured family accuracy minus final, frozen/noacquisition can't pass. Tiny
family counts are exploratory, not a broad retention certification. No boundary
indicator supplied to model; stage markers are external evaluation metadata.

Same-GPU checkpoint+next inference exact, persist RNG/replay/mode/input/action
version; hidden-reset on clones only, no live or old weights imported. Source
local equations unchanged; previous CPU/CUDA hard-threshold failure staysFAIL,
this CUDA-only study cannot certify cross-device trajectories. Artifact250MB,
worker host2GB/physical dedicatedGPU512MiB reserve, no paid APIs or quantization.
Publish actual errors/model answers/loss/deltas/guard/retention events to8775.

Any quality-gate failure is an honest negative conclusion; do not scale model,
rewrite thresholds, drop a strong control, tune epochs/LR or promote intoHermes.
