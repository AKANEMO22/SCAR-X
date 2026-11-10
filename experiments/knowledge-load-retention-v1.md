# Knowledge-load and retention diagnostic v1 — registered before execution

2026-11-09. User requests increased knowledge load, long retention and Qwen4B
comparison. This diagnostic does not approve a larger SCAR-X model or language
pretraining. Main deployed checkpoint and private session are never replaced.

## Hypotheses and scope

H1: adding verified lessons improves unseen paraphrase accuracy by >=10pp vs
the frozen start. H2: acquired knowledge survives interference and restart.
H3: hidden local updates contribute beyond readout-only. H4: bounded replay
improves acquisition/retention under the same number of local update slots.
No result is sufficient to claim general knowledge equal to a4B model.

Qwen reference is Qwen/Qwen3-4B-Instruct-2507, pinned community GGUF Q4_K_M from
lmstudio-community/Qwen3-4B-Instruct-2507-GGUF revision
4edb920b6f14e3b9284d4502a6485103d72cde05; file
Qwen3-4B-Instruct-2507-Q4_K_M.gguf,2497280448bytes, SHA256
8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc.
Quantized reference, not the original BF16 model. Run once on CPU with4threads,
context1024, max12 generated tokens, temperature0, retrieval/tools off, numeric
answer instruction. Its runtime may exceed2GB; report actual RAM and do not
claim matched-memory comparison. Never load it into Hermes or SCAR-X core.

## Data and split

Author128 distinct integer addition/subtraction associations (0..19), deterministic
generator seed8107, balanced operation types, answer independently recomputed.
Deduplicate addition up to operand permutation. One train wording per expression,
one unrehearsed rewording for diagnostic transfer. Another32 expressions with
disjoint canonical keys are held out from all training. These are closed-world
toy tasks, not estimates of encyclopedic knowledge. Targets never enter input.
Vocabulary expansion uses only the current lesson target. All conditions start
from the same public G1 release and have the same initial output vocabulary.
Old skill evaluation uses authored development prompts, not historic acceptance.

## Conditions, budgets and independence

Fixed SCAR-X population48, no growth or new rule. Order seeds17,29,43 provide
three independent curriculum orders from ONE pretrained initialization; they
are NOT three independent model initializations. Do not generalize beyond that.

- frozen: same teacher-forced processing slots, no weight updates.
- full: existing full local rule,32 slots per lesson, no guard (isolated diagnostic).
- readout: same32 slots, output-local rule only.
- full_replay:24 slots new lesson +8 slots uniformly selected from earlier lessons;
  first lesson has32 new slots. Same update count as full/readout, replay pool128max.
- guarded: actual Generation online learner, <=32 attempts per lesson with its
  existing development guard. Different compute overhead/early stopping must be
  reported; not a matched-compute causal comparison to the other conditions.

Evaluate after8,16,32,64,128 lessons. No checkpoint/model selection on evaluation.
Stop a condition after TWO successive milestones whose current-block held-out
wording gain over frozen is <10pp; always record both failure points. Stop at
180s per order/condition or12minutes total, memory2,000,000,000bytes for SCAR-X.
Partial runs remain partial, never fabricated as all128 learned. Qwen comparison
may run separately within6minutes and4.5GB, then must be stopped.

## Retention measurements

For each presentation, measure the fact on a diagnostic clone immediately after
its32-slot training budget. At each milestone, evaluate all previously presented
facts and all old development skills. Evaluation must not alter training state.
Canonical independent probes clear only diagnostic neuron state; also report an
unreset snapshot-based sequence. Disable retrieval and controller answer text.

At termination, fork the checkpoint into three isolated branches:
1.10,000 quiet neural ticks, weights frozen.
2.8 distractor associations with32 local slots each, no original-label replay.
3.the same8 distractors with24 new/8 original-label replay slots.
Keep memory and update budgets bounded; interference uses512? No: exactly
8*32=256 slots per branch. Record actual ticks, updates, bytes and elapsed time.
Checkpoint reload is followed by clearing diagnostic working state and identical
probes, to distinguish weights from transient state and SSD retrieval.

Acquired = correct on immediate unseen wording AFTER teaching but incorrect at
baseline. Report final retained/acquired count and Wilson95% interval. If acquired
count0, retention is UNDEFINED, never100%. Forgetting is max(0, immediate accuracy
minus current accuracy) per old skill/fact block. Historical gates remain mean<=3pp
and worst<=5pp; these are necessary, not sufficient: preserving a nonlearner fails.
Report seed-order paired effects and bootstrap95% with warning of only3orders and
one initialization. Quiet-tick/restart tests are not days/months retention claims.

## Outputs and decision

Unique run with data, source hashes, all predictions, learning records, resources,
artifacts checkpoint(s), report, summary and Qwen receipt if available. No live
weights replacement without a separately reviewed quality gain. If acquisition
fails twice, decision PIVOT_NO_SCALE; do not fix thresholds or add more data to
hide failure. Trace URL stays http://127.0.0.1:8765/ and shows the user's live
model, NOT the isolated stress-test clones or Qwen reference.
