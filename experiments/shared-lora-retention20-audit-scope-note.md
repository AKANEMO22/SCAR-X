# Audit scope note during the unchanged20stage worker

Data-only tokenization inspection found the final frozen-base-RAG prompt view
includes full user metadata/schema inside each of2TRAIN examples, yielding up
to682tokens;24of40 exceed the locked512prompt+target bound. This check made0
generation/learning calls and did not inspect a RAG model answer. The original
worker source, examples, token bound, update count and retention gates are not
changed while it is active. No truncation/increase or silent comparator switch.

The20stage acquisition/interference trajectory is an independently useful
part of the preregistered experiment and continues within its unchanged budget;
it will not be called whole-protocol PASS if its comparator subsequently stops.
The original full receipt remains STOP/FAIL when that happens. No model retry or
new training is authorized by this note, and no whole-protocol verdict is
replaced based on a convenient component.

Audit may use --diagnostic-comparator-stop only if the original exact token-bound
failure occurred after all20stages/480updates were saved. It then recomputes the
registered acquisition/forgetting metrics from all original TEST predictions,
checks TRAIN-only updates and checkpoint replay/RNG, and executes saved final
answers through the source simulator. No new inference/training or RAG result
is manufactured. Comparator/old-general-population/whole-protocol requirements
not proven are kept missing/failed, not treated as passed.

This is a diagnosis of Codex's omitted combined-RAG prompt-budget preflight,
not a new scientific effect or permission to loosen3/5pp/CI/acquisition gates.
Any corrected comparator configuration requires a separately registered fixed
source/view before fresh calls; no posthoc superiority claim from missing data.
