# Live observatory acceptance v1

Product instrumentation acceptance, not a new model-selection/retention trial.
Run once against the serving Generation1 on loopback8765 after saving a checkpoint.
No new seeds, weights, architecture, gate or large training corpus.

1. Run the24 existing public development dialogue prefixes through the actual
   serving core. Serving receives latest user text only; the archived Qwen3-4B
   reference used full prefixes. Show this unequal input scope prominently.
   Targets and Qwen output are never sent into inference and never used as labels.
2. Submit one independently checkable explicit integer lesson `7 + 8` → `15`
   through the existing learning channel. Use the unchanged bounded32-attempt
   controller, guard and idle scheduler. Do not bypass a rejection to make a graph.
3. Observe loss, token update steps, changed-weight counts, attempted/retained
   weight norms and guard results. Wait at most60s; unresolved queue is not success.
4. Website requires no start/model-selection action. GET/poll never starts a job.
   New inference and learning events appear automatically; history survives reload.
5. Run software tests (including existing native parity) and inspect the real UI.

Pass means faithful measured visibility and preserved state, not normal language,
useful hidden representation, solved arithmetic or continual-learning success.
Keep all actual outputs and failures. Qwen is an archived measured reference;
do not launch its server, generate a fake answer or replace SCAR-X with it.
