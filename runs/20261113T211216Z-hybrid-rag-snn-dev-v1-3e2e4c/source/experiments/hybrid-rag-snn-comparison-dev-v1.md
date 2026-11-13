# Matched RAG / SNN comparison DEV — 14/11/2026

User requests evidence whether LLM+SNN improves on LLM+RAG, learns from errors,
detects repeat-tool mistakes and how to use remaining memory for production.
This is a small isolated DEV audit of the CURRENT learner, not formal hybridP2,
not a model-size increase, new learning formula, adapter or live-weight change.
Keep every historical scientific gate. No BPTT, external teacher/API or private
chat/project data. GPU numerical mode for all candidate ALIF/VLM computation.

## Hypotheses and claims fixed before results

H1: current SNN relevance learning yields >=5pp end-to-end grounded exact QA over
the strongest matched non-SNN memory control; paired95%lower bound>0.
H2: learned hidden beats readout-only; hidden-reset ablation removes its advantage.
A moving weight or different answer alone is not useful representation proof.
H3: a verified correction improves held-out source selection and persists through
restart, with20stage old-skill forgetting mean<=3pp/no group>5pp; acquisition
reported jointly. No changing thresholds after seeing results.
H4: repeated-tool incident recognition is measurable from concrete transcript
signals; a root-cause hypothesis is not verified merely by a self-critique.

Failure/no advantage means PIVOT or retain strongest non-SNN controller; do not
force an SNN into production just to claim hybrid superiority.3seeds are small
DEV evidence and cannot certify broad advantages; hierarchical pairedbootstrap
over seeds/families is exploratory. Family/per-seed counts and all errors required.

## Event-memory task

Procedural, deterministic, owned synthetic project records with source/revision,
timestamp/status/entity. Cases: latest approved value, value before a cutoff,
correction after retracted record, absent evidence. Multiple distractors and
unverified newer assertions, so not a one-item recall score. New entity names
and wording in evaluation. Do not pass gold labels, family or split IDs as input.
Freeze JSONL hashes before learning/inference,3seeds260113/260137/260171,
20stages,24balanced relevance-pair updates/stage (480updates/learned arm).
Evaluate48held-out queries (12/family),8fixed retention probes at each stage.
One training order per seed shared by readout/hidden. No test feedback/replay.
No tuning or second search if scores fail. Policy input remains CURRENT
`features(query+' [event] '+document, (length,))`; missing metadata in that encoder
is a diagnosed limitation, not secretly fixed before scoring.

All arms get exactly the same candidate records/query and source information:
1. SQLiteFTS5/BM25 top1 (with deterministic tie order).
2. Strong source/status/time rule top1, same metadata available to every arm.
3. Bounded64-example episodic nearest-feedback ranker, same64D hash representation.
4. Current ALIF frozen.
5. ALIF readout-only.
6. ALIF hidden+readout local update, unchanged formula.
7. Hidden-reset diagnostic: restore initial w_in/w_rec after training, keep
   learned readout/statistics. This is an ablation, not a new independent seed.
SNN relevance>0.5 required for non-abstention; threshold fixed here. Knn ranks by
positive cosine vote among5neighbors. No winner selected on these48queries.

Measured compute and resident/committed/dedicated GPU memory per method;
neural arms same update budget. Rules intentionally may be cheaper; report cost
instead of burning dummy compute to look matched. Strong rule can approach a
ceiling on this finite contract; this can reject a claimed advantage, never
establish general memory/intelligence success.

## Actual frozen VLM use and resource bound

Use the same native Qwen3.5-0.8B BF16/F32/GPU, same prompt, max96outputtokens,
temperature0, repeat_penalty1.05; one selected evidence or no-evidence marker,
output JSON{"value":number|null}. Exact gold value plus selected-source validity
required for grounded correctness. Each distinct (query,selected evidence)
gets one actual native call. Identical inputs reused by arms share that raw
response and measured cost; do not count cached outputs as new VLM replicates.
Budget <=1000uniquecalls, <=15minutes, stop on model/resource/integrity errors;
GPUuncapped per user, CPUall-neural remains capped4.5GB. No weights compression.
Trace actual prediction/answer/learning/source metrics automatically at8775,
separate orchestrator measurement events from live neural counters. Inference
does not replace the normal gateway or import research checkpoints into it.

## Error mechanism and limitations

Audit real repeated tool transcript signatures and authoritative guards. A
detector may assert “identical calls/results, no observed progress”; it must not
invent a semantic cause. Distinguish badrouting, source absence, rate limits,
spillover/read denial and weak language model using observations/probes.
Existing learned policy can be updated only on identified corrective feedback
or successful independent verifier. Generated self-critique is not a true label.
Use the event-source correction stream above to test current learning rather
than adding an easier new policy oracle and claiming autonomous understanding.

Exact persistence before/after next update on sameGPU, cold learner retrieval-off,
readout-only/hidden-reset comparisons, all20stage curves and losses/deltas remain
in unique runs. No production routing/policy promotion based only on toy scores.
Reflexion/ExpeL show non-neural episodic experience can also help; include that
control. Self-RAG trains reflection behavior in its own LM; current frozen0.8B
does not inherit that ability. E-prop eligibility is a credit-assignment method,
not evidence of semantic error awareness.
